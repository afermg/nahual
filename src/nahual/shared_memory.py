"""Local shared-memory data plane for NumPy process calls.

The control plane remains pynng. Ordinary NumPy calls use a client-owned segment
for one request. Optional Appose inputs are externally owned, input-only
borrows. The server only attaches to either kind. Processor inputs are
read-only views and must not be retained beyond the synchronous call; that
prohibition is not enforceable and retained views may become unsafe.
"""

from __future__ import annotations

import inspect
import json
import math
import mmap
import os
import re
import secrets
from typing import Any, Callable

import numpy

from nahual.serial import serialize_numpy
from nahual.transport import request_receive

_PROTOCOL_PREFIX = b"\x00NHSM"
REQUEST_MAGIC = _PROTOCOL_PREFIX + b"\x01"
_RESPONSE_PREFIX = b"\x00NHSR"
RESPONSE_MAGIC = _RESPONSE_PREFIX + b"\x01"
MAX_DESCRIPTOR_BYTES = 16 * 1024
MAX_NDIM = 32
MAX_CAPACITY = (1 << 63) - 1
_NAME_RE = re.compile(r"[A-Za-z0-9_-]{1,128}\Z")
_TOKEN_RE = re.compile(r"[0-9a-f]{32}\Z")
_APPOSE_VERSION_RE = re.compile(r"0\.12(?:\..*)?\Z")
_OUTPUT_MODES = {"reuse", "serialized"}


class SharedMemoryProtocolError(ValueError):
    """Raised when a local shared-memory envelope is invalid."""


def is_shared_request(payload: bytes) -> bool:
    """Return whether payload belongs to the shared-memory protocol family."""
    return payload.startswith(_PROTOCOL_PREFIX)


def require_ipc_address(address: str) -> None:
    """Require the exact local transport supported by shared memory."""
    if (
        not isinstance(address, str)
        or not address.startswith("ipc://")
        or len(address) <= 6
    ):
        raise ValueError("shared_memory=True requires an exact local ipc:// address")


def _reject_json_constant(value: str) -> None:
    raise SharedMemoryProtocolError(f"invalid JSON constant {value!r}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SharedMemoryProtocolError(f"duplicate descriptor field {key!r}")
        result[key] = value
    return result


def _encode_descriptor(descriptor: dict[str, Any]) -> bytes:
    body = json.dumps(descriptor, separators=(",", ":"), sort_keys=True).encode("utf-8")
    if len(body) > MAX_DESCRIPTOR_BYTES:
        raise SharedMemoryProtocolError("shared-memory descriptor is too large")
    return body


def _decode_descriptor(payload: bytes, magic: bytes, prefix: bytes) -> dict[str, Any]:
    if payload.startswith(prefix) and not payload.startswith(magic):
        raise SharedMemoryProtocolError("unsupported shared-memory protocol version")
    if not payload.startswith(magic):
        raise SharedMemoryProtocolError("invalid shared-memory protocol marker")
    body = payload[len(magic) :]
    if not body or len(body) > MAX_DESCRIPTOR_BYTES:
        raise SharedMemoryProtocolError("invalid shared-memory descriptor size")
    try:
        value = json.loads(
            body.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SharedMemoryProtocolError(
            "malformed shared-memory descriptor JSON"
        ) from error
    if not isinstance(value, dict):
        raise SharedMemoryProtocolError(
            "shared-memory descriptor must be a JSON object"
        )
    return value


def _dtype_string(dtype: numpy.dtype) -> str:
    dtype = numpy.dtype(dtype)
    if (
        dtype.fields is not None
        or dtype.subdtype is not None
        or dtype.hasobject
        or dtype.kind not in "biufc"
        or dtype.itemsize <= 0
        or not dtype.isnative
    ):
        raise TypeError(
            "shared memory v1 supports only native fixed-size bool, integer, "
            "unsigned, float, and complex NumPy dtypes"
        )
    return dtype.str


def _parse_dtype(value: Any) -> numpy.dtype:
    if not isinstance(value, str) or len(value) > 32:
        raise SharedMemoryProtocolError("descriptor dtype must be a bounded string")
    try:
        dtype = numpy.dtype(value)
        canonical = _dtype_string(dtype)
    except (TypeError, ValueError) as error:
        raise SharedMemoryProtocolError("descriptor dtype is not supported") from error
    if value != canonical:
        raise SharedMemoryProtocolError("descriptor dtype is not canonical dtype.str")
    return dtype


def _parse_shape(value: Any) -> tuple[int, ...]:
    if not isinstance(value, list) or len(value) > MAX_NDIM:
        raise SharedMemoryProtocolError(
            f"descriptor shape must have at most {MAX_NDIM} axes"
        )
    shape = []
    for dimension in value:
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension < 0
        ):
            raise SharedMemoryProtocolError(
                "descriptor shape dimensions must be non-negative integers"
            )
        if dimension > MAX_CAPACITY:
            raise SharedMemoryProtocolError("descriptor shape dimension is too large")
        shape.append(dimension)
    return tuple(shape)


def _expected_nbytes(shape: tuple[int, ...], dtype: numpy.dtype) -> int:
    count = math.prod(shape)
    nbytes = count * dtype.itemsize
    if nbytes > MAX_CAPACITY:
        raise SharedMemoryProtocolError("descriptor array byte count is too large")
    return nbytes


def _parse_nonnegative_int(value: Any, field: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= MAX_CAPACITY
    ):
        raise SharedMemoryProtocolError(
            f"descriptor {field} must be a bounded non-negative integer"
        )
    return value


def _create_segment(size: int):
    from multiprocessing import shared_memory

    return shared_memory.SharedMemory(create=True, size=size)


class _PosixSharedMemoryAttachment:
    """Minimal untracked attachment for CPython 3.9-3.12 on POSIX.

    This experimental compatibility wrapper deliberately bypasses
    ``SharedMemory`` so attaching never mutates the process resource tracker.
    It depends only on the isolated private ``_posixshmem.shm_open`` entry
    point and otherwise uses public OS primitives.
    """

    def __init__(
        self, shm_open: Callable[..., int], name: str, *, read_only: bool = False
    ):
        self._fd = -1
        self._mmap: mmap.mmap | None = None
        self._buf: memoryview | None = None
        self.size = 0
        flags = (os.O_RDONLY if read_only else os.O_RDWR) | getattr(os, "O_CLOEXEC", 0)
        try:
            self._fd = shm_open(name, flags, mode=0o600)
            self.size = os.fstat(self._fd).st_size
            access = mmap.ACCESS_READ if read_only else mmap.ACCESS_DEFAULT
            self._mmap = mmap.mmap(self._fd, self.size, access=access)
            self._buf = memoryview(self._mmap)
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    @property
    def buf(self) -> memoryview:
        if self._buf is None:
            raise ValueError("the shared-memory attachment is closed")
        return self._buf

    def close(self) -> None:
        """Release all local resources, retry-safely and without unlinking."""
        error: BaseException | None = None
        if self._buf is not None:
            try:
                self._buf.release()
            except BaseException as caught:
                error = caught
            finally:
                self._buf = None
        if self._mmap is not None:
            try:
                self._mmap.close()
            except BaseException as caught:
                if error is None:
                    error = caught
            else:
                self._mmap = None
        if self._fd >= 0:
            try:
                os.close(self._fd)
            except BaseException as caught:
                if error is None:
                    error = caught
            finally:
                self._fd = -1
        if error is not None:
            raise error


def _attach_segment(name: str, *, read_only: bool = False):
    """Attach without registering server ownership of client memory."""
    from multiprocessing import shared_memory

    try:
        supports_track = (
            "track" in inspect.signature(shared_memory.SharedMemory).parameters
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "cannot determine safe shared-memory attachment support on this runtime"
        ) from error
    if supports_track and not read_only:
        return shared_memory.SharedMemory(name=name, create=False, track=False)

    if os.name != "posix":
        detail = "read-only" if read_only else "untracked"
        raise RuntimeError(
            f"{detail} shared-memory attachment requires POSIX, or Python 3.13 "
            "track=False for writable attachments"
        )

    # Match SharedMemory's public-name normalization exactly. These private
    # CPython POSIX details are isolated here for safe untracked attachments.
    try:
        if not shared_memory._USE_POSIX:
            raise AttributeError("_USE_POSIX is false")
        shm_open = shared_memory._posixshmem.shm_open
        prepend_slash = shared_memory.SharedMemory._prepend_leading_slash
    except AttributeError as error:
        raise RuntimeError(
            "untracked POSIX shared-memory attachment is unsupported because "
            "required CPython internals are unavailable"
        ) from error
    normalized_name = f"/{name}" if prepend_slash else name
    return _PosixSharedMemoryAttachment(shm_open, normalized_name, read_only=read_only)


def _request_descriptor(
    array: numpy.ndarray, segment: Any, token: str, output_mode: str
) -> dict[str, Any]:
    return {
        "capacity": array.nbytes,
        "dtype": _dtype_string(array.dtype),
        "name": segment.name,
        "nbytes": array.nbytes,
        "output_mode": output_mode,
        "shape": list(array.shape),
        "token": token,
    }


def _validate_request(
    descriptor: dict[str, Any],
) -> tuple[str, str, numpy.dtype, tuple[int, ...], int, int, str]:
    required = {
        "capacity",
        "dtype",
        "name",
        "nbytes",
        "output_mode",
        "shape",
        "token",
    }
    if set(descriptor) != required:
        raise SharedMemoryProtocolError(
            "shared-memory request fields do not match protocol v1"
        )
    name = descriptor["name"]
    token = descriptor["token"]
    if not isinstance(name, str) or _NAME_RE.fullmatch(name) is None:
        raise SharedMemoryProtocolError("invalid shared-memory segment name")
    if not isinstance(token, str) or _TOKEN_RE.fullmatch(token) is None:
        raise SharedMemoryProtocolError("invalid shared-memory request token")
    dtype = _parse_dtype(descriptor["dtype"])
    shape = _parse_shape(descriptor["shape"])
    nbytes = _parse_nonnegative_int(descriptor["nbytes"], "nbytes")
    capacity = _parse_nonnegative_int(descriptor["capacity"], "capacity")
    output_mode = descriptor["output_mode"]
    if not isinstance(output_mode, str) or output_mode not in _OUTPUT_MODES:
        raise SharedMemoryProtocolError("unsupported shared-memory output mode")
    if nbytes != _expected_nbytes(shape, dtype) or capacity != nbytes:
        raise SharedMemoryProtocolError(
            "request shape, dtype, nbytes, and capacity disagree"
        )
    return name, token, dtype, shape, nbytes, capacity, output_mode


def _response_descriptor(result: numpy.ndarray, token: str) -> bytes:
    descriptor = {
        "dtype": _dtype_string(result.dtype),
        "nbytes": result.nbytes,
        "shape": list(result.shape),
        "token": token,
    }
    return RESPONSE_MAGIC + _encode_descriptor(descriptor)


def _parse_response(
    payload: bytes, token: str, capacity: int, segment: Any
) -> numpy.ndarray:
    descriptor = _decode_descriptor(payload, RESPONSE_MAGIC, _RESPONSE_PREFIX)
    if set(descriptor) != {"dtype", "nbytes", "shape", "token"}:
        raise SharedMemoryProtocolError(
            "shared-memory response fields do not match protocol v1"
        )
    if descriptor["token"] != token:
        raise SharedMemoryProtocolError(
            "shared-memory response token does not match request"
        )
    dtype = _parse_dtype(descriptor["dtype"])
    shape = _parse_shape(descriptor["shape"])
    nbytes = _parse_nonnegative_int(descriptor["nbytes"], "nbytes")
    if nbytes != _expected_nbytes(shape, dtype) or nbytes > capacity:
        raise SharedMemoryProtocolError(
            "shared-memory response metadata exceeds request capacity"
        )
    if segment.size != max(1, capacity):
        raise SharedMemoryProtocolError(
            "shared-memory segment size changed during request"
        )
    view = numpy.ndarray(shape, dtype=dtype, buffer=segment.buf, order="C")
    try:
        return view.copy(order="C")
    finally:
        del view


def _cleanup_client_segment(segment: Any) -> None:
    error: BaseException | None = None
    try:
        segment.close()
    except BaseException as caught:
        error = caught
    try:
        segment.unlink()
    except FileNotFoundError:
        pass
    except BaseException as caught:
        if error is None:
            error = caught
    if error is not None:
        raise error


def _send_shared_request(
    array: numpy.ndarray,
    segment: Any,
    address: str,
    output_mode: str,
    timeout_ms: int | None,
) -> numpy.ndarray:
    token = secrets.token_hex(16)
    descriptor = _request_descriptor(array, segment, token, output_mode)
    packet = REQUEST_MAGIC + _encode_descriptor(descriptor)
    response = request_receive(packet, address=address, timeout_ms=timeout_ms)
    if response.startswith(_RESPONSE_PREFIX):
        if output_mode != "reuse":
            raise SharedMemoryProtocolError(
                "server attempted to reuse a borrowed shared-memory input"
            )
        return _parse_response(response, token, array.nbytes, segment)
    if response.startswith(b"\x00"):
        raise SharedMemoryProtocolError("invalid response to shared-memory request")

    # Serialized mode is mandatory for Appose-owned input. Internal requests
    # also use this lossless fallback when an output cannot fit in the segment.
    from nahual.serial import deserialize_numpy

    return deserialize_numpy(response).copy(order="C")


def _client_numpy(
    array: numpy.ndarray, address: str, timeout_ms: int | None
) -> numpy.ndarray:
    _dtype_string(array.dtype)
    capacity = array.nbytes
    segment = _create_segment(max(1, capacity))
    primary_error: BaseException | None = None
    staged = None
    try:
        staged = numpy.ndarray(
            array.shape, dtype=array.dtype, buffer=segment.buf, order="C"
        )
        numpy.copyto(staged, array)
        return _send_shared_request(array, segment, address, "reuse", timeout_ms)
    except BaseException as error:
        primary_error = error
        raise
    finally:
        if staged is not None:
            del staged
        try:
            _cleanup_client_segment(segment)
        except BaseException:
            if primary_error is None:
                raise


def _require_appose_ndarray(value: Any):
    try:
        import appose
    except ModuleNotFoundError as error:
        if error.name != "appose":
            raise
        raise ImportError(
            "Appose shared-memory input requires the optional dependency; "
            "install it with `pip install 'nahual[appose]'`"
        ) from error

    version = getattr(appose, "__version__", "")
    if _APPOSE_VERSION_RE.fullmatch(version) is None:
        raise RuntimeError(
            "Appose shared-memory input requires appose>=0.12.0,<0.13; "
            f"found {version or 'an unknown version'}"
        )
    if not isinstance(value, appose.NDArray):
        raise TypeError(
            "shared_memory=True requires a NumPy ndarray or appose.NDArray input"
        )
    return value


def _client_appose(value: Any, address: str, timeout_ms: int | None) -> numpy.ndarray:
    array = _require_appose_ndarray(value)
    if timeout_ms != -1:
        raise ValueError(
            "Appose-owned shared-memory input requires timeout_ms=-1 so its "
            "exclusive borrow cannot outlive a timed-out request"
        )

    view = array.ndarray()
    if not isinstance(view, numpy.ndarray) or not view.flags.c_contiguous:
        raise TypeError("appose.NDArray must expose a C-contiguous NumPy view")
    if tuple(array.shape) != view.shape or numpy.dtype(array.dtype) != view.dtype:
        raise ValueError("appose.NDArray metadata does not match its NumPy view")
    _dtype_string(view.dtype)
    if view.nbytes == 0:
        raise ValueError("empty appose.NDArray inputs are not supported")

    # The caller owns this allocation. Nahual only borrows it synchronously:
    # no staging, close, unlink, disposal, or server-side output overwrite.
    return _send_shared_request(view, array.shm, address, "serialized", timeout_ms)


def client_process(
    value: Any, address: str, timeout_ms: int | None = None
) -> numpy.ndarray:
    """Execute one local process call through the shared-memory data plane."""
    require_ipc_address(address)
    if isinstance(value, numpy.ndarray):
        return _client_numpy(value, address, timeout_ms)
    if isinstance(value, (dict, list, tuple)):
        raise TypeError(
            "shared_memory=True requires a NumPy ndarray or appose.NDArray input"
        )
    return _client_appose(value, address, timeout_ms)


def _eligible_output(result: numpy.ndarray, capacity: int) -> bool:
    try:
        _dtype_string(result.dtype)
    except TypeError:
        return False
    return result.ndim <= MAX_NDIM and result.nbytes <= capacity


def _legacy_output_eligible(result: numpy.ndarray) -> bool:
    """Return whether the one-character legacy wire is lossless for result."""
    try:
        _dtype_string(result.dtype)
        char_dtype = numpy.dtype(result.dtype.char)
    except (TypeError, ValueError):
        return False
    return (
        char_dtype == result.dtype
        and result.ndim <= 255
        and all(dimension <= 65535 for dimension in result.shape)
    )


def _prepare_shared_output(
    result: numpy.ndarray, input_view: numpy.ndarray
) -> numpy.ndarray:
    exact_input = (
        result is input_view
        and result.dtype == input_view.dtype
        and result.shape == input_view.shape
        and result.strides == input_view.strides
        and result.flags.c_contiguous
    )
    if exact_input:
        return result
    if numpy.shares_memory(result, input_view):
        return result.copy(order="C")
    return numpy.ascontiguousarray(result)


def handle_server_request(payload: bytes, pipe_url: str, processor: Callable) -> bytes:
    """Attach, process synchronously, close, and build the control response."""
    require_ipc_address(pipe_url)
    descriptor = _decode_descriptor(payload, REQUEST_MAGIC, _PROTOCOL_PREFIX)
    name, token, dtype, shape, nbytes, capacity, output_mode = _validate_request(
        descriptor
    )

    segment = _attach_segment(name, read_only=output_mode == "serialized")
    input_view = None
    result = None
    prepared = None
    try:
        if segment.size != max(1, capacity):
            raise SharedMemoryProtocolError(
                "descriptor capacity does not match actual segment size"
            )
        input_view = numpy.ndarray(shape, dtype=dtype, buffer=segment.buf, order="C")
        if input_view.nbytes != nbytes:
            raise SharedMemoryProtocolError(
                "descriptor byte count does not match input view"
            )
        input_view.flags.writeable = False
        result = processor(input_view)
        if not isinstance(result, numpy.ndarray):
            result = result.cpu().detach().numpy()
        if output_mode == "serialized":
            if _legacy_output_eligible(result):
                return serialize_numpy(result)
            raise TypeError(
                "Appose shared-memory output cannot be represented safely by "
                f"the NumPy wire format: dtype={result.dtype!s}, "
                f"shape={result.shape!r}"
            )
        if _eligible_output(result, capacity):
            prepared = _prepare_shared_output(result, input_view)
            if prepared is not input_view:
                destination = numpy.ndarray(
                    prepared.shape, dtype=prepared.dtype, buffer=segment.buf, order="C"
                )
                try:
                    numpy.copyto(destination, prepared)
                finally:
                    del destination
            return _response_descriptor(prepared, token)
        if _legacy_output_eligible(result):
            # A losslessly representable output that cannot use the request
            # segment (normally because it is larger) may use the legacy wire.
            return serialize_numpy(result)
        raise TypeError(
            "shared-memory output cannot be represented safely by protocol v1 "
            f"or the legacy NumPy wire format: dtype={result.dtype!s}, "
            f"shape={result.shape!r}"
        )
    finally:
        # Retaining a borrowed processor input (or a view derived from it) is
        # forbidden but cannot be enforced reliably. close() may still succeed,
        # leaving retained views dangling and unsafe after this request.
        del prepared
        del result
        del input_view
        segment.close()
