from __future__ import annotations

import builtins
import os
import select
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import appose
import numpy
import pynng
import pytest
import trio

from nahual.process import dispatch_setup_process, send_receive_process
from nahual.serial import serialize_numpy
from nahual.server import responder
from nahual.shared_memory import client_process

SERVER = Path(__file__).with_name("shared_memory_server.py")


@contextmanager
def execed_server(mode: str):
    with tempfile.TemporaryDirectory(prefix="nahual-shm-test-") as directory:
        address = f"ipc://{Path(directory) / 'server.ipc'}"
        server = subprocess.Popen(
            [sys.executable, str(SERVER), address, mode],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert server.stdout is not None
        ready, _, _ = select.select([server.stdout], [], [], 10)
        if not ready or server.stdout.readline().strip() != "READY":
            stdout, stderr = server.communicate(timeout=5)
            raise RuntimeError(f"test server did not start: {stdout}\n{stderr}")
        try:
            yield address
        finally:
            server.terminate()
            try:
                stdout, stderr = server.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                stdout, stderr = server.communicate(timeout=5)
            assert "resource_tracker" not in stderr, f"{stdout}\n{stderr}"


def process_callable(output: str = "numpy"):
    return dispatch_setup_process("test", signature=("dict", output))[1]


def capture_segment_names(monkeypatch):
    import nahual.shared_memory as protocol

    names = []
    original = protocol._create_segment

    def capture(size):
        segment = original(size)
        names.append(segment.name)
        return segment

    monkeypatch.setattr(protocol, "_create_segment", capture)
    return names


def assert_segments_unlinked(names):
    from multiprocessing import shared_memory

    for name in names:
        with pytest.raises(FileNotFoundError):
            shared_memory.SharedMemory(name=name, create=False)


@pytest.mark.parametrize(
    "source",
    [
        numpy.arange(24, dtype=numpy.float32).reshape(4, 6),
        numpy.arange(30, dtype=numpy.float32).reshape(5, 6)[:, ::2],
        numpy.asfortranarray(numpy.arange(24, dtype=numpy.int16).reshape(4, 6)),
        numpy.array(7, dtype=numpy.int64),
        numpy.empty((2, 0, 3), dtype=numpy.float64),
    ],
    ids=["contiguous", "strided", "fortran", "scalar", "empty"],
)
def test_identity_stages_layouts_returns_owner_and_cleans_up(source, monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address:
        output = process_callable()(source, address=address, shared_memory=True)

    numpy.testing.assert_array_equal(output, source)
    assert output.dtype == source.dtype
    assert output.shape == source.shape
    assert output.flags.owndata
    assert_segments_unlinked(names)


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("identity", lambda x: x),
        ("smaller", lambda x: x.reshape(-1)[: x.size // 2]),
        ("alias", lambda x: x.reshape(-1)[1:]),
        ("larger", lambda x: numpy.concatenate((x.reshape(-1), x.reshape(-1)))),
    ],
)
def test_same_smaller_alias_and_larger_outputs(mode, expected):
    source = numpy.arange(12, dtype=numpy.int32).reshape(3, 4)
    with execed_server(mode) as address:
        output = process_callable()(source, address=address, shared_memory=True)

    numpy.testing.assert_array_equal(output, expected(source))
    assert output.flags.owndata


def test_default_and_shared_requests_interleave_after_setup(monkeypatch):
    names = capture_segment_names(monkeypatch)
    setup, process = dispatch_setup_process("test", signature=("dict", "numpy"))
    source = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)

    with execed_server("identity") as address:
        assert setup({}, address=address) == {"mode": "identity"}
        legacy_before = process(source, address=address)
        shared = process(source, address=address, shared_memory=True)
        legacy_after = process(source + 1, address=address)
        shared_again = process(source + 1, address=address, shared_memory=True)

    numpy.testing.assert_array_equal(legacy_before, source)
    numpy.testing.assert_array_equal(shared, source)
    numpy.testing.assert_array_equal(legacy_after, source + 1)
    numpy.testing.assert_array_equal(shared_again, source + 1)
    assert_segments_unlinked(names)


def test_responder_materializes_each_default_message_once():
    class StopResponder(BaseException):
        pass

    class CountedMessage:
        def __init__(self, payload):
            self.payload = payload
            self.accesses = 0

        @property
        def bytes(self):
            self.accesses += 1
            return self.payload

    class OneMessageSocket:
        def __init__(self, message):
            self.message = message
            self.received = False
            self.responses = []

        async def arecv_msg(self):
            if self.received:
                raise StopResponder
            self.received = True
            return self.message

        async def asend(self, response):
            self.responses.append(response)

    source = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    message = CountedMessage(serialize_numpy(source))
    socket = OneMessageSocket(message)

    async def exercise():
        try:
            await responder(socket, setup=lambda: None, processor=lambda array: array)
        except StopResponder:
            pass

    trio.run(exercise)

    assert message.accesses == 1
    assert len(socket.responses) == 1


def test_lossy_output_errors_and_client_unlinks(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("non-native-output") as address:
        with pytest.raises(RuntimeError, match="cannot be represented safely"):
            process_callable()(
                numpy.arange(8, dtype=numpy.int32),
                address=address,
                shared_memory=True,
            )
    assert_segments_unlinked(names)


def test_server_error_unlinks_segment(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("error") as address:
        with pytest.raises(
            RuntimeError, match="injected shared-memory processor error"
        ):
            process_callable()(
                numpy.arange(8, dtype=numpy.uint16),
                address=address,
                shared_memory=True,
            )
    assert_segments_unlinked(names)


def test_timeout_unlinks_internal_segment(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("sleep") as address:
        with pytest.raises(pynng.Timeout):
            process_callable()(
                numpy.arange(8, dtype=numpy.float32),
                address=address,
                shared_memory=True,
                timeout_ms=100,
            )
    assert_segments_unlinked(names)


@pytest.mark.skipif(
    os.name != "posix" or sys.version_info >= (3, 13),
    reason="low-level compatibility applies to POSIX Python 3.9-3.12",
)
def test_low_level_attachment_does_not_touch_resource_tracker(monkeypatch):
    from multiprocessing import resource_tracker, shared_memory

    import nahual.shared_memory as protocol

    owner = shared_memory.SharedMemory(create=True, size=8)
    tracker_calls = []
    original_unregister = resource_tracker.unregister
    monkeypatch.setattr(
        resource_tracker,
        "register",
        lambda *_args: pytest.fail("attachment must not register"),
    )
    monkeypatch.setattr(
        resource_tracker, "unregister", lambda *args: tracker_calls.append(args)
    )
    try:
        attachment = protocol._attach_segment(owner.name)
        attachment.buf[:] = b"attached"
        assert bytes(owner.buf) == b"attached"
        attachment.close()
        attachment.close()
        assert tracker_calls == []
    finally:
        owner.close()
        monkeypatch.setattr(resource_tracker, "unregister", original_unregister)
        owner.unlink()


@pytest.mark.skipif(sys.version_info < (3, 13), reason="requires Python 3.13")
def test_python_313_attachment_uses_track_false(monkeypatch):
    from multiprocessing import shared_memory

    import nahual.shared_memory as protocol

    owner = shared_memory.SharedMemory(create=True, size=1)
    original = shared_memory.SharedMemory
    calls = []

    def capture(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(shared_memory, "SharedMemory", capture)
    monkeypatch.setattr(
        protocol.inspect,
        "signature",
        lambda _callable: protocol.inspect.Signature(
            [
                protocol.inspect.Parameter(
                    "track", protocol.inspect.Parameter.KEYWORD_ONLY
                )
            ]
        ),
    )
    try:
        attachment = protocol._attach_segment(owner.name)
        attachment.close()
    finally:
        owner.close()
        owner.unlink()

    assert calls == [{"name": owner.name, "create": False, "track": False}]


def test_shared_memory_opt_in_is_numpy_process_only_and_ipc_only(monkeypatch):
    import nahual.shared_memory as protocol

    monkeypatch.setattr(
        protocol,
        "_create_segment",
        lambda _size: pytest.fail("invalid calls must fail before allocation"),
    )
    with pytest.raises(ValueError, match="ipc://"):
        client_process(numpy.arange(2), "tcp://127.0.0.1:1234")
    with pytest.raises(TypeError, match="process-only"):
        send_receive_process({}, "dict", "ipc:///tmp/x", shared_memory=True)
    with pytest.raises(TypeError, match="NumPy output"):
        send_receive_process(
            numpy.arange(2), "dict", "ipc:///tmp/x", shared_memory=True
        )
    with pytest.raises(TypeError, match="NumPy ndarray or appose.NDArray"):
        send_receive_process([1, 2], "numpy", "ipc:///tmp/x", shared_memory=True)


def test_default_imports_do_not_load_appose():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import nahual.process, nahual.server; "
            "assert 'appose' not in sys.modules",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_missing_appose_extra_has_actionable_error(monkeypatch):
    original_import = builtins.__import__

    def without_appose(name, *args, **kwargs):
        if name == "appose":
            raise ModuleNotFoundError("test simulates missing appose", name="appose")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_appose)
    with pytest.raises(ImportError, match=r"nahual\[appose\]"):
        client_process(object(), "ipc:///tmp/not-contacted.ipc", timeout_ms=-1)


def test_appose_requires_explicit_unbounded_timeout_before_contact():
    with appose.NDArray("float32", [2, 3]) as shared:
        with pytest.raises(ValueError, match="timeout_ms=-1"):
            client_process(shared, "ipc:///tmp/not-contacted.ipc")
        with pytest.raises(ValueError, match="timeout_ms=-1"):
            client_process(shared, "ipc:///tmp/not-contacted.ipc", timeout_ms=100)


def test_appose_direct_input_is_borrowed_unchanged_and_output_owning(monkeypatch):
    import nahual.shared_memory as protocol

    monkeypatch.setattr(
        protocol,
        "_create_segment",
        lambda _size: pytest.fail("Appose input must not allocate a staging segment"),
    )
    source = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    with appose.NDArray("float32", list(source.shape)) as shared:
        shared.ndarray()[:] = source
        name = shared.shm.name
        with execed_server("increment") as address:
            output = process_callable()(
                shared,
                address=address,
                shared_memory=True,
                timeout_ms=-1,
            )

        numpy.testing.assert_array_equal(shared.ndarray(), source)
        numpy.testing.assert_array_equal(output, source + 1)
        assert output.flags.owndata
        assert shared.shm.name == name

        with execed_server("identity") as address:
            identity = process_callable()(
                shared,
                address=address,
                shared_memory=True,
                timeout_ms=-1,
            )
        numpy.testing.assert_array_equal(identity, source)
        assert identity.flags.owndata

        # Serialized mode is attached read-only, so even a processor that tries
        # to re-enable writes cannot mutate the externally owned input.
        with execed_server("mutate") as address:
            with pytest.raises(RuntimeError, match="WRITEABLE"):
                process_callable()(
                    shared,
                    address=address,
                    shared_memory=True,
                    timeout_ms=-1,
                )
        numpy.testing.assert_array_equal(shared.ndarray(), source)

    from multiprocessing import shared_memory

    with pytest.raises(FileNotFoundError):
        shared_memory.SharedMemory(name=name, create=False)


def test_descriptor_requires_known_output_mode_and_version():
    import nahual.shared_memory as protocol

    valid = {
        "capacity": 8,
        "dtype": numpy.dtype("float32").str,
        "name": "psm_01234567",
        "nbytes": 8,
        "output_mode": "reuse",
        "shape": [2],
        "token": "a" * 32,
    }
    protocol._validate_request(valid)

    missing = valid.copy()
    missing.pop("output_mode")
    with pytest.raises(protocol.SharedMemoryProtocolError, match="fields"):
        protocol._validate_request(missing)

    for invalid_mode in ("overwrite-borrowed", [], {}):
        unknown = {**valid, "output_mode": invalid_mode}
        with pytest.raises(protocol.SharedMemoryProtocolError, match="output mode"):
            protocol._validate_request(unknown)

    wrong_version = protocol._PROTOCOL_PREFIX + b"\x02{}"
    with pytest.raises(protocol.SharedMemoryProtocolError, match="version"):
        protocol._decode_descriptor(
            wrong_version, protocol.REQUEST_MAGIC, protocol._PROTOCOL_PREFIX
        )
