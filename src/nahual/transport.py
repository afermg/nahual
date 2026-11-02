from __future__ import annotations

import json
import os
import threading

import pynng

# Default IPC timeout, overridable via NAHUAL_IPC_TIMEOUT_MS. 300_000 ms = 5
# minutes -- well above any legitimate server latency (model load, large
# inference) but bounded so a stuck server can't deadlock the client.
_DEFAULT_TIMEOUT_MS = int(os.environ.get("NAHUAL_IPC_TIMEOUT_MS", "300000"))


def request_receive(
    packet: bytes, address: str, timeout_ms: int | None = None
) -> bytes:
    """Send a request and receive a response using pynng.

    Parameters
    ----------
    packet : bytes
        The data to send in the request.
    address : str, optional
        The endpoint address to connect to, such as "ipc:///tmp/reqrep.ipc".
    timeout_ms : int or None, optional
        Per-call send/recv timeout in milliseconds. Defaults to the value of
        the ``NAHUAL_IPC_TIMEOUT_MS`` environment variable, or 300000 (5 min)
        if unset. Set ``-1`` to wait indefinitely (legacy behaviour).

    Returns
    -------
    bytes
        The response data received from the server.

    Raises
    ------
    pynng.exceptions.Timeout
        If the request or response exceeds ``timeout_ms``.

    Notes
    ------
    While the server responder may be async, this method is desygned to run
    in one thread.

    """
    if timeout_ms is None:
        timeout_ms = _DEFAULT_TIMEOUT_MS
    with pynng.Req0(send_timeout=timeout_ms, recv_timeout=timeout_ms) as sock:
        sock.dial(address)
        print(f"REQ: SENDING {len(packet)} bytes to {address}")
        sock.send(packet)
        response = sock.recv_msg()
        response_bytes = response.bytes
        print(f"REQ: RECEIVED {len(response_bytes)} bytes")
        return _check_response(response_bytes)


def _check_response(response: bytes) -> bytes:
    if response[:1] == b"!":
        info = json.loads(response[1:].decode())
        raise RuntimeError(
            f"nahual server error in stage '{info.get('stage')}': {info.get('error')}"
        )
    return response


class RequestSession:
    """An explicitly owned, reusable synchronous NNG request connection.

    Use one session per worker. Overlapping requests/close operations and use
    after fork are rejected. A failed exchange discards the socket; the next
    call opens a fresh connection and never consumes the old reply. Requests
    are not automatically retried, including by NNG's REQ resend mechanism.
    A timeout does not cancel work already running on the server.
    """

    def __init__(self, address: str, timeout_ms: int | None = None):
        self._address = address
        self.timeout_ms = _DEFAULT_TIMEOUT_MS if timeout_ms is None else timeout_ms
        self._socket = None
        self._closed = False
        self._pid = os.getpid()
        self._lock = threading.Lock()

    @property
    def address(self):
        return self._address

    def _check_available(self):
        if os.getpid() != self._pid:
            raise RuntimeError("create a new RequestSession after fork")
        if self._closed:
            raise RuntimeError("RequestSession is closed")

    def _open(self, timeout_ms):
        if self._socket is None:
            sock = pynng.Req0(
                send_timeout=timeout_ms, recv_timeout=timeout_ms, resend_time=-1
            )
            try:
                sock.dial(self.address)
            except BaseException:
                sock.close()
                raise
            self._socket = sock
        self._socket.send_timeout = timeout_ms
        self._socket.recv_timeout = timeout_ms
        return self._socket

    def request(
        self, packet: bytes, address: str | None = None, timeout_ms: int | None = None
    ) -> bytes:
        self._check_available()
        if address is not None and address != self.address:
            raise ValueError("RequestSession is bound to a different address")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("RequestSession already has an operation in progress")
        try:
            self._check_available()
            effective_timeout = self.timeout_ms if timeout_ms is None else timeout_ms
            try:
                sock = self._open(effective_timeout)
                sock.send(packet)
                response = sock.recv_msg().bytes
            except BaseException:
                if self._socket is not None:
                    self._socket.close()
                    self._socket = None
                raise
            # An error envelope is a completed exchange, not a transport fault.
            return _check_response(response)
        finally:
            self._lock.release()

    def close(self):
        if os.getpid() != self._pid:
            raise RuntimeError("cannot close an inherited RequestSession after fork")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("RequestSession already has an operation in progress")
        try:
            if self._socket is not None:
                self._socket.close()
                self._socket = None
            self._closed = True
        finally:
            self._lock.release()

    def __enter__(self):
        self._check_available()
        return self

    def __exit__(self, *_exc):
        self.close()
