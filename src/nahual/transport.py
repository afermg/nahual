from __future__ import annotations

import json
import os

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
        # Server signals failure with a b"!" prefix + JSON body. Raise so the
        # caller sees a real error instead of a malformed numpy header.
        if response_bytes[:1] == b"!":
            info = json.loads(response_bytes[1:].decode())
            raise RuntimeError(
                f"nahual server error in stage '{info.get('stage')}': {info.get('error')}"
            )
        return response_bytes
