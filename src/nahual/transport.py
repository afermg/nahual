import pynng


def request_receive(packet: bytes, address: str) -> bytes:
    """Send a request and receive a response using pynng.

    Parameters
    ----------
    packet : bytes
        The data to send in the request.
    address : str, optional
        The endpoint address to connect to, such as "ipc:///tmp/reqrep.ipc".

    Returns
    -------
    bytes
        The response data received from the server.

    Raises
    ------
    pynng.exceptions.Timeout
        If the request times out.

    Notes
    ------
    While the server responder may be async, this method is desygned to run
    in one thread.

    """
    with pynng.Req0() as sock:
        sock.dial(address)
        print(f"REQ: SENDING {len(packet)} bytes to {address}")
        sock.send(packet)
        response = sock.recv_msg()
        response_bytes = response.bytes
        print(f"REQ: RECEIVED {len(response_bytes)} bytes")
        return response_bytes
