"""Benchmark-only public-buffer PyZMQ adapter; not a supported Nahual backend."""

from __future__ import annotations

import json

import numpy as np


def send_array(socket, array):
    header = json.dumps(
        {"kind": "numpy", "dtype": array.dtype.str, "shape": array.shape}
    ).encode()
    tracker = socket.send_multipart([header, memoryview(array)], copy=False, track=True)
    return tracker


def decode_array(frames, header):
    result = np.frombuffer(frames[1].buffer, dtype=header["dtype"]).reshape(
        header["shape"]
    )
    result.flags.writeable = False
    return result


def serve(address, setup, processor, ready):
    import zmq

    with zmq.Context() as context, context.socket(zmq.REP) as socket:
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(address)
        ready()
        while True:
            frames = socket.recv_multipart(copy=False)
            header = json.loads(bytes(frames[0]))
            try:
                if header["kind"] == "setup":
                    processor, evidence = setup(**header["parameters"])
                    socket.send_json({"kind": "dict", "value": evidence})
                else:
                    array = decode_array(frames, header)
                    result = processor(array)
                    tracker = send_array(socket, result)
                    tracker.wait()  # keep all input/output owners until send completion
            except Exception as error:
                socket.send_json(
                    {"kind": "error", "error": f"{type(error).__name__}: {error}"}
                )


class Session:
    def __init__(self, address):
        import zmq

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.SNDTIMEO, 120_000)
        self._socket.setsockopt(zmq.RCVTIMEO, 120_000)
        self._socket.connect(address)

    def _receive(self):
        frames = self._socket.recv_multipart(copy=False)
        header = json.loads(bytes(frames[0]))
        if header["kind"] == "error":
            raise RuntimeError(header["error"])
        if header["kind"] == "dict":
            return header["value"]
        return decode_array(frames, header)

    def setup(self, parameters):
        self._socket.send_json({"kind": "setup", "parameters": parameters})
        return self._receive()

    def process(self, array):
        tracker = send_array(self._socket, array)
        result = self._receive()
        tracker.wait()
        return result

    def close(self):
        self._socket.close(linger=0)
        self._context.term()
