from __future__ import annotations

import threading

import appose
import numpy
import pynng
import pytest
from test_shared_memory import execed_server

from nahual.process import Client
from nahual.transport import RequestSession


def test_client_reuses_connection_across_setup_wire_and_shared_calls(monkeypatch):
    original = pynng.Req0
    opened = []

    def capture(**kwargs):
        opened.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(pynng, "Req0", capture)
    data = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    with execed_server("identity") as address:
        with Client(address) as client:
            assert client.setup({}) == {"mode": "identity"}
            for shared in [False, True, False, True]:
                numpy.testing.assert_array_equal(
                    client.process(data, shared_memory=shared), data
                )
            assert client.setup({}) == {"mode": "identity"}
            assert len(opened) == 1
            assert opened[0]["resend_time"] == -1
        client.close()  # idempotent
        with pytest.raises(RuntimeError, match="closed"):
            client.process(data)
        with pytest.raises(RuntimeError, match="closed"):
            client.__enter__()


def test_timeout_override_is_per_call():
    with execed_server("identity") as address:
        with Client(address, timeout_ms=10_000) as client:
            client.process(numpy.arange(4), timeout_ms=1234)
            assert client._session._socket.recv_timeout == 1234
            client.process(numpy.arange(4))
            assert client._session._socket.recv_timeout == 10_000


def test_appose_honors_client_unbounded_timeout():
    with execed_server("increment") as address:
        with appose.NDArray("float32", [2, 3]) as shared:
            shared.ndarray()[:] = 5
            with Client(address, timeout_ms=-1) as client:
                output = client.process(shared, shared_memory=True)
                assert output.flags.owndata
                numpy.testing.assert_array_equal(output, 6)
                numpy.testing.assert_array_equal(shared.ndarray(), 5)


def test_server_error_keeps_completed_connection():
    with execed_server("error") as address:
        with Client(address) as client:
            with pytest.raises(RuntimeError, match="injected"):
                client.process(numpy.arange(4))
            socket = client._session._socket
            with pytest.raises(RuntimeError, match="injected"):
                client.process(numpy.arange(4))
            assert client._session._socket is socket


@pytest.mark.timeout(15)
def test_timeout_discards_socket_and_does_not_consume_stale_reply():
    with execed_server("sleep") as address:
        with Client(address) as client:
            with pytest.raises(pynng.Timeout):
                client.process(numpy.array([1]), timeout_ms=100)
            assert client._session._socket is None
            output = client.process(numpy.array([2]), timeout_ms=6000)
            numpy.testing.assert_array_equal(output, [2])


def test_session_rejects_other_addresses_and_inherited_use(monkeypatch):
    session = RequestSession("ipc:///tmp/not-contacted.ipc")
    with pytest.raises(ValueError, match="different address"):
        session.request(b"x", address="ipc:///tmp/another.ipc")
    monkeypatch.setattr("nahual.transport.os.getpid", lambda: session._pid + 1)
    with pytest.raises(RuntimeError, match="fork"):
        session.request(b"x")
    with pytest.raises(RuntimeError, match="fork"):
        session.close()


def test_overlapping_request_and_close_are_rejected(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    failures = []

    class BlockingSocket:
        def __init__(self, **_kwargs):
            pass

        def dial(self, _address):
            pass

        def send(self, _packet):
            started.set()
            assert release.wait(5)

        def recv_msg(self):
            return type("Reply", (), {"bytes": b"reply"})()

        def close(self):
            pass

    monkeypatch.setattr(pynng, "Req0", BlockingSocket)
    with RequestSession("ipc:///tmp/fake.ipc") as session:

        def request():
            try:
                assert session.request(b"first") == b"reply"
            except BaseException as error:
                failures.append(error)

        worker = threading.Thread(target=request)
        worker.start()
        try:
            assert started.wait(5)
            with pytest.raises(RuntimeError, match="in progress"):
                session.request(b"second")
            with pytest.raises(RuntimeError, match="in progress"):
                session.close()
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()
        assert failures == []


@pytest.mark.parametrize("signature", [("numpy",), ("dict", "bad"), "numpy"])
def test_invalid_signature(signature):
    with pytest.raises(ValueError, match="signature"):
        Client("ipc:///tmp/not-contacted.ipc", signature=signature)


def test_client_rejects_misrouted_setup_and_shared_dict_output():
    with Client("ipc:///tmp/not-contacted.ipc", signature=("dict", "dict")) as client:
        with pytest.raises(TypeError, match="parameter dictionary"):
            client.setup(numpy.arange(4))
        with pytest.raises(TypeError, match="setup dictionaries"):
            client.process({})
        with pytest.raises(TypeError, match="NumPy output"):
            client.process(numpy.arange(4), shared_memory=True)
