from __future__ import annotations

import threading

import appose
import numpy
import pynng
import pytest
from test_shared_memory import (
    assert_segments_unlinked,
    capture_segment_names,
    execed_server,
)

import nahual.shared_memory as protocol
from nahual.process import Client


@pytest.mark.parametrize(
    "mode", ["identity", "increment", "smaller", "alias", "larger"]
)
def test_successful_calls_reuse_segment_but_outputs_are_independent(mode, monkeypatch):
    names = capture_segment_names(monkeypatch)
    source = numpy.arange(12, dtype=numpy.float32).reshape(3, 4)
    with execed_server(mode) as address:
        with Client(address) as client:
            first = client.process(source, shared_memory=True)
            snapshot = first.copy()
            second = client.process(source + 10, shared_memory=True)
            assert first.flags.owndata and second.flags.owndata
            numpy.testing.assert_array_equal(first, snapshot)
            numpy.testing.assert_array_equal(second, first + 10)
            numpy.testing.assert_array_equal(source, numpy.arange(12).reshape(3, 4))
            assert len(names) == 1
    assert_segments_unlinked(names)


def test_appose_input_is_not_added_to_existing_cache(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address, Client(address, timeout_ms=-1) as client:
        client.process(numpy.arange(4, dtype=numpy.float32), shared_memory=True)
        with appose.NDArray("float32", [4]) as shared:
            shared.ndarray()[:] = 7
            output = client.process(shared, shared_memory=True)
            numpy.testing.assert_array_equal(output, 7)
            numpy.testing.assert_array_equal(shared.ndarray(), 7)
            assert len(names) == 1
            assert [segment.name for segment in client._pool._idle.values()] == names
    assert_segments_unlinked(names)


def test_inherited_client_cannot_reuse_or_unlink_parent_cache(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address, Client(address) as client:
        client.process(numpy.arange(4), shared_memory=True)
        with monkeypatch.context() as child:
            child.setattr("nahual.process.os.getpid", lambda: client._session._pid + 1)
            with pytest.raises(RuntimeError, match="fork"):
                client.process(numpy.arange(4), shared_memory=True)
            with pytest.raises(RuntimeError, match="fork"):
                client.close()
        # The parent can still use the same segment and close it normally.
        client.process(numpy.arange(4), shared_memory=True)
        assert len(names) == 1
    assert_segments_unlinked(names)


def test_cache_matches_byte_size_not_shape_or_dtype(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address, Client(address) as client:
        for data in [
            numpy.arange(12, dtype=numpy.float32).reshape(3, 4),
            numpy.arange(6, dtype=numpy.float64),
            numpy.arange(24, dtype=numpy.uint16)[::-1],
        ]:
            output = client.process(data, shared_memory=True)
            numpy.testing.assert_array_equal(output, data)
            assert output.dtype == data.dtype and output.shape == data.shape
        assert len(names) == 1
    assert_segments_unlinked(names)


def test_cache_byte_budget_uses_lru_eviction(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address:
        with Client(address, shared_memory_cache_bytes=6) as client:
            for size in [2, 3, 2, 4]:
                client.process(
                    numpy.arange(size, dtype=numpy.uint8), shared_memory=True
                )
            assert len(names) == 3
            assert list(client._pool._idle) == [2, 4]
            assert client._pool._cached_bytes == 6
            assert_segments_unlinked([names[1]])
    assert_segments_unlinked(names)


def test_cache_caps_number_of_idle_segments(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address, Client(address) as client:
        for size in range(1, 10):
            client.process(numpy.arange(size, dtype=numpy.uint8), shared_memory=True)
        assert len(client._pool._idle) == 8
        assert_segments_unlinked([names[0]])
    assert_segments_unlinked(names)


@pytest.mark.parametrize("budget", [0, 1])
def test_disabled_or_oversized_cache_never_retains_segments(budget, monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address:
        with Client(address, shared_memory_cache_bytes=budget) as client:
            for _ in range(2):
                client.process(numpy.arange(4), shared_memory=True)
                assert_segments_unlinked(names)
            assert len(names) == 2
    assert_segments_unlinked(names)


def test_empty_arrays_use_reusable_one_byte_allocation(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address, Client(address) as client:
        for shape in [(0,), (2, 0, 3)]:
            output = client.process(numpy.empty(shape), shared_memory=True)
            assert output.shape == shape and output.flags.owndata
        assert len(names) == 1
    assert_segments_unlinked(names)


@pytest.mark.parametrize("error", [pynng.Timeout, KeyboardInterrupt, RuntimeError])
def test_uncertain_exchange_discards_previously_cached_segment(error, monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address, Client(address) as client:
        data = numpy.arange(4)
        client.process(data, shared_memory=True)
        original = client._session.request

        def fail(*_args, **_kwargs):
            if error is pynng.Timeout:
                raise error("injected", 5)
            raise error("injected")

        monkeypatch.setattr(client._session, "request", fail)
        with pytest.raises(error, match="injected"):
            client.process(data, shared_memory=True)
        assert_segments_unlinked(names)
        assert client._pool._cached_bytes == 0
        monkeypatch.setattr(client._session, "request", original)
        numpy.testing.assert_array_equal(client.process(data, shared_memory=True), data)
        assert len(names) == 2
    assert_segments_unlinked(names)


def test_bad_response_token_is_never_cached(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address, Client(address) as client:
        original = client._session.request

        def corrupt(*args, **kwargs):
            response = original(*args, **kwargs)
            descriptor = protocol._decode_descriptor(
                response, protocol.RESPONSE_MAGIC, protocol._RESPONSE_PREFIX
            )
            descriptor["token"] = "0" * 32
            return protocol.RESPONSE_MAGIC + protocol._encode_descriptor(descriptor)

        monkeypatch.setattr(client._session, "request", corrupt)
        with pytest.raises(protocol.SharedMemoryProtocolError, match="token"):
            client.process(numpy.arange(4), shared_memory=True)
        assert_segments_unlinked(names)
        assert client._pool._cached_bytes == 0


@pytest.mark.timeout(15)
def test_timed_out_server_mapping_cannot_overwrite_next_call(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("sleep") as address, Client(address) as client:
        with pytest.raises(pynng.Timeout):
            client.process(numpy.array([1]), shared_memory=True, timeout_ms=100)
        assert_segments_unlinked(names)
        output = client.process(numpy.array([2]), shared_memory=True, timeout_ms=6000)
        numpy.testing.assert_array_equal(output, [2])
        assert len(names) == 2 and names[0] != names[1]
    assert_segments_unlinked(names)


def test_client_rejects_overlapping_entire_calls_and_close(monkeypatch):
    started = threading.Event()
    release = threading.Event()
    failures = []
    with Client("ipc:///tmp/not-contacted.ipc") as client:

        def blocked(*_args, **_kwargs):
            started.set()
            assert release.wait(5)
            raise RuntimeError("finished")

        monkeypatch.setattr(client._session, "request", blocked)

        def process():
            try:
                client.process(numpy.arange(4), shared_memory=True)
            except RuntimeError as error:
                failures.append(str(error))

        worker = threading.Thread(target=process)
        worker.start()
        try:
            assert started.wait(5)
            with pytest.raises(RuntimeError, match="in progress"):
                client.process(numpy.arange(4), shared_memory=True)
            with pytest.raises(RuntimeError, match="in progress"):
                client.close()
        finally:
            release.set()
            worker.join(5)
        assert not worker.is_alive()
        assert failures == ["finished"]
        assert client._pool._cached_bytes == 0


def test_close_attempts_all_cached_cleanup_even_after_error(monkeypatch):
    names = capture_segment_names(monkeypatch)
    with execed_server("identity") as address:
        client = Client(address)
        for size in [2, 3]:
            client.process(numpy.arange(size), shared_memory=True)
        original = protocol._cleanup_client_segment
        calls = []

        def fail_once(segment):
            calls.append(segment.name)
            original(segment)
            if len(calls) == 1:
                raise OSError("injected cleanup error")

        monkeypatch.setattr(protocol, "_cleanup_client_segment", fail_once)
        with pytest.raises(OSError, match="cleanup error"):
            client.close()
        assert calls == names
        client.close()
    assert_segments_unlinked(names)


@pytest.mark.parametrize("budget", [-1, True, 1.5, "64"])
def test_invalid_cache_budget_fails_before_contact(budget):
    with pytest.raises(ValueError, match="non-negative integer"):
        Client("ipc:///tmp/not-contacted.ipc", shared_memory_cache_bytes=budget)
