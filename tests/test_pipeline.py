from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
import runpy
import sys
import threading
import time
import weakref

import numpy as np
import pytest
from test_shared_memory import (
    assert_segments_unlinked,
    capture_segment_names,
    execed_server,
)

from nahual import pipeline
from nahual.pipeline import PipelineError, process_map


@pytest.fixture
def fake_clients(monkeypatch):
    clients = []

    class FakeClient:
        def __init__(self, address, **options):
            self.address, self.options = address, options
            self.active = False
            self.closed = False
            self.calls = []
            clients.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            assert not self.active
            self.closed = True

        def setup(self, parameters):
            if parameters.get("fail"):
                raise ValueError("setup failed")
            return {"address": self.address}

        def process(self, data, **kwargs):
            assert not self.active and not self.closed
            self.active = True
            try:
                self.calls.append(int(data[0]))
                time.sleep(0.001)
                return data.copy() + 1
            finally:
                self.active = False

    monkeypatch.setattr(pipeline, "Client", FakeClient)
    return clients


ADDRESSES = ["ipc:///tmp/pipeline-test-a", "ipc:///tmp/pipeline-test-b"]


@pytest.mark.parametrize("ordered", [False, True])
def test_map_once_bounded_and_closed(fake_clients, ordered):
    with process_map(
        range(30),
        ADDRESSES,
        loader=lambda i: np.array([i]),
        ordered=ordered,
        max_in_flight=5,
        max_input_bytes=24,
        size_of=lambda i: 8,
        setup_parameters={},
    ) as results:
        assert len(results.setup_results) == 2
        output = list(results)
    assert results.stats["peak_in_flight"] <= 3
    assert results.stats["peak_reserved_input_bytes"] <= 24
    assert results.stats["yielded"] == 30
    assert results.stats["in_flight"] == 0
    assert sorted(r.index for r in output) == list(range(30))
    if ordered:
        assert [r.index for r in output] == list(range(30))
    for result in output:
        np.testing.assert_array_equal(result.output, [result.item + 1])
        assert result.input_nbytes == 8
    assert all(c.closed for c in fake_clients)
    assert sorted(i for c in fake_clients for i in c.calls) == list(range(30))


def test_empty_and_early_exit(fake_clients):
    with process_map([], ADDRESSES) as empty:
        assert list(empty) == []
    seen = []

    def source():
        for i in range(100):
            seen.append(i)
            yield i

    with process_map(
        source(), ADDRESSES, loader=lambda i: np.array([i]), max_in_flight=3
    ) as results:
        next(results)
        time.sleep(0.03)  # no further consumption: no unbounded source exhaustion
        assert len(seen) == 4
    assert all(c.closed for c in fake_clients)
    assert list(results) == []


@pytest.mark.parametrize("stage", ["source", "size", "load", "process", "setup"])
def test_failures_terminal_and_cleanup(fake_clients, monkeypatch, stage):
    def source():
        if stage == "source":
            raise ValueError("source failed")
        yield 0

    def size(item):
        if stage == "size":
            raise ValueError("size failed")
        return 8

    def load(item):
        if stage == "load":
            raise ValueError("load failed")
        return np.array([item])

    if stage == "process":

        def fail(*args, **kwargs):
            raise ValueError("process failed")

        monkeypatch.setattr(pipeline.Client, "process", fail)
    with pytest.raises(PipelineError) as error:
        with process_map(
            source(),
            ADDRESSES,
            loader=load,
            size_of=size,
            setup_parameters={"fail": stage == "setup"},
        ) as results:
            try:
                next(results)
            except PipelineError:
                assert list(results) == []
                raise
    assert error.value.stage == stage
    assert isinstance(error.value.__cause__, ValueError)
    assert all(c.closed for c in fake_clients)


@pytest.mark.parametrize("reserve,budget", [(4, 8), (16, 8)])
def test_inaccurate_or_oversized_estimates_fail_before_processing(
    fake_clients, reserve, budget
):
    with pytest.raises(PipelineError):
        with process_map(
            [0],
            ADDRESSES,
            loader=lambda _: np.ones(8, dtype="u1"),
            size_of=lambda _: reserve,
            max_input_bytes=budget,
        ) as results:
            list(results)
    assert not any(c.calls for c in fake_clients)


def test_requires_numpy_loader(fake_clients):
    with pytest.raises(PipelineError, match="load") as error:
        with process_map([1], ADDRESSES, loader=lambda _: [1]) as results:
            list(results)
    assert isinstance(error.value.__cause__, TypeError)


@pytest.mark.parametrize(
    "options",
    [
        {"max_in_flight": 0},
        {"max_in_flight": True},
        {"load_workers": -1},
        {"max_input_bytes": 0},
        {"max_input_bytes": 8},
        {"addresses": []},
        {"addresses": ["same", "same"]},
        {"addresses": ["tcp://127.0.0.1:1"]},
        {"setup_parameters": []},
        {"signature": ("dict", "dict")},
    ],
)
def test_invalid_options(options):
    args = {"addresses": ADDRESSES, **options}
    with pytest.raises((ValueError, TypeError)):
        with process_map([], **args):
            pytest.fail("entered invalid pipeline")


@pytest.mark.timeout(10)
def test_ordered_head_backpressure(fake_clients, monkeypatch):
    release = threading.Event()
    admitted = []
    original = pipeline.Client.process

    def process(client, data, **kwargs):
        if data[0] == 0:
            assert release.wait(5)
        return original(client, data, **kwargs)

    def source():
        for i in range(100):
            admitted.append(i)
            yield i

    monkeypatch.setattr(pipeline.Client, "process", process)
    errors = []

    def consume():
        try:
            with process_map(
                source(),
                ADDRESSES,
                loader=lambda i: np.array([i]),
                ordered=True,
                max_in_flight=4,
            ) as results:
                assert next(results).index == 0
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=consume)
    thread.start()
    try:
        deadline = time.monotonic() + 3
        while (
            sum(len(c.calls) for c in fake_clients) < 3 and time.monotonic() < deadline
        ):
            time.sleep(0.005)
        assert len(admitted) == 4
        assert sum(len(c.calls) for c in fake_clients) == 3
    finally:
        release.set()
        thread.join(5)
    assert not thread.is_alive()
    assert errors == []


def test_loaded_array_released_before_next_budget(fake_clients):
    arrays = []

    def load(i):
        assert all(ref() is None for ref in arrays)
        array = np.array([i])
        arrays.append(weakref.ref(array))
        return array

    with process_map(
        range(6),
        ADDRESSES,
        loader=load,
        size_of=lambda i: 8,
        max_input_bytes=8,
        max_in_flight=1,
    ) as results:
        list(results)
    assert all(ref() is None for ref in arrays)


def test_foreign_thread_and_fork_rejected(fake_clients, monkeypatch):
    errors = []
    with process_map([], ADDRESSES) as results:

        def consume():
            for action in (lambda: next(results), results.close):
                try:
                    action()
                except RuntimeError as error:
                    errors.append(error)

        thread = threading.Thread(target=consume)
        thread.start()
        thread.join()
        assert len(errors) == 2
        with monkeypatch.context() as patch:
            patch.setattr(pipeline.os, "getpid", lambda: results._pid + 1)
            with pytest.raises(RuntimeError, match="creating process"):
                results.close()
        assert list(results) == []


@pytest.mark.parametrize("shared", [False, True])
def test_real_nng_workers(shared, monkeypatch):
    names = capture_segment_names(monkeypatch)
    with ExitStack() as stack:
        addresses = [stack.enter_context(execed_server("increment")) for _ in range(2)]
        with process_map(
            range(20),
            addresses,
            loader=lambda i: np.full((3, 4), i, dtype="f4"),
            shared_memory=shared,
            ordered=True,
            max_in_flight=4,
            setup_parameters={},
            timeout_ms=5000,
        ) as results:
            for result in results:
                np.testing.assert_array_equal(result.output, result.index + 1)
                if shared:
                    assert result.output.flags.owndata
        assert results.stats["yielded"] == 20
    assert_segments_unlinked(names)


@pytest.mark.timeout(15)
def test_timeout_stops_pipeline_and_unlinks_shared_storage(monkeypatch):
    import pynng

    names = capture_segment_names(monkeypatch)
    with execed_server("sleep") as address:
        with process_map([np.array([1])], [address], timeout_ms=100) as results:
            with pytest.raises(PipelineError) as error:
                next(results)
            assert isinstance(error.value.__cause__, pynng.Timeout)
            assert list(results) == []
    assert names
    assert_segments_unlinked(names)


@pytest.mark.parametrize("fail", [False, True])
def test_stream_arrays_example(tmp_path, monkeypatch, fail):
    source = tmp_path / "input.npy"
    np.save(source, np.arange(4))
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps(dict(id="site", path=str(source), nbytes=1 if fail else 32)) + "\n"
    )
    output = tmp_path / "output"
    script = Path(__file__).resolve().parents[1] / "examples" / "stream_arrays.py"
    with execed_server("increment") as address:
        monkeypatch.setattr(
            sys, "argv", [str(script), str(manifest), str(output), "--address", address]
        )
        if fail:
            with pytest.raises(PipelineError):
                runpy.run_path(str(script), run_name="__main__")
            assert not (output / "SUCCESS.json").exists()
        else:
            runpy.run_path(str(script), run_name="__main__")
            np.testing.assert_array_equal(
                np.load(output / "000000000.npy"), np.arange(4) + 1
            )
            assert json.loads((output / "SUCCESS.json").read_text())["yielded"] == 1
            assert json.loads((output / "receipt.jsonl").read_text())["id"] == "site"
        # Both completed and partial directories are create-only, never resumed.
        with pytest.raises(FileExistsError):
            runpy.run_path(str(script), run_name="__main__")


def test_real_nng_failure():
    with execed_server("error") as address:
        with process_map([np.arange(4)], [address], timeout_ms=5000) as results:
            with pytest.raises(PipelineError, match="process") as error:
                next(results)
            assert "injected" in str(error.value.__cause__)
            assert list(results) == []
