from __future__ import annotations

import gc
import json
import subprocess
import sys

import numpy
import pytest

from benchmarks import transport as module


@pytest.mark.timeout(30)
@pytest.mark.parametrize("zmq", [False, True])
def test_transport_benchmark_smoke(zmq):
    if zmq:
        pytest.importorskip("zmq")
    command = [
        sys.executable,
        "-m",
        "benchmarks.transport",
        "--sizes",
        "4096",
        "--iterations",
        "2",
        "--warmup",
        "1",
    ]
    if zmq:
        command.append("--zmq")
    result = subprocess.run(command, capture_output=True, text=True, timeout=25)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert len(report["results"]) == 2
    for row in report["results"]:
        assert row["input_bytes"] == 4096
        assert row["shm_session_cached"]["median_ms"] > 0
        assert ("zmq_multipart_session" in row) == zmq
    assert "resource_tracker" not in result.stderr


@pytest.mark.timeout(30)
def test_zmq_zero_copy_result_retains_its_owner_after_next_call_and_close(tmp_path):
    pytest.importorskip("zmq")
    source = numpy.arange(262144, dtype=numpy.float32).reshape(256, 1024)
    expected = source.copy()
    with module.server("zmq", "identity", tmp_path) as address:
        with module.zmq_client(address) as call:
            first = call(source)
            source[:] = -1  # call returned only after the send released its input
            second = call(numpy.full_like(source, 7))
            numpy.testing.assert_array_equal(first, expected)
            numpy.testing.assert_array_equal(second, 7)
            assert not first.flags.writeable
    gc.collect()
    numpy.testing.assert_array_equal(first, expected)
