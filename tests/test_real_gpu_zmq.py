from __future__ import annotations

import gc
import select
import subprocess
import sys

import numpy as np
import pytest


@pytest.mark.timeout(30)
def test_benchmark_zmq_setup_lifetime_error_and_recovery(tmp_path):
    pytest.importorskip("zmq")
    from benchmarks import real_gpu_zmq as module

    address = f"ipc://{tmp_path}/benchmark.ipc"
    code = """
import sys
from benchmarks.real_gpu_zmq import serve
state = {"fail": False, "offset": 0}
def processor(array):
    if state["fail"]:
        raise ValueError("injected processor failure")
    return array + state["offset"]
def setup(**parameters):
    state.update(parameters)
    return processor, state.copy()
serve(sys.argv[1], setup, processor, lambda: print("READY", flush=True))
"""
    server = subprocess.Popen(
        [sys.executable, "-u", "-c", code, address],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    client = None
    try:
        ready, _, _ = select.select([server.stdout], [], [], 10)
        assert ready and server.stdout.readline().strip() == "READY"
        client = module.Session(address)
        assert client.setup({"offset": 3}) == {"fail": False, "offset": 3}
        source = np.arange(262144, dtype=np.float32).reshape(256, 1024)
        expected = source + 3
        first = client.process(source)
        source[:] = -100
        second = client.process(np.full_like(source, 7))
        np.testing.assert_array_equal(first, expected)
        np.testing.assert_array_equal(second, 10)
        assert not first.flags.writeable
        client.setup({"fail": True})
        with pytest.raises(RuntimeError, match="injected processor failure"):
            client.process(source)
        client.setup({"fail": False, "offset": 4})
        np.testing.assert_array_equal(client.process(np.ones_like(source)), 5)
        client.close()
        client = None
        gc.collect()
        np.testing.assert_array_equal(first, expected)
    finally:
        if client is not None:
            client.close()
        server.terminate()
        _, error = server.communicate(timeout=5)
        assert not error
