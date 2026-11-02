"""Tests that the responder accepts multiple setup() calls in one lifetime.

Spins up a tiny in-process Nahual server (pynng Rep0) whose ``setup`` returns
a closure capturing a tag from the setup-time parameters. Then:

  1. setup(tag="A")     → process(x) returns x + ord("A")
  2. setup(tag="B")     → process(x) must now return x + ord("B"), proving the
                          new closure replaced the old one.

This test would fail against the pre-patch responder, which treated the
second dict-shaped message as a numpy payload and crashed (or fed it to the
processor).
"""

import os
import tempfile
from functools import partial

import numpy
import pynng
import pytest
import trio

from nahual.process import Client, dispatch_setup_process
from nahual.server import responder


def _make_setup():
    """Build a setup that returns a closure parameterised by a string tag.

    The closure adds ``ord(tag)`` to every element of the input array, so
    different tags produce verifiably different outputs.
    """

    def setup(tag: str = "A"):
        offset = ord(tag)

        def processor(arr: numpy.ndarray) -> numpy.ndarray:
            return arr.astype(numpy.float32) + float(offset)

        info = {"tag": tag, "offset": offset}
        return processor, info

    return setup


@pytest.mark.timeout(15)
@pytest.mark.parametrize("persistent", [False, True])
def test_setup_can_be_recalled_with_new_params(persistent):
    # Unique IPC path per test run.
    tmpdir = tempfile.mkdtemp(prefix="nahual_test_")
    address = f"ipc://{os.path.join(tmpdir, 'reload.ipc')}"

    setup_fn = _make_setup()
    client_setup, client_process = dispatch_setup_process(
        "test", signature=("dict", "numpy")
    )

    client = Client(address)
    if persistent:
        # Keep the same call signatures below while exercising a single socket.
        def client_setup(data, address):
            return client.setup(data)

        def client_process(data, address):
            return client.process(data)

    results = {}

    async def server_task(sock):
        await responder(sock, setup=setup_fn)

    async def client_task(cancel_scope):
        # Run blocking pynng-Req0 client calls in a worker thread so they
        # don't block the trio loop hosting the server.
        info_a = await trio.to_thread.run_sync(
            partial(client_setup, {"tag": "A"}, address=address)
        )
        results["info_a"] = info_a

        x = numpy.zeros((2, 3), dtype=numpy.float32)
        out_a = await trio.to_thread.run_sync(
            partial(client_process, x, address=address)
        )
        results["out_a"] = out_a

        # Re-setup with a different tag — this must re-run setup, not be fed
        # to the existing processor.
        info_b = await trio.to_thread.run_sync(
            partial(client_setup, {"tag": "B"}, address=address)
        )
        results["info_b"] = info_b

        out_b = await trio.to_thread.run_sync(
            partial(client_process, x, address=address)
        )
        results["out_b"] = out_b

        cancel_scope.cancel()

    async def main():
        with pynng.Rep0(listen=address, recv_timeout=5000) as sock:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(server_task, sock)
                nursery.start_soon(client_task, nursery.cancel_scope)

    try:
        trio.run(main)
    finally:
        client.close()

    assert results["info_a"] == {"tag": "A", "offset": ord("A")}
    assert results["info_b"] == {"tag": "B", "offset": ord("B")}

    expected_a = numpy.full((2, 3), float(ord("A")), dtype=numpy.float32)
    expected_b = numpy.full((2, 3), float(ord("B")), dtype=numpy.float32)
    numpy.testing.assert_array_equal(results["out_a"], expected_a)
    numpy.testing.assert_array_equal(results["out_b"], expected_b)
