"""Execed responder used by shared-memory integration tests and benchmarks."""

import argparse
import sys
import time

import numpy
import pynng
import trio

from nahual.server import responder


def processor_for(mode: str):
    if mode == "identity":
        return lambda array: array
    if mode == "smaller":
        return (
            lambda array: numpy.asarray(array)
            .reshape(-1)[: max(0, array.size // 2)]
            .copy()
        )
    if mode == "larger":
        return lambda array: numpy.concatenate((array.reshape(-1), array.reshape(-1)))
    if mode == "increment":
        return lambda array: array + 1
    if mode == "mutate":

        def mutate(array):
            array.flags.writeable = True
            array[...] = 0
            return array

        return mutate
    if mode == "alias":
        return lambda array: array.reshape(-1)[1:]
    if mode == "non-native-output":
        byte_order = ">" if sys.byteorder == "little" else "<"
        return lambda array: numpy.asarray(array, dtype=f"{byte_order}i4")
    if mode == "error":

        def fail(_array):
            raise ValueError("injected shared-memory processor error")

        return fail
    if mode == "sleep":

        def sleep_then_identity(array):
            time.sleep(2)
            return array

        return sleep_then_identity
    raise ValueError(f"unknown processor mode {mode!r}")


async def main(address: str, mode: str) -> None:
    if mode == "preexisting-tracker":
        from multiprocessing import shared_memory

        probe = shared_memory.SharedMemory(create=True, size=1)
        probe.close()
        probe.unlink()

    def unused_setup(**_parameters):
        return processor_for(mode), {"mode": mode}

    with pynng.Rep0(listen=address, recv_timeout=60_000) as socket:
        print("READY", flush=True)
        processor_mode = "identity" if mode == "preexisting-tracker" else mode
        await responder(
            socket, setup=unused_setup, processor=processor_for(processor_mode)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("address")
    parser.add_argument("mode")
    arguments = parser.parse_args()
    trio.run(main, arguments.address, arguments.mode)
