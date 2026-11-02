"""Bounded decode/prefetch and site-sharded processing over existing clients.

No model, GPU, transport, or inference-batch policy is imposed. Each endpoint is
an independently loaded worker; a site is sent to exactly one free worker.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
import numbers
import os
import threading
import time
from typing import Any, Callable, Iterable, Iterator

import numpy

from nahual.process import Client


class PipelineError(RuntimeError):
    """A load/setup/process failure; the original exception is chained."""

    def __init__(self, stage, index=None, address=None):
        self.stage, self.index, self.address = stage, index, address
        super().__init__(f"pipeline {stage} failed (index={index}, address={address})")


@dataclass(frozen=True)
class PipelineResult:
    index: int
    item: Any
    output: Any
    address: str
    input_nbytes: int
    load_seconds: float
    process_seconds: float


@dataclass
class _Work:
    index: int
    item: Any
    reserved: int
    data: Any = None
    input_nbytes: int = 0
    load_seconds: float = 0.0


def _positive(value, name, *, zero=False):
    if (
        isinstance(value, bool)
        or not isinstance(value, numbers.Integral)
        or value < (0 if zero else 1)
    ):
        raise ValueError(
            f"{name} must be a {'non-negative' if zero else 'positive'} integer"
        )
    return int(value)


def _load(loader, item):
    started = time.perf_counter()
    data = loader(item) if loader is not None else item
    if not isinstance(data, numpy.ndarray):
        raise TypeError("pipeline loaders must return a NumPy ndarray")
    return data, time.perf_counter() - started


def _process(client, data, shared_memory):
    started = time.perf_counter()
    output = client.process(data, shared_memory=shared_memory)
    return output, time.perf_counter() - started


class PipelineResults(Iterator[PipelineResult]):
    """Single-consumer iterator owned by a ``process_map`` context.

    ``stats`` and ``setup_results`` remain inspectable after context exit.
    """

    def __init__(
        self,
        items,
        addresses,
        loader,
        size_of,
        max_in_flight,
        max_input_bytes,
        load_workers,
        ordered,
        client_options,
        setup_parameters,
        shared_memory,
    ):
        self._pid, self._thread = os.getpid(), threading.get_ident()
        self._source = iter(items)
        self._loader, self._size_of = loader, size_of
        self._limit, self._byte_limit = max_in_flight, max_input_bytes
        self._ordered, self._shared = ordered, shared_memory
        self._jobs, self._loading, self._running = {}, {}, {}
        self._ready, self._done_order = deque(), deque()
        self._done = {}
        self._lookahead = None
        self._source_done, self._closed = False, False
        self._next_index, self._next_yield = 0, 0
        self._reserved = 0
        self._stats = dict(
            admitted=0,
            loaded=0,
            processed=0,
            yielded=0,
            peak_in_flight=0,
            peak_reserved_input_bytes=0,
        )
        self._stack = ExitStack()
        self.setup_results = ()
        try:
            self._clients = [
                self._stack.enter_context(Client(address, **client_options))
                for address in addresses
            ]
            # Executors drain/cancel before ExitStack closes any active client.
            self._loads = ThreadPoolExecutor(
                max_workers=load_workers, thread_name_prefix="nahual-load"
            )
            self._stack.callback(self._loads.shutdown, wait=True, cancel_futures=True)
            self._requests = ThreadPoolExecutor(
                max_workers=len(addresses), thread_name_prefix="nahual-process"
            )
            self._stack.callback(
                self._requests.shutdown, wait=True, cancel_futures=True
            )
            self._addresses = addresses
            self._free = deque(range(len(addresses)))
            if setup_parameters is not None:
                futures = [
                    self._requests.submit(client.setup, setup_parameters)
                    for client in self._clients
                ]
                evidence = []
                for address, future in zip(addresses, futures):
                    try:
                        evidence.append(future.result())
                    except Exception as error:
                        raise PipelineError("setup", address=address) from error
                self.setup_results = tuple(evidence)
        except BaseException:
            self.close()
            raise

    def _check_owner(self):
        if os.getpid() != self._pid or threading.get_ident() != self._thread:
            raise RuntimeError(
                "consume/close pipeline results only in their creating process and thread"
            )

    @property
    def stats(self):
        return {
            **self._stats,
            "in_flight": len(self._jobs),
            "reserved_input_bytes": self._reserved,
        }

    def _high_water(self):
        self._stats["peak_in_flight"] = max(
            self._stats["peak_in_flight"], len(self._jobs)
        )
        self._stats["peak_reserved_input_bytes"] = max(
            self._stats["peak_reserved_input_bytes"], self._reserved
        )

    def _admit(self):
        while not self._source_done and len(self._jobs) < self._limit:
            if self._lookahead is None:
                try:
                    item = next(self._source)
                except StopIteration:
                    self._source_done = True
                    break
                except Exception as error:
                    raise PipelineError("source", self._next_index) from error
                try:
                    reserve = (
                        _positive(self._size_of(item), "size_of result", zero=True)
                        if self._size_of
                        else 0
                    )
                except Exception as error:
                    raise PipelineError("size", self._next_index) from error
                self._lookahead = _Work(self._next_index, item, reserve)
                if self._byte_limit is not None and reserve > self._byte_limit:
                    raise PipelineError("size", self._next_index) from ValueError(
                        "one input exceeds max_input_bytes"
                    )
            work = self._lookahead
            if (
                self._byte_limit is not None
                and self._reserved + work.reserved > self._byte_limit
            ):
                break
            self._lookahead = None
            self._next_index += 1
            self._jobs[work.index] = work
            self._reserved += work.reserved
            self._stats["admitted"] += 1
            self._high_water()
            future = self._loads.submit(_load, self._loader, work.item)
            self._loading[future] = work

    def _dispatch(self):
        while self._ready and self._free:
            work, worker = self._ready.popleft(), self._free.popleft()
            future = self._requests.submit(
                _process, self._clients[worker], work.data, self._shared
            )
            self._running[future] = (work, worker)

    def _advance(self):
        # Keep Future/result references in this non-generator helper, so decoded
        # arrays are released before yielding and refilling their byte budget.
        completed, _ = wait(
            tuple(self._loading) + tuple(self._running), return_when=FIRST_COMPLETED
        )
        for future in completed:
            if future in self._loading:
                work = self._loading.pop(future)
                try:
                    data, seconds = future.result()
                    if self._byte_limit is not None and data.nbytes > work.reserved:
                        raise ValueError(
                            "decoded input exceeds its size_of reservation"
                        )
                except Exception as error:
                    raise PipelineError("load", work.index) from error
                work.data, work.input_nbytes, work.load_seconds = (
                    data,
                    data.nbytes,
                    seconds,
                )
                if self._byte_limit is None:
                    self._reserved += data.nbytes - work.reserved
                    work.reserved = data.nbytes
                    self._high_water()
                self._stats["loaded"] += 1
                self._ready.append(work)
            else:
                work, worker = self._running.pop(future)
                try:
                    output, seconds = future.result()
                except Exception as error:
                    raise PipelineError(
                        "process", work.index, self._addresses[worker]
                    ) from error
                work.data = None
                self._free.append(worker)
                self._stats["processed"] += 1
                self._done[work.index] = PipelineResult(
                    work.index,
                    work.item,
                    output,
                    self._addresses[worker],
                    work.input_nbytes,
                    work.load_seconds,
                    seconds,
                )
                self._done_order.append(work.index)
        self._dispatch()

    def __next__(self):
        self._check_owner()
        if self._closed:
            raise StopIteration
        try:
            return self._next()
        except BaseException:
            # Errors are terminal even when caught inside the with block.
            self.close()
            raise

    def _next(self):
        while True:
            self._admit()
            self._dispatch()
            index = (
                self._next_yield
                if self._ordered
                else (self._done_order[0] if self._done_order else None)
            )
            if index in self._done:
                result = self._done.pop(index)
                self._done_order.remove(index)
                work = self._jobs.pop(index)
                self._reserved -= work.reserved
                self._stats["yielded"] += 1
                self._next_yield += 1
                # Refill before the consumer saves the output, overlapping CPU
                # preparation with inference and downstream output handling.
                self._admit()
                self._dispatch()
                return result
            if self._source_done and not self._jobs:
                raise StopIteration
            self._advance()

    def close(self):
        self._check_owner()
        if self._closed:
            return
        self._closed = True
        try:
            self._stack.close()
        finally:
            self._jobs.clear()
            self._loading.clear()
            self._running.clear()
            self._ready.clear()
            self._done.clear()
            self._done_order.clear()
            self._lookahead = None
            self._source = iter(())
            self._reserved = 0


@contextmanager
def process_map(
    items: Iterable,
    addresses: Iterable[str],
    *,
    loader: Callable | None = None,
    size_of: Callable | None = None,
    max_in_flight: int | None = None,
    max_input_bytes: int | None = None,
    load_workers: int = 4,
    ordered: bool = False,
    setup_parameters: dict | None = None,
    shared_memory: bool = True,
    timeout_ms: int | None = None,
    shared_memory_cache_bytes: int = 64 * 1024**2,
    signature: tuple[str, str] = ("dict", "numpy"),
):
    """Overlap bounded input loading and synchronous requests to model replicas.

    Enter with ``with process_map(...) as results`` and consume PipelineResult
    objects. Each unique endpoint needs a dedicated, independently loaded
    replica accepting the same inputs. Work is dynamically site-sharded; no
    inference batches, model setup changes, or automatic retries are introduced.

    ``max_in_flight`` counts admitted inputs until their results are yielded,
    including pending loads and ordered results (default twice worker count).
    ``max_input_bytes`` additionally bounds reserved decoded-input bytes and
    requires a cheap ``size_of(item)`` upper bound before loading. Underestimates
    fail before processing, but cannot undo an oversized loader allocation.
    These are queue bounds, not total RSS limits: client SHM caches, loader
    temporaries, outputs and consumer-retained objects are separate. At most one
    cheap source-record lookahead is held outside the admitted window.

    Input arrays must remain alive and unmodified until their result is yielded;
    loaders should return fresh arrays rather than reuse a mutable buffer.
    Output order is completion order unless ``ordered=True``. Slow ordered heads
    apply backpressure instead of allowing unlimited completed-result buffering.

    Consume/close on the creating thread/process. Context exit cancels queued
    work and waits for active loads/requests before closing clients. A request
    timeout does not cancel server computation; failed SHM storage is discarded
    by Client. Finite timeouts are recommended; no hard cancellation is promised
    for arbitrary blocking loader callbacks or an unbounded request.
    """
    addresses = tuple(addresses)
    if (
        not addresses
        or any(not isinstance(address, str) or not address for address in addresses)
        or len(set(addresses)) != len(addresses)
    ):
        raise ValueError("addresses must be nonempty, unique endpoint strings")
    load_workers = _positive(load_workers, "load_workers")
    window = (
        2 * len(addresses)
        if max_in_flight is None
        else _positive(max_in_flight, "max_in_flight")
    )
    if max_input_bytes is not None:
        max_input_bytes = _positive(max_input_bytes, "max_input_bytes")
        if size_of is None:
            raise ValueError("max_input_bytes requires size_of")
    if setup_parameters is not None and not isinstance(setup_parameters, dict):
        raise TypeError("setup_parameters must be a dictionary")
    if shared_memory:
        from nahual.shared_memory import require_ipc_address

        for address in addresses:
            require_ipc_address(address)
        if tuple(signature[1:]) != ("numpy",):
            raise ValueError("shared-memory pipeline requires a NumPy output signature")
    stream = PipelineResults(
        items,
        addresses,
        loader,
        size_of,
        window,
        max_input_bytes,
        load_workers,
        ordered,
        dict(
            signature=signature,
            timeout_ms=timeout_ms,
            shared_memory_cache_bytes=shared_memory_cache_bytes,
        ),
        setup_parameters,
        shared_memory,
    )
    try:
        yield stream
    finally:
        stream.close()
