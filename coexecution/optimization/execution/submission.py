"""Exclude CUDA graph capture from every other endpoint's CUDA submissions.

Normal endpoint handlers share reader access. Cold graph calls upgrade by
dropping their reader slot while retaining Python/tensor references, so two
simultaneous upgrades cannot deadlock. Exclusive entry quiesces submitted GPU
work before capture. All CUDA API paths, including startup/teardown, must join.
"""

import threading
from contextlib import contextmanager, nullcontext

_CURRENT = threading.local()


class CudaSubmissionCoordinator:
    def __init__(self, *, quiesce=lambda: None):
        self._condition = threading.Condition(threading.RLock())
        self._readers = {}
        self._writer = None
        self._write_depth = self._waiting = 0
        self._quiesce = quiesce
        self.captures = 0

    @contextmanager
    def submission(self):
        thread = threading.get_ident()
        with self._condition:
            self._condition.wait_for(
                lambda: (
                    self._writer == thread
                    or (
                        self._writer is None
                        and (not self._waiting or thread in self._readers)
                    )
                )
            )
            self._readers[thread] = self._readers.get(thread, 0) + 1
        previous = getattr(_CURRENT, "coordinator", None)
        _CURRENT.coordinator = self
        try:
            yield
        finally:
            _CURRENT.coordinator = previous
            with self._condition:
                self._readers[thread] -= 1
                if not self._readers[thread]:
                    del self._readers[thread]
                self._condition.notify_all()

    @contextmanager
    def capture(self):
        thread = threading.get_ident()
        with self._condition:
            nested = self._writer == thread
            suspended = 0
            if not nested:
                suspended = self._readers.pop(thread, 0)
                self._waiting += 1
                self._condition.notify_all()
                try:
                    self._condition.wait_for(
                        lambda: self._writer is None and not self._readers
                    )
                    self._writer = thread
                finally:
                    self._waiting -= 1
            self._write_depth += 1
        try:
            if not nested:
                self._quiesce()
                self.captures += 1
            yield
        finally:
            with self._condition:
                self._write_depth -= 1
                if not self._write_depth:
                    if suspended:
                        self._readers[thread] = suspended
                    self._writer = None
                    self._condition.notify_all()


def cold_capture(cold):
    coordinator = getattr(_CURRENT, "coordinator", None)
    return coordinator.capture() if cold and coordinator is not None else nullcontext()
