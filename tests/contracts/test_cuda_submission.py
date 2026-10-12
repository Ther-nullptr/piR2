"""Host concurrency contracts for process-wide CUDA capture exclusion."""

import threading
from concurrent.futures import ThreadPoolExecutor


def test_two_simultaneous_capture_upgrades_do_not_deadlock_or_overlap():
    from coexecution.optimization.execution.submission import CudaSubmissionCoordinator

    coordinator = CudaSubmissionCoordinator()
    barrier = threading.Barrier(2)
    active = []
    seen = []

    def worker(index):
        with coordinator.submission():
            barrier.wait(timeout=2)
            with coordinator.capture():
                assert active == []
                active.append(index)
                seen.append(index)
                with coordinator.capture():
                    assert active == [index]
                active.pop()

    with ThreadPoolExecutor(2) as workers:
        futures = [workers.submit(worker, index) for index in range(2)]
        for future in futures:
            future.result(timeout=3)
    assert sorted(seen) == [0, 1]


def test_capture_waits_for_reader_and_quiesces_before_excluding_new_submissions():
    from coexecution.optimization.execution.submission import CudaSubmissionCoordinator

    active, observations = [], []
    coordinator = CudaSubmissionCoordinator(
        quiesce=lambda: observations.append(list(active))
    )
    reader_ready, release_reader, captured = (threading.Event() for _ in range(3))

    def reader():
        with coordinator.submission():
            active.append("reader")
            reader_ready.set()
            assert release_reader.wait(2)
            active.pop()

    def writer():
        with coordinator.submission(), coordinator.capture():
            captured.set()

    with ThreadPoolExecutor(2) as workers:
        first = workers.submit(reader)
        assert reader_ready.wait(2)
        second = workers.submit(writer)
        assert not captured.wait(0.02)
        release_reader.set()
        first.result(timeout=2)
        second.result(timeout=2)
    assert observations == [[]]


def test_exception_restores_submission_and_exclusive_ownership():
    import pytest

    from coexecution.optimization.execution.submission import CudaSubmissionCoordinator

    coordinator = CudaSubmissionCoordinator()
    with coordinator.submission():
        with pytest.raises(RuntimeError), coordinator.capture():
            raise RuntimeError("capture failed")
        with coordinator.capture():
            pass
    with coordinator.submission(), coordinator.capture():
        pass
