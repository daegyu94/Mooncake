import random
import threading
from pathlib import Path

import pytest

from range_planner import PlannedStore, plan_ranges
from test_generation_store import MemoryIO


def test_dense_sparse_duplicates_and_bounds():
    data = bytes(range(200))
    for locations in [
        [],
        [(0, 0)],
        [(14, 7), (0, 7), (14, 7), (7, 7)],
        [(0, 7), (70, 7)],
        [(0, 100)],
    ]:
        for limit in [7, 28, 64]:
            for amp in [1, 1.25]:
                plan = plan_ranges(locations, limit, amp)
                buffers = [data[off : off + size] for off, size in plan.ranges]
                values, _ = plan.scatter(buffers)
                assert values == [data[off : off + size] for off, size in locations]
                assert all(size <= limit for _, size in plan.ranges)
                assert sum(size for _, size in plan.ranges) <= amp * plan.unique_bytes
    assert len(plan_ranges([(i * 7, 7) for i in range(8)], 64).ranges) == 1
    assert len(plan_ranges([(0, 7), (70, 7)], 64).ranges) == 2


def test_gap_budget_and_duplicate_denominator():
    locations = [(0, 4), (5, 4), (5, 4)]
    assert plan_ranges(locations, 100).ranges == [(0, 4), (5, 4)]
    assert plan_ranges(locations, 100, 1.25).ranges == [(0, 9)]
    assert plan_ranges(locations, 8, 1.25).ranges == [(0, 4), (5, 4)]
    with pytest.raises(ValueError):
        plan_ranges([(0, 10), (5, 10)], 100)
    for amp in [0.9, float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            plan_ranges([], 4, amp)
    with pytest.raises(ValueError):
        plan_ranges([(-1, 2)], 4)


def test_random_scatter():
    rng = random.Random(127)
    data = bytes(range(256)) * 16
    for _ in range(100):
        offsets, cursor = [], 0
        for _ in range(50):
            size = rng.randrange(0, 40)
            offsets.append((cursor, size))
            cursor += size + rng.randrange(0, 10)
        locations = rng.choices(offsets, k=40)
        plan = plan_ranges(locations, rng.randrange(20, 128), 1.25)
        values, _ = plan.scatter([data[off : off + size] for off, size in plan.ranges])
        assert values == [data[off : off + size] for off, size in locations]
        assert sum(size for _, size in plan.ranges) <= 1.25 * plan.unique_bytes


def test_completion_and_lease_errors(tmp_path):
    plan = plan_ranges([(0, 4)], 8)
    with pytest.raises(ValueError):
        plan.scatter([])
    with pytest.raises(ValueError):
        plan.scatter([b"bad"])
    s = PlannedStore(MemoryIO(), tmp_path / "store", 100)
    g = s.create()
    s.append(g, [b"old"])
    with pytest.raises(KeyError):
        s.get_planned(g, [9])
    assert s.generations[g].readers == 0


def test_inflight_read_fences_reclaim(tmp_path):
    class GatedIO(MemoryIO):
        def io(self, fd, ranges, read=False):
            if read:
                entered.set()
                assert proceed.wait(5)
            return super().io(fd, ranges, read)

    entered, proceed = threading.Event(), threading.Event()
    io = GatedIO()
    s = PlannedStore(io, Path(tmp_path) / "store", 100)
    old = s.create()
    s.append(old, [b"old", b"kv"])
    s.seal(old)
    output, errors = [], []

    def reader():
        try:
            output.extend(s.get_planned(old, [1, 0, 1]))
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=reader)
    thread.start()
    try:
        assert entered.wait(5)
        s.invalidate(old)
        assert s.gc() == 0
        with pytest.raises(KeyError):
            s.get_planned(old, [0])
        with pytest.raises(KeyError):
            s.append(old, [b"late"])
        new = s.create()
        s.append(new, [b"new"])
    finally:
        proceed.set()
        thread.join(5)
    assert not thread.is_alive() and not errors
    assert output == [b"kv", b"old", b"kv"]
    assert s.gc() == 1
    assert s.get_planned(new, [0]) == [b"new"]
