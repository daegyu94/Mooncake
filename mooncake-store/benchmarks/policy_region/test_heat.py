"""Memory-backend correctness, causal decisions and adversarial admission."""

from dataclasses import replace

import pytest
from bench_heat import trace
from coordinator import StoreError
from heat_store import HeatResidentStore, RecentFrequency
from test_resident import MemoryIO, fill, read


def create(root, mode="adaptive", slots=8):
    root.mkdir(exist_ok=True)
    io = MemoryIO()
    return HeatResidentStore(io, root, slots * 64, mode), io


@pytest.mark.parametrize(
    "pattern",
    [
        "cold",
        "hot",
        "cold_to_hot",
        "hot_to_cold",
        "scan",
        "late",
        "barrier",
        "working_set",
        "mixed_size",
    ],
)
@pytest.mark.parametrize("mode", ["heat_lru", "adaptive"])
def test_full_trace_conservation(tmp_path, pattern, mode):
    store, io = create(tmp_path, mode)
    result = trace(store, [io], 64, pattern, epochs=8)
    m = result["final"]["metrics"]
    assert m["accepted_bytes"] == io.writes + m["discard_dirty_bytes"]
    assert result["logical_get_bytes"] == io.reads + m["ram_read_bytes"]
    assert not io.files


def test_no_future_heat_or_deadline_at_cold_start(tmp_path):
    store, io = create(tmp_path)
    old = store.create("A", 64)
    lease = store.acquire(old, "A")
    fill(store, old, 8)
    store.revoke(old)
    new = store.create("B", 64)
    fill(store, new, 8)
    assert io.writes == 8 * 64  # No trained lifetime evidence: baseline choice.
    assert store.drain_samples == 0 and store.metrics["protection_decisions"] == 0
    read(store, lease, list(range(8)))
    store.release(lease)
    assert store.drain_forecast == 8 * 64
    store.revoke(new)


def test_sketch_is_bounded_and_decays():
    sketch = RecentFrequency(16)
    for _ in range(64):
        sketch.record((1, 7))
    assert sketch.decays == 1 and sketch.estimate((1, 7)) == 7
    assert sum(map(len, sketch.rows)) == 64
    assert sketch.estimate((2, 7)) <= 7


@pytest.mark.parametrize("mode", ["heat_lru", "adaptive"])
def test_hot_clean_block_not_unconditionally_evicted(tmp_path, mode):
    s, io = create(tmp_path, mode, 2)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 3)
    # Third demand beats a cold dirty incumbent; cost comparison is shared.
    for _ in range(4):
        read(s, lease, [0])
    assert s.policies[p].blocks[0].data is not None
    writes = io.writes
    fill(s, p, 1, 3)
    assert s.policies[p].blocks[0].data is not None
    assert io.writes > writes  # Cold dirty eviction wins over hot clean eviction.
    s.revoke(p)
    s.release(lease)


def test_retired_get_cannot_refill_or_cross_weight_identity(tmp_path):
    s, _ = create(tmp_path, slots=1)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 2)
    s.revoke(p)
    for _ in range(3):
        read(s, lease, [0])
    assert s.policies[p].blocks[0].data is None
    assert s.metrics["admission_stale_rejections"] == 3
    with pytest.raises(StoreError), s.read(replace(lease, identity="B"), [0]):
        pass
    s.release(lease)


def test_admission_is_optional_when_writer_or_pins_own_capacity(tmp_path):
    s, _ = create(tmp_path, slots=1)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 2)
    assert s.mutation.acquire(blocking=False)
    try:
        read(s, lease, [0])
        read(s, lease, [0])
    finally:
        s.mutation.release()
    assert s.metrics["admission_busy"] == 2
    with s.read(lease, [1]):
        read(s, lease, [0])
        assert s.policies[p].blocks[0].data is None
    s.revoke(p)
    s.release(lease)


def test_failed_admission_spill_does_not_fail_successful_get(tmp_path):
    s, io = create(tmp_path, slots=1)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 2)
    read(s, lease, [0])
    read(s, lease, [0])  # Equality preserves the dirty incumbent.
    io.fail = True
    read(s, lease, [0])
    assert s.metrics["admission_failures"] == 1
    assert s.policies[p].blocks[1].data is not None and s.used == 64
    io.fail = False
    s.revoke(p)
    s.release(lease)


def test_active_frequency_does_not_survive_as_retired_prediction(tmp_path):
    s, _ = create(tmp_path)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 1)
    for _ in range(4):
        read(s, lease, [0])
    block = s.policies[p].blocks[0]
    assert s.frequency.estimate(s._frequency_key(block)) == 4
    s.revoke(p)
    assert s.frequency.estimate(s._frequency_key(block)) == 0
    s.release(lease)


def test_large_admission_plans_all_small_victims(tmp_path):
    s, io = create(tmp_path, slots=8)
    big = s.create("large", 512)
    bl = s.acquire(big, "large")
    s.put(big, b"L" * 512)
    small = s.create("small", 64)
    fill(s, small, 8)
    for _ in range(3):
        with s.read(bl, [0]) as values:
            assert values[0] == b"L" * 512
    assert s.policies[big].blocks[0].data is not None
    assert s.metrics["admission_dirty_spill_bytes"] == 512
    s.revoke(big)
    s.release(bl)
    s.revoke(small)
    assert not io.files


def test_impossible_large_admission_has_no_partial_spills(tmp_path):
    s, io = create(tmp_path, slots=8)
    big = s.create("large", 512)
    bl = s.acquire(big, "large")
    s.put(big, b"L" * 512)
    small = s.create("small", 64)
    sl = s.acquire(small, "small")
    fill(s, small, 8)
    with s.read(sl, [0]):
        before = io.writes
        for _ in range(2):
            with s.read(bl, [0]) as values:
                assert values[0] == b"L" * 512
        assert io.writes == before
        assert s.policies[big].blocks[0].data is None
    s.revoke(big)
    s.release(bl)
    s.revoke(small)
    s.release(sl)


@pytest.mark.parametrize("mode", ["heat_lru", "adaptive"])
@pytest.mark.parametrize("slots", [1, 4])
def test_uniform_hot_scan_preserves_equal_value_incumbents(tmp_path, mode, slots):
    s, io = create(tmp_path, mode, slots)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 8)
    incumbents = list(range(8 - slots, 8))
    for _ in range(8):
        for ordinal in range(8):
            read(s, lease, [ordinal])
    assert [b.ordinal for b in s.resident.values()] == incumbents
    assert s.metrics["admission_count"] == 0
    assert s.metrics["admission_rejections"] == (8 - slots) * 8
    assert s.metrics["ram_read_bytes"] == slots * 8 * 64
    assert io.reads == (8 - slots) * 8 * 64
    assert io.writes == (8 - slots) * 64
    s.revoke(p)
    s.release(lease)
    assert not io.files


@pytest.mark.parametrize("mode", ["heat_lru", "adaptive"])
@pytest.mark.parametrize("slots", [1, 4])
def test_uniform_hot_scan_preserves_equal_value_clean_incumbents(tmp_path, mode, slots):
    s, io = create(tmp_path, mode, slots)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 8)
    for b in list(s.resident.values()):
        s._spill(b)
    # Successful reads train equal heat while a writer disables optional fill.
    with s.mutation:
        for _ in range(8):
            for ordinal in range(8):
                read(s, lease, [ordinal])
    incumbents = list(range(8 - slots, 8))
    for ordinal in incumbents:
        read(s, lease, [ordinal])  # Free-space admission of clean entries.
    before = s.snapshot()["metrics"]
    reads, writes = io.reads, io.writes
    for _ in range(8):
        for ordinal in range(8):
            read(s, lease, [ordinal])
    assert [b.ordinal for b in s.resident.values()] == incumbents
    assert all(b.dfs for b in s.resident.values())
    assert s.metrics["admission_count"] == before["admission_count"] == slots
    assert s.metrics["ram_read_bytes"] - before["ram_read_bytes"] == slots * 8 * 64
    assert io.reads - reads == (8 - slots) * 8 * 64
    assert io.writes == writes and s.metrics["admission_dirty_spill_bytes"] == 0
    s.revoke(p)
    s.release(lease)


@pytest.mark.parametrize("mode", ["heat_lru", "adaptive"])
def test_stronger_candidate_admitted_after_equal_value_rejection(tmp_path, mode):
    s, io = create(tmp_path, mode, slots=1)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 2)
    for _ in range(2):
        read(s, lease, [0])
    assert s.policies[p].blocks[0].data is None
    assert s.metrics["admission_dirty_spill_bytes"] == 0
    read(s, lease, [0])
    assert s.policies[p].blocks[0].data is not None
    assert s.metrics["admission_count"] == 1
    assert s.metrics["admission_dirty_spill_bytes"] == 64
    assert io.reads == 3 * 64 and io.writes == 2 * 64
    s.revoke(p)
    s.release(lease)


@pytest.mark.parametrize("mode", ["heat_lru", "adaptive"])
def test_free_capacity_still_admits_on_second_demand(tmp_path, mode):
    s, io = create(tmp_path, mode, slots=1)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 2)
    s._spill(s.policies[p].blocks[1])
    assert s.used == 0
    writes = io.writes
    read(s, lease, [0])
    assert s.policies[p].blocks[0].data is None
    read(s, lease, [0])
    assert s.policies[p].blocks[0].data is not None
    assert s.metrics["admission_count"] == 1
    assert s.metrics["admission_dirty_spill_bytes"] == 0
    assert io.writes == writes and io.reads == 2 * 64
    s.revoke(p)
    s.release(lease)
