"""Protocol tests use a memory backend; native validation is separate."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
from coordinator import StoreError
from resident_store import ResidentStore


class MemoryIO:
    def __init__(self):
        self.files = {}
        self.serial = 0
        self.fail = False
        self.writes = 0
        self.reads = 0
        self.enter = None
        self.resume = None

    def open(self, path):
        path.touch()
        self.serial += 1
        self.files[self.serial] = {}
        return self.serial

    def close_fd(self, fd):
        del self.files[fd]

    def io(self, fd, ranges, read=False):
        if not read and self.enter:
            self.enter.set()
            assert self.resume.wait(5)
        if self.fail and not read:
            raise OSError("injected write failure")
        if read:
            self.reads += sum(size for _, size in ranges)
            return [self.files[fd][off] for off, _ in ranges]
        for off, value in ranges:
            self.files[fd][off] = bytes(value)
            self.writes += len(value)
        return []


def create(tmp_path, mode="drain", slots=8):
    io = MemoryIO()
    return ResidentStore(io, tmp_path, slots * 64, mode), io


def fill(s, p, n, start=0):
    for k in range(start, start + n):
        assert s.put(p, bytes([k % 251]) * 64)[0] == k


def read(s, lease, keys):
    with s.read(lease, keys) as result:
        assert result == [bytes([k % 251]) * 64 for k in keys]


@pytest.mark.parametrize("mode", ["eager", "lru", "drain"])
def test_policy_identity_revoke_and_full_payload(tmp_path, mode):
    s, io = create(tmp_path, mode)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 8)
    read(s, lease, list(range(8)))
    with pytest.raises(StoreError):
        s.acquire(p, "B")
    with pytest.raises(StoreError), s.read(replace(lease, boot=lease.boot + 1), [0]):
        pass
    assert not s.revoke(p)
    with pytest.raises(StoreError):
        s.acquire(p, "A")
    with pytest.raises(StoreError):
        s.put(p, b"x" * 64)
    read(s, lease, [0, 7])
    assert s.release(lease)
    assert not s.policies and s.used == 0 and not io.files


@pytest.mark.parametrize("mode", ["lru", "drain"])
def test_spill_failure_preserves_source(tmp_path, mode):
    s, io = create(tmp_path, mode, 1)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 1)
    io.fail = True
    with pytest.raises(OSError):
        s.put(p, b"y" * 64)
    read(s, lease, [0])
    assert s.used == 64
    io.fail = False
    s.revoke(p)
    s.release(lease)


def test_bounded_protection_strict_budget_and_common_cache(tmp_path):
    results = {}
    for mode in ["lru", "drain"]:
        d = tmp_path / mode
        d.mkdir()
        s, io = create(d, mode)
        old = s.create("A", 64)
        lease = s.acquire(old, "A")
        fill(s, old, 8)
        s.revoke(old)
        new = s.create("B", 64)
        nl = s.acquire(new, "B")
        fill(s, new, 8)
        read(s, lease, list(range(8)))
        s.release(lease)
        fill(s, new, 8, 8)
        read(s, nl, list(range(16)))
        s.revoke(new)
        s.release(nl)
        assert s.metrics["peak_resident_bytes"] <= 8 * 64
        results[mode] = io.writes
    assert results == {"lru": 16 * 64, "drain": 12 * 64}


def test_pin_blocks_eviction_and_release(tmp_path):
    s, _io = create(tmp_path, slots=1)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 1)
    with s.read(lease, [0]) as values:
        assert values[0] == b"\0" * 64
        with pytest.raises(BufferError):
            s.put(p, b"a" * 64)
        with pytest.raises(StoreError):
            s.release(lease)
        assert not s.revoke(p)
    assert s.release(lease)
    with pytest.raises(ValueError):
        values[0].tobytes()  # view lifetime enforced


def test_active_hot_extra_reads_are_trace_attributed(tmp_path):
    """No OS/storage counters: count only this owner's exact data requests."""
    costs = {}
    for mode in ["lru", "drain"]:
        root = tmp_path / mode
        root.mkdir()
        s, io = create(root, mode)
        old = s.create("A", 64)
        old_lease = s.acquire(old, "A")
        fill(s, old, 8)
        s.revoke(old)
        new = s.create("B", 64)
        new_lease = s.acquire(new, "B")
        fill(s, new, 8)
        for _ in range(8):
            read(s, new_lease, list(range(8)))
        read(s, old_lease, list(range(8)))
        s.release(old_lease)
        fill(s, new, 8, 8)
        read(s, new_lease, list(range(16)))
        s.revoke(new)
        s.release(new_lease)
        assert io.reads == s.metrics["dfs_read_bytes"]
        costs[mode] = (io.writes // 64, io.reads // 64)
    assert costs == {"lru": (16, 16), "drain": (12, 44)}


def test_migration_finishes_after_revoke_without_republication(tmp_path):
    s, io = create(tmp_path, mode="lru", slots=1)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 1)
    q = s.create("B", 64)
    ql = s.acquire(q, "B")
    io.enter, io.resume = threading.Event(), threading.Event()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(s.put, q, b"\0" * 64)
        assert io.enter.wait(5)
        assert not s.revoke(p)
        # Borrow may start during migration; spill completion cannot free it.
        with s.read(lease, [0]) as values:
            io.resume.set()
            with pytest.raises(BufferError):
                future.result(5)
            assert values[0] == b"\0" * 64 and s.used == 64
        assert s.policies[p].state == "retired"
        with pytest.raises(StoreError):
            s.acquire(p, "A")
    assert s.release(lease)
    s.revoke(q)
    s.release(ql)


def test_late_new_put_is_fenced(tmp_path):
    s, io = create(tmp_path, mode="eager")
    p = s.create("A", 64)
    io.enter, io.resume = threading.Event(), threading.Event()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(s.put, p, b"x" * 64)
        assert io.enter.wait(5)
        assert not s.revoke(p)
        io.resume.set()
        with pytest.raises(StoreError):
            future.result(5)
    assert not s.policies and not io.files and s.metrics["accepted_bytes"] == 0


def test_all_retired_pressure_has_fallback(tmp_path):
    io = MemoryIO()
    s = ResidentStore(io, tmp_path, 128, "drain", 1.0)
    a = s.create("A", 64)
    lease = s.acquire(a, "A")
    fill(s, a, 2)
    s.revoke(a)
    b = s.create("B", 64)
    fill(s, b, 1)
    assert s.metrics["protected_fallbacks"] == 1
    read(s, lease, [0, 1])
    s.release(lease)
    s.revoke(b)


def test_clean_first_and_lru_hit_are_common(tmp_path):
    for mode in ["lru", "drain"]:
        d = tmp_path / mode
        d.mkdir()
        s, _io = create(d, mode, 2)
        p = s.create("A", 64)
        lease = s.acquire(p, "A")
        fill(s, p, 2)
        read(s, lease, [0])
        fill(s, p, 1, 2)
        assert s.policies[p].blocks[1].dfs
        assert not s.policies[p].blocks[0].dfs
        s.revoke(p)
        s.release(lease)


def test_selection_reservation_gap_revalidates(tmp_path):
    s, io = create(tmp_path, slots=1)
    p = s.create("A", 64)
    lease = s.acquire(p, "A")
    fill(s, p, 1)
    with s.lock:
        victim = s._victim()
    with s.read(lease, [0]):
        assert s._spill(victim) is False
    s.revoke(p)
    s.release(lease)
    assert s._spill(victim) is False
    assert not s.policies and not io.files


def test_failed_spill_retries_deferred_collect(tmp_path):
    s, io = create(tmp_path, mode="lru", slots=1)
    old = s.create("A", 64)
    lease = s.acquire(old, "A")
    fill(s, old, 1)
    new = s.create("B", 64)
    io.enter, io.resume = threading.Event(), threading.Event()
    io.fail = True
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(s.put, new, b"x" * 64)
        assert io.enter.wait(5)
        s.revoke(old)
        assert not s.release(lease)
        io.resume.set()
        with pytest.raises(OSError):
            future.result(5)
    assert old not in s.policies and not io.files
    s.revoke(new)
