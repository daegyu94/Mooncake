import tempfile
from pathlib import Path

import pytest
from tail_store import TailBufferedStore
from test_generation_store import MemoryIO


def make(chunk=16, budget=64, capacity=1000):
    tmp = tempfile.TemporaryDirectory()
    store = TailBufferedStore(
        MemoryIO(), Path(tmp.name) / "kv", capacity, chunk, budget
    )
    return tmp, store


def test_cross_chunk_random_order_duplicates_and_flush():
    tmp, s = make()
    with tmp:
        g = s.create()
        values = [bytes([i]) * n for i, n in enumerate([7, 12, 33, 2, 17])]
        for i, value in enumerate(values):
            s.append(g, [value])
            assert s.get(g, list(reversed(range(i + 1)))) == list(
                reversed(values[: i + 1])
            )
        assert s.get(g, [4, 1, 4]) == [values[4], values[1], values[4]]
        assert s.get(g, []) == []
        s.seal(g)
        s.flush(g)
        assert s.generations[g].tail == b""
        assert s.io.files[s.generations[g].fd] == b"".join(values)
        with pytest.raises(ValueError):
            s.append(g, [b"late"])


def test_expiry_discards_only_unpinned_tail_and_preserves_old_lease():
    tmp, s = make()
    with tmp:
        old = s.create()
        s.append(old, [b"a" * 19])
        lease = s.acquire(old)
        s.invalidate(old)
        assert s.resident_tail_bytes == 3 and s.gc() == 0
        with pytest.raises(KeyError):
            s.get(old, [0])
        assert s.read_lease(lease, [0]) == [b"a" * 19]
        new = s.create()
        s.append(new, [b"b" * 19])
        assert s.get(new, [0]) == [b"b" * 19]
        s.release(lease)
        assert s.metrics["discarded_tail_bytes"] == 3
        assert s.gc() == 1
        s.invalidate(new)
        assert s.metrics["discarded_tail_bytes"] == 6
        assert s.resident_tail_bytes == 0


def test_pressure_flush_does_not_drop_active_or_retired_data():
    tmp, s = make(budget=8)
    with tmp:
        old = s.create()
        s.append(old, [b"a" * 7])
        lease = s.acquire(old)
        s.invalidate(old)
        current = s.create()
        for i in range(5):
            s.append(current, [bytes([i]) * 7])
            assert s.resident_tail_bytes <= 8
        assert s.read_lease(lease, [0]) == [b"a" * 7]
        assert s.get(current, list(range(5))) == [bytes([i]) * 7 for i in range(5)]
        assert s.metrics["pressure_flush_bytes"] > 0
        s.release(lease)
        s.gc()


def test_failure_fails_closed_and_does_not_publish_keys():
    tmp, s = make()
    with tmp:
        g = s.create()
        s.append(g, [b"a"])

        def fail(*args, **kwargs):
            raise OSError("injected write failure")

        s.io.io = fail
        with pytest.raises(OSError):
            s.append(g, [b"b" * 16])
        assert s.generations[g].count == 1
        for operation in [
            lambda: s.get(g, [0]),
            lambda: s.seal(g),
            lambda: s.append(g, [b"c"]),
        ]:
            with pytest.raises(ValueError):
                operation()
        s.invalidate(g)
        assert s.resident_tail_bytes == 0
        assert s.gc() == 1


def test_capacity_includes_retired_and_restart_fails_closed():
    tmp, s = make(capacity=20)
    with tmp:
        g = s.create()
        s.append(g, [b"a" * 19])
        lease = s.acquire(g)
        s.invalidate(g)
        new = s.create()
        with pytest.raises(BufferError):
            s.append(new, [b"bb"])
        s.release(lease)
        s.gc()
        s.append(new, [b"bb"])
        with pytest.raises(FileExistsError):
            TailBufferedStore(s.io, s.root, 100)


def test_failed_multi_put_reserves_capacity(tmp_path):
    class FailingIO(MemoryIO):
        calls = 0

        def io(self, fd, ranges, read=False):
            if not read:
                self.calls += 1
                if self.calls == 2:
                    raise OSError("injected second-write failure")
            return super().io(fd, ranges, read)

    io = FailingIO()
    s = TailBufferedStore(io, tmp_path / "reserve", 8, chunk_bytes=4, tail_budget=8)
    failed = s.create()
    with pytest.raises(OSError):
        s.append(failed, [b"aaaa", b"bbbb"])
    assert s.generations[failed].count == 0
    assert s.generations[failed].cursor == 8
    new = s.create()
    with pytest.raises(BufferError):
        s.append(new, [b"c"])
    s.invalidate(failed)
    s.gc()
    s.append(new, [b"c"])
    assert s.get(new, [0]) == [b"c"]
