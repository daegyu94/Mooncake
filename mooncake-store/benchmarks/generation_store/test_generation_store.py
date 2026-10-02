import tempfile
import unittest
from pathlib import Path
from generation_store import GenerationStore


class MemoryIO:
    slot_bytes = 1 << 20

    def __init__(self):
        self.files = {}
        self.closed = []

    def open(self, path):
        path.touch()
        fd = len(self.files)
        self.files[fd] = bytearray()
        return fd

    def close_fd(self, fd):
        self.closed.append(fd)

    def io(self, fd, ranges, read=False):
        data = self.files[fd]
        if read:
            return [bytes(data[off : off + size]) for off, size in ranges]
        for off, value in ranges:
            data.extend(b"\0" * max(0, off + len(value) - len(data)))
            data[off : off + len(value)] = value


class GenerationTest(unittest.TestCase):
    def make(self, compact=True, capacity=100):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        return GenerationStore(MemoryIO(), Path(temp.name) / "new", capacity, compact)

    def test_overlap_fence_and_reader_gc(self):
        s = self.make()
        old = s.create()
        s.append(old, [b"old", b"kv"])
        s.seal(old)
        with self.assertRaises(ValueError):
            s.append(old, [b"late"])
        new = s.create()
        s.append(new, [b"new"])
        lease = s.acquire(old)
        s.invalidate(old)
        with self.assertRaises(KeyError):
            s.get(old, [0])
        self.assertEqual(s.gc(), 0)
        self.assertEqual(s.get(new, [0]), [b"new"])
        self.assertEqual(s.io.files[lease.fd], b"oldkv")
        s.release(lease)
        self.assertEqual(s.gc(), 1)

    def test_capacity_pins_and_no_live_copy(self):
        s = self.make(capacity=5)
        g = s.create()
        s.append(g, [b"12345"])
        lease = s.acquire(g)
        s.invalidate(g)
        n = s.create()
        with self.assertRaises(BufferError):
            s.append(n, [b"x"])
        s.release(lease)
        s.gc()
        s.append(n, [b"x"])
        self.assertEqual(s.metrics["live_copy_bytes"], 0)

    def test_range_equals_whole_extent(self):
        for compact in [False, True]:
            s = self.make(compact)
            g = s.create()
            values = [bytes([i]) * 7 for i in range(5)]
            s.append(g, values)
            for keys in [[0, 4], [1, 2, 3], [], [3, 3]]:
                expected = [values[i] for i in keys]
                self.assertEqual(s.get(g, keys), expected)
                self.assertEqual(s.get(g, keys, whole_extent=3), expected)
            with self.assertRaises(KeyError):
                s.get(g, [5])

    def test_no_restart_republish(self):
        s = self.make()
        g = s.create()
        s.append(g, [b"old"])
        with self.assertRaises(FileExistsError):
            GenerationStore(s.io, s.root, 100)


if __name__ == "__main__":
    unittest.main()
