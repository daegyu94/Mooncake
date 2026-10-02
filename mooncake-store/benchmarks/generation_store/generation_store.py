"""Single-owner generational segment prototype over native 3FS USRBIO.

Independent research data path, not the production Mooncake master protocol.
IDs are dense rollout-local ordinals; arbitrary prefix hashes need a directory.
Crash recovery is fail-closed: persisted data is never republished implicitly.
"""

import array
import ctypes
from pathlib import Path
import threading


class NativeIO:
    def __init__(self, library, mount, depth=1, slot_bytes=4 << 20):
        self.lib = ctypes.CDLL(str(library))
        self.lib.gs_create.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_int]
        self.lib.gs_create.restype = ctypes.c_void_p
        self.lib.gs_destroy.argtypes = [ctypes.c_void_p]
        self.lib.gs_open.argtypes = [ctypes.c_char_p]
        self.lib.gs_close.argtypes = [ctypes.c_int]
        self.lib.gs_io.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        self.lib.gs_io.restype = ctypes.c_long
        self.handle = self.lib.gs_create(str(mount).encode(), slot_bytes, depth)
        if not self.handle:
            raise RuntimeError("USRBIO initialization failed")
        self.depth, self.slot_bytes = depth, slot_bytes
        self.lock = threading.Lock()
        self.stats = dict(
            read_bytes=0,
            write_bytes=0,
            submissions=0,
            requests=0,
            fd_register=0,
            fd_deregister=0,
            copy_bytes=0,
        )

    def open(self, path):
        fd = self.lib.gs_open(str(path).encode())
        if fd < 0:
            raise OSError(-fd, str(path))
        self.stats["fd_register"] += 1
        return fd

    def close_fd(self, fd):
        self.lib.gs_close(fd)
        self.stats["fd_deregister"] += 1

    def io(self, fd, ranges, read=False):
        with self.lock:
            result = []
            for start in range(0, len(ranges), self.depth):
                batch = ranges[start : start + self.depth]
                buffers = [
                    ctypes.create_string_buffer(value if not read else value)
                    for _, value in batch
                ]
                sizes = [len(value) if not read else value for _, value in batch]
                offsets = (ctypes.c_uint64 * len(batch))(*(off for off, _ in batch))
                lengths = (ctypes.c_uint64 * len(batch))(*sizes)
                ptrs = (ctypes.c_void_p * len(batch))(
                    *(ctypes.addressof(b) for b in buffers)
                )
                count = self.lib.gs_io(
                    self.handle, fd, int(read), len(batch), offsets, lengths, ptrs
                )
                if count != sum(sizes):
                    raise OSError("USRBIO short/error completion: " + str(count))
                self.stats["read_bytes" if read else "write_bytes"] += count
                self.stats["requests"] += len(batch)
                self.stats["submissions"] += 1
                self.stats["copy_bytes"] += count
                if read:
                    result.extend(b.raw[:size] for b, size in zip(buffers, sizes))
            return result

    def close(self):
        self.lib.gs_destroy(self.handle)
        self.handle = None


class Generation:
    def __init__(self, path, fd, compact):
        self.path, self.fd = path, fd
        self.compact = compact
        self.index = array.array("Q") if compact else {}
        self.count = self.cursor = self.readers = 0
        self.state = "active"

    def append_index(self, offset, size):
        if self.compact:
            self.index.extend([offset, size])
        else:
            self.index[self.count] = (offset, size)
        self.count += 1

    def location(self, key):
        if key < 0 or key >= self.count:
            raise KeyError(key)
        return (
            (self.index[2 * key], self.index[2 * key + 1])
            if self.compact
            else self.index[key]
        )


class GenerationStore:
    def __init__(self, io, root, capacity, compact=True):
        self.io, self.root, self.capacity = io, Path(root), capacity
        self.root.mkdir(exist_ok=False)
        self.compact = compact
        self.generations, self.retired = {}, {}
        self.lock = threading.RLock()
        self.next_id = 0
        self.metrics = dict(
            invalidations=0,
            object_deletes=0,
            segment_unlinks=0,
            live_copy_bytes=0,
            capacity_rejections=0,
        )

    def create(self):
        with self.lock:
            gid = self.next_id
            self.next_id += 1
            path = self.root / f"g{gid}.data"
            self.generations[gid] = Generation(path, self.io.open(path), self.compact)
            return gid

    def append(self, gid, values):
        # Holds fence through completion: transition cannot publish partial PUT.
        with self.lock:
            g = self.generations[gid]
            if g.state != "active":
                raise ValueError("late PUT to sealed generation")
            blob = b"".join(values)
            used = sum(
                x.cursor
                for x in list(self.generations.values()) + list(self.retired.values())
            )
            if used + len(blob) > self.capacity:
                self.metrics["capacity_rejections"] += 1
                raise BufferError("capacity includes pinned retired generations")
            start = g.cursor
            ranges = [
                (start + i, blob[i : i + self.io.slot_bytes])
                for i in range(0, len(blob), self.io.slot_bytes)
            ]
            self.io.io(g.fd, ranges)
            for value in values:
                g.append_index(g.cursor, len(value))
                g.cursor += len(value)
            return list(range(g.count - len(values), g.count))

    def seal(self, gid):
        with self.lock:
            self.generations[gid].state = "sealed"

    def acquire(self, gid):
        with self.lock:
            g = self.generations[gid]
            g.readers += 1
            return g

    def release(self, g):
        with self.lock:
            if g.readers <= 0:
                raise ValueError("lease underflow")
            g.readers -= 1

    def get(self, gid, keys, whole_extent=0, channel=None):
        with self.lock:
            g = self.acquire(gid)
        try:
            io = channel or self.io
            locs = [g.location(key) for key in keys]
            if not whole_extent:
                return io.io(g.fd, locs, read=True)
            # P04-like whole-extent staging on the exact same segment layout.
            groups = {}
            for key in keys:
                first = key // whole_extent * whole_extent
                last = min(first + whole_extent, g.count) - 1
                off, _ = g.location(first)
                end, size = g.location(last)
                groups[first] = (off, end + size - off)
            staging = {}
            for first, (off, size) in groups.items():
                chunks = [
                    (off + i, min(self.io.slot_bytes, size - i))
                    for i in range(0, size, self.io.slot_bytes)
                ]
                staging[first] = b"".join(io.io(g.fd, chunks, read=True))
            return [
                staging[key // whole_extent * whole_extent][
                    off - g.location(key // whole_extent * whole_extent)[0] : off
                    - g.location(key // whole_extent * whole_extent)[0]
                    + size
                ]
                for key, (off, size) in zip(keys, locs)
            ]
        finally:
            self.release(g)

    def invalidate(self, gid, individual=False):
        with self.lock:
            if individual and self.generations[gid].readers:
                raise ValueError("individual cleanup must drain readers")
            g = self.generations.pop(gid)
            g.state = "retired"
            self.retired[gid] = g
            self.metrics["invalidations"] += 1
            if individual:
                for key in range(g.count):
                    # Control comparison: per-object metadata cleanup only.
                    self.metrics["object_deletes"] += 1
                    if not g.compact:
                        del g.index[key]

    def gc(self):
        # Explicit completion pump: called off transition path; no extra worker.
        with self.lock:
            ready = [(gid, g) for gid, g in self.retired.items() if not g.readers]
            for gid, g in ready:
                self.io.close_fd(g.fd)
                g.path.unlink()
                del self.retired[gid]
                self.metrics["segment_unlinks"] += 1
            return len(ready)
