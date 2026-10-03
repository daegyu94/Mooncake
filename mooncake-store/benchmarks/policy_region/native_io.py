"""Native USRBIO adapter reused from generation prototype 78831f69.

Owns slots until completion. This is not a new optimization.
"""

import ctypes
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
        self.stats = {
            "read_bytes": 0,
            "write_bytes": 0,
            "submissions": 0,
            "requests": 0,
            "fd_register": 0,
            "fd_deregister": 0,
            "copy_bytes": 0,
        }

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
                buffers = [ctypes.create_string_buffer(value) for _, value in batch]
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
