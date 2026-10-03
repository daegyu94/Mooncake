"""CPython-owned outputs with one direct gather/scatter registered-slot copy.

Private PyBytes buffers are initialized fully before they can be returned or
hashed. Sources are immutable bytes retained through synchronous completion.
The registered IOV capacity and single-owner ring lock match NativeIO.
"""

import ctypes

from generation_store import NativeIO

_new_bytes = ctypes.pythonapi.PyBytes_FromStringAndSize
_new_bytes.argtypes = [ctypes.c_void_p, ctypes.c_ssize_t]
_new_bytes.restype = ctypes.py_object
_bytes_pointer = ctypes.pythonapi.PyBytes_AsString
_bytes_pointer.argtypes = [ctypes.py_object]
_bytes_pointer.restype = ctypes.c_void_p


def allocate_output(size):
    return _new_bytes(None, size)


def pointer(value, offset=0):
    if type(value) is not bytes or not 0 <= offset <= len(value):
        raise ValueError("immutable owned bytes and valid offset required")
    return _bytes_pointer(value) + offset


class VectorIO(NativeIO):
    def __init__(self, library, mount, depth=8, slot_bytes=4 << 20, read_plan="exact"):
        if read_plan not in ("exact", "adjacent"):
            raise ValueError("invalid native vector read plan")
        super().__init__(library, mount, depth, slot_bytes)
        self.read_plan = read_plan
        arguments = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        for name in ("vs_write", "vs_read"):
            function = getattr(self.lib, name)
            function.argtypes = arguments
            function.restype = ctypes.c_long
        self.lib.vs_copy.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_uint64),
        ]
        self.lib.vs_copy.restype = ctypes.c_uint64
        self.stats.update(gather_copy_bytes=0, scatter_copy_bytes=0, ram_copy_bytes=0)

    def _batch(self, fd, ranges, fragments, read):
        """Fragments: (request_index, slot_offset, length, Python byte pointer)."""
        n, m = len(ranges), len(fragments)
        offsets = (ctypes.c_uint64 * n)(*(off for off, _ in ranges))
        sizes = (ctypes.c_uint64 * n)(*(size for _, size in ranges))
        requests = (ctypes.c_int * m)(*(item[0] for item in fragments))
        starts = (ctypes.c_uint64 * m)(*(item[1] for item in fragments))
        lengths = (ctypes.c_uint64 * m)(*(item[2] for item in fragments))
        ptrs = (ctypes.c_void_p * m)(*(item[3] for item in fragments))
        copied = ctypes.c_uint64()
        count = (self.lib.vs_read if read else self.lib.vs_write)(
            self.handle,
            fd,
            n,
            offsets,
            sizes,
            m,
            requests,
            starts,
            lengths,
            ptrs,
            ctypes.byref(copied),
        )
        self.stats["copy_bytes"] += copied.value
        self.stats["scatter_copy_bytes" if read else "gather_copy_bytes"] += (
            copied.value
        )
        if count != sum(size for _, size in ranges):
            raise OSError("USRBIO short/error completion: " + str(count))
        self.stats["read_bytes" if read else "write_bytes"] += count
        self.stats["requests"] += n
        self.stats["submissions"] += 1

    def write_parts(self, fd, ranges):
        """Ranges contain (file_offset, [(immutable_bytes, start, length), ...])."""
        with self.lock:
            for start in range(0, len(ranges), self.depth):
                batch = ranges[start : start + self.depth]
                native, fragments = [], []
                for index, (off, parts) in enumerate(batch):
                    cursor = 0
                    for value, begin, length in parts:
                        if begin < 0 or length <= 0 or begin + length > len(value):
                            raise ValueError("invalid source fragment")
                        fragments.append((index, cursor, length, pointer(value, begin)))
                        cursor += length
                    native.append((off, cursor))
                self._batch(fd, native, fragments, False)

    def read_into(self, fd, ranges, pieces, outputs):
        """pieces per output: (range_index, range_offset, size, output_offset)."""
        if len(pieces) != len(outputs):
            raise ValueError("scatter output count mismatch")
        by_range = [[] for _ in ranges]
        for output, spans in zip(outputs, pieces):
            for index, source, length, destination in spans:
                if (
                    not 0 <= index < len(ranges)
                    or source < 0
                    or length <= 0
                    or source + length > ranges[index][1]
                    or destination < 0
                    or destination + length > len(output)
                ):
                    raise ValueError("invalid scatter fragment")
                by_range[index].append((source, length, pointer(output, destination)))
        with self.lock:
            for start in range(0, len(ranges), self.depth):
                batch = ranges[start : start + self.depth]
                fragments = [
                    (index, off, length, destination)
                    for index in range(len(batch))
                    for off, length, destination in by_range[start + index]
                ]
                self._batch(fd, batch, fragments, True)

    def copy_into(self, fragments):
        """(source_bytes, source_offset, size, destination_bytes, dest_offset)."""
        n = len(fragments)
        sources = (ctypes.c_void_p * n)(*(pointer(a, b) for a, b, _, _, _ in fragments))
        destinations = (ctypes.c_void_p * n)(
            *(pointer(d, e) for _, _, _, d, e in fragments)
        )
        sizes = (ctypes.c_uint64 * n)(*(size for _, _, size, _, _ in fragments))
        for source, off, length, output, end in fragments:
            if (
                off < 0
                or length < 0
                or off + length > len(source)
                or end < 0
                or end + length > len(output)
            ):
                raise ValueError("invalid owned-buffer copy")
        copied = self.lib.vs_copy(n, sources, destinations, sizes)
        self.stats["ram_copy_bytes"] += copied
        return copied

    def copy_parts(self, parts):
        result = allocate_output(sum(length for _, _, length in parts))
        fragments, offset = [], 0
        for value, begin, length in parts:
            fragments.append((value, begin, length, result, offset))
            offset += length
        self.copy_into(fragments)
        return result

    def io(self, fd, ranges, read=False):
        if not read:
            self.write_parts(
                fd, [(off, [(value, 0, len(value))]) for off, value in ranges]
            )
            return []
        outputs = [allocate_output(size) for _, size in ranges]
        if self.read_plan == "adjacent":
            # Lazy import avoids the stream store's adapter dependency cycle.
            # Physical bytes count unique ranges; duplicate delivered outputs
            # still receive their own private buffers and scatter copies.
            from stream_store import plan_adjacent

            native_ranges, pieces, _ = plan_adjacent(ranges, self.slot_bytes)
            self.read_into(fd, native_ranges, pieces, outputs)
            return outputs
        self.read_into(
            fd,
            ranges,
            [[(i, 0, size, 0)] for i, (_, size) in enumerate(ranges)],
            outputs,
        )
        return outputs
