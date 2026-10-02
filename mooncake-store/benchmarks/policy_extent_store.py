"""Single-owner research PoC: immutable KV extents with a drained policy fence.

The in-process index is not distributed or durable. Use an exclusive namespace
and a cache-fitting working set. Failed native reads remain cache misses; this
prototype does not implement automatic refill after external eviction.
"""

import ctypes
import threading
from dataclasses import dataclass

from agent_rl_dfs_bench import Buffer


@dataclass(frozen=True)
class BlockLocation:
    extent: str
    offset: int
    size: int
    extent_size: int


class PolicyExtentStore:
    def __init__(self, client, replica, namespace, extent_blocks, device="cpu"):
        if extent_blocks < 1:
            raise ValueError("extent_blocks must be positive")
        self.client = client
        self.replica = replica
        self.namespace = namespace
        self.extent_blocks = extent_blocks
        self.device = device
        self.epoch = 0
        self.index = {}
        self.extents = []
        self.scratch = {}
        self.sequence = 0
        self.lock = threading.RLock()
        self.read_bytes = 0
        self.write_bytes = 0

    def _check_epoch(self, epoch):
        if epoch != self.epoch:
            raise ValueError("stale or future policy epoch")

    def put(self, epoch, keys, buffers, slices=56):
        with self.lock:
            self._check_epoch(epoch)
            if len(keys) != len(buffers) or len(set(keys)) != len(keys):
                raise ValueError("keys must be unique and match buffers")
            missing = [
                (key, buf) for key, buf in zip(keys, buffers) if key not in self.index
            ]
            groups = [
                missing[i : i + self.extent_blocks]
                for i in range(0, len(missing), self.extent_blocks)
            ]
            if not groups:
                return
            names, pointers, sizes, entries = [], [], [], []
            for group in groups:
                name = f"{self.namespace}/policy-{epoch}/extent-{self.sequence}"
                self.sequence += 1
                total = sum(buf.size for _, buf in group)
                offset, addrs, lengths, locations = 0, [], [], []
                for key, buf in group:
                    locations.append(
                        (key, BlockLocation(name, offset, buf.size, total))
                    )
                    part, remainder = divmod(buf.size, slices)
                    if part == 0:
                        raise ValueError("more slices than payload bytes")
                    start = 0
                    for i in range(slices):
                        length = part + (i < remainder)
                        addrs.append(buf.address + start)
                        lengths.append(length)
                        start += length
                    offset += buf.size
                names.append(name)
                pointers.append(addrs)
                sizes.append(lengths)
                entries.append(locations)
            buffers[0].synchronize()
            results = self.client.batch_put_from_multi_buffers(
                names, pointers, sizes, self.replica
            )
            # Retain all attempted names for cleanup after a partial failure.
            self.extents.extend(names)
            if len(results) != len(names) or any(rc != 0 for rc in results):
                raise RuntimeError(f"extent put failed: {results}")
            for locations in entries:
                self.index.update(locations)
            self.write_bytes += sum(sum(v) for v in sizes)

    def get(self, epoch, keys, buffers):
        with self.lock:
            self._check_epoch(epoch)
            if len(keys) != len(buffers):
                raise ValueError("keys must match destinations")
            locations = [self.index[key] for key in keys]
            unique = {location.extent: location.extent_size for location in locations}
            slots, names = [], list(unique)
            for slot, name in enumerate(names):
                size = unique[name]
                direct = next(
                    (
                        dest
                        for loc, dest in zip(locations, buffers)
                        if loc.extent == name and loc.offset == 0 and loc.size == size
                    ),
                    None,
                )
                if direct is not None:
                    direct.clear()
                    slots.append(direct)
                    continue
                cache_key = (slot, size)
                if cache_key not in self.scratch:
                    buf = Buffer(size, self.device)
                    rc = self.client.register_buffer(buf.address, size)
                    if rc != 0:
                        raise RuntimeError(f"scratch registration failed: {rc}")
                    self.scratch[cache_key] = buf
                buf = self.scratch[cache_key]
                buf.clear()
                slots.append(buf)
            if not names:
                return
            slots[0].synchronize()
            results = self.client.batch_get_into_multi_buffers(
                names, [[buf.address] for buf in slots], [[buf.size] for buf in slots]
            )
            if results != [unique[name] for name in names]:
                raise RuntimeError(f"extent restore failed: {results}")
            source = dict(zip(names, slots))
            for location, dest in zip(locations, buffers):
                if dest.size != location.size:
                    raise ValueError("destination must match logical block size")
                buf = source[location.extent]
                if buf.address == dest.address:
                    continue
                if buf.torch is None:
                    ctypes.memmove(
                        dest.address, buf.address + location.offset, location.size
                    )
                else:
                    dest.data.copy_(
                        buf.data[location.offset : location.offset + location.size]
                    )
            buffers[0].synchronize()
            self.read_bytes += sum(unique.values())

    def advance_policy(self, next_epoch):
        # Holding the same lock through native completion drains old operations.
        with self.lock:
            if next_epoch != self.epoch + 1:
                raise ValueError("policy epochs must increase by one")
            old = list(self.extents)
            self.epoch = next_epoch
            self.index.clear()
            if old:
                results = self.client.batch_remove(old, force=True)
                if len(results) != len(old) or any(rc != 0 for rc in results):
                    raise RuntimeError(f"extent reclamation failed: {results}")
            self.extents.clear()
            return old

    def close(self):
        with self.lock:
            for buf in self.scratch.values():
                self.client.unregister_buffer(buf.address)
            self.scratch.clear()
