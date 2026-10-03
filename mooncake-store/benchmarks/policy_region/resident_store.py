"""Volatile region ownership with bounded protection for draining policies.

The ordinary LRU baseline has identical invalidation, read pins, clean-first
selection and acceptance semantics. This is a single-owner research prototype;
RAM views are borrowed until context exit, never remote address capabilities.
"""

import secrets
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field

from coordinator import StoreError


@dataclass
class Block:
    policy: int
    ordinal: int
    size: int
    data: bytes | None
    dfs: bool = False
    pins: int = 0
    moving: bool = False


@dataclass
class Generation:
    identity: str
    stride: int
    state: str = "active"
    blocks: list = field(default_factory=list)
    leases: dict = field(default_factory=dict)
    writers: int = 0
    io_refs: int = 0
    fd: int | None = None
    path: object = None


@dataclass(frozen=True)
class Lease:
    boot: int
    policy: int
    token: int
    identity: str


class ResidentStore:
    def __init__(self, io, root, budget, mode, protected_fraction=0.5):
        if mode not in ("eager", "lru", "drain") or budget < 0:
            raise ValueError("invalid mode or budget")
        if not 0 <= protected_fraction <= 1:
            raise ValueError("invalid protected fraction")
        self.io, self.root, self.budget, self.mode = io, root, budget, mode
        self.protected_limit = int(budget * protected_fraction)
        self.boot, self.serial, self.token = secrets.randbits(63), 0, 0
        self.policies, self.resident = {}, OrderedDict()
        self.lock, self.mutation = threading.RLock(), threading.Lock()
        self.used = 0
        self.clock = time.perf_counter_ns()
        self.metrics = dict.fromkeys(
            [
                "accepted_bytes",
                "put_count",
                "get_count",
                "ram_read_bytes",
                "dfs_read_bytes",
                "spill_bytes",
                "spill_count",
                "discard_dirty_bytes",
                "clean_evictions",
                "victim_scans",
                "peak_resident_bytes",
                "put_copy_bytes",
                "protected_fallbacks",
                "spill_failures",
                "late_put_rejections",
                "collect_count",
                "revoke_ns",
                "gc_ns",
                "spill_ns",
                "peak_policies",
                "protected_byte_ns",
                "retired_dirty_byte_ns",
                "resident_byte_ns",
                "read_pin_rejections",
                "account_cpu_ns",
            ],
            0,
        )

    def _account(self):
        cpu = time.thread_time_ns()
        now = time.perf_counter_ns()
        retired = protected = 0
        # The same diagnostic scan is used in every mode. Its CPU is exposed.
        for b in self.resident.values():
            if not b.dfs and self.policies[b.policy].state == "retired":
                retired += b.size
                if (
                    not b.pins
                    and not b.moving
                    and protected + b.size <= self.protected_limit
                ):
                    protected += b.size
        self.metrics["resident_byte_ns"] += self.used * (now - self.clock)
        if self.mode == "drain":
            self.metrics["protected_byte_ns"] += protected * (now - self.clock)
        self.metrics["retired_dirty_byte_ns"] += retired * (now - self.clock)
        self.clock = now
        self.metrics["account_cpu_ns"] += time.thread_time_ns() - cpu

    def create(self, identity, stride):
        with self.lock:
            if not identity or stride <= 0:
                raise ValueError("identity/geometry required")
            self.serial += 1
            self.policies[self.serial] = Generation(
                identity, stride, path=self.root / f"{self.boot}-{self.serial}.kv"
            )
            self.metrics["peak_policies"] = max(
                self.metrics["peak_policies"], len(self.policies)
            )
            return self.serial

    def acquire(self, policy, identity):
        with self.lock:
            g = self.policies[policy]
            if g.state != "active" or g.identity != identity:
                raise StoreError("policy identity is not active")
            self.token += 1
            g.leases[self.token] = 0
            return Lease(self.boot, policy, self.token, identity)

    def _lease(self, lease):
        if lease.boot != self.boot or lease.policy not in self.policies:
            raise StoreError("invalid incarnation/policy")
        g = self.policies[lease.policy]
        if g.identity != lease.identity or lease.token not in g.leases:
            raise StoreError("invalid read lease")
        return g

    def _forget_ram(self, b):
        assert b.pins == 0 and not b.moving
        self._account()
        self.resident.pop((b.policy, b.ordinal))
        self.used -= b.size
        b.data = None

    def _collect(self, policy):
        g = self.policies[policy]
        if g.state != "retired" or g.leases or g.writers or g.io_refs:
            return False
        start = time.perf_counter_ns()
        for b in g.blocks:
            assert not b.pins and not b.moving
            if b.data is not None:
                if not b.dfs:
                    self.metrics["discard_dirty_bytes"] += b.size
                self._forget_ram(b)
        if g.fd is not None:
            self.io.close_fd(g.fd)
            g.path.unlink()
        del self.policies[policy]
        self.metrics["collect_count"] += 1
        self.metrics["gc_ns"] += time.perf_counter_ns() - start
        return True

    def revoke(self, policy):
        start = time.perf_counter_ns()
        with self.lock:
            self._account()
            self.policies[policy].state = "retired"
            self.metrics["revoke_ns"] += time.perf_counter_ns() - start
            return self._collect(policy)

    def release(self, lease):
        with self.lock:
            g = self._lease(lease)
            if g.leases[lease.token]:
                self.metrics["read_pin_rejections"] += 1
                raise StoreError("read operations still own buffers")
            del g.leases[lease.token]
            return self._collect(lease.policy)

    def _victim(self):
        candidates = [b for b in self.resident.values() if not b.pins and not b.moving]
        self.metrics["victim_scans"] += len(self.resident)
        if not candidates:
            raise BufferError("all resident data operations are pinned")
        clean = next((b for b in candidates if b.dfs), None)
        if clean is not None:
            return clean
        if self.mode != "drain":
            return candidates[0]
        protected = set()
        size = 0
        for b in candidates:
            if (
                self.policies[b.policy].state == "retired"
                and size + b.size <= self.protected_limit
            ):
                protected.add((b.policy, b.ordinal))
                size += b.size
        candidate = next(
            (b for b in candidates if (b.policy, b.ordinal) not in protected), None
        )
        if candidate is not None:
            return candidate
        self.metrics["protected_fallbacks"] += 1
        return candidates[0]  # Protection is a preference, not an unbounded pin.

    def _write(self, g, offset, data):
        if g.fd is None:
            g.fd = self.io.open(g.path)
        self.io.io(g.fd, [(offset, data)])

    def _spill(self, b):
        # mutation lock serializes writers/migrations, not readers or revoke.
        with self.lock:
            # Selection and reservation are separated by a lock handoff.
            # Release/collect or a new RAM borrower may win that race.
            if self.resident.get((b.policy, b.ordinal)) is not b or b.pins or b.moving:
                return False
            if b.dfs:
                self.metrics["clean_evictions"] += 1
                self._forget_ram(b)
                return
            g = self.policies[b.policy]
            self._account()
            b.moving = True
            g.io_refs += 1
        start = time.perf_counter_ns()
        try:
            self._write(g, b.ordinal * b.size, b.data)
        except Exception:
            with self.lock:
                self._account()
                b.moving = False
                g.io_refs -= 1
                self.metrics["spill_failures"] += 1
                # The backend raises only after all submitted IO has drained.
                self._collect(b.policy)
            raise
        with self.lock:
            self.metrics["spill_ns"] += time.perf_counter_ns() - start
            self.metrics["spill_bytes"] += b.size
            self.metrics["spill_count"] += 1
            self._account()
            b.dfs = True
            b.moving = False
            g.io_refs -= 1
            # A RAM borrower may have arrived while storage IO was in flight.
            if b.pins == 0:
                self._forget_ram(b)
            self._collect(b.policy)

    def put(self, policy, value):
        with self.mutation:
            with self.lock:
                g = self.policies[policy]
                if g.state != "active" or len(value) != g.stride:
                    raise StoreError("inactive policy or incompatible geometry")
                g.writers += 1
                ordinal = len(g.blocks)
            try:
                if len(value) <= self.budget:
                    while True:
                        with self.lock:
                            if self.used + len(value) <= self.budget:
                                break
                            victim = self._victim()
                        self._spill(victim)
                owned = memoryview(value).tobytes()
                self.metrics["put_copy_bytes"] += len(value)
                on_disk = self.mode == "eager" or len(value) > self.budget
                if on_disk:
                    self._write(g, ordinal * g.stride, owned)
                with self.lock:
                    if g.state != "active":
                        self.metrics["late_put_rejections"] += 1
                        raise StoreError("publication fenced after revoke")
                    b = Block(
                        policy,
                        ordinal,
                        g.stride,
                        owned if len(value) <= self.budget else None,
                        on_disk,
                    )
                    g.blocks.append(b)
                    if b.data is not None:
                        self._account()
                        self.resident[(policy, ordinal)] = b
                        self.used += b.size
                        self.metrics["peak_resident_bytes"] = max(
                            self.metrics["peak_resident_bytes"], self.used
                        )
                    self.metrics["accepted_bytes"] += len(value)
                    self.metrics["put_count"] += 1
                    return ordinal, "dfs-complete" if on_disk else "resident-only"
            finally:
                with self.lock:
                    g.writers -= 1
                    self._collect(policy)

    @contextmanager
    def read(self, lease, ordinals, io=None):
        with self.lock:
            g = self._lease(lease)
            if any(type(o) is not int or o < 0 or o >= len(g.blocks) for o in ordinals):
                raise StoreError("unpublished ordinal")
            self._account()
            blocks = [g.blocks[o] for o in ordinals]
            g.leases[lease.token] += 1
            g.io_refs += 1
            values = []
            ranges = []
            positions = []
            for i, b in enumerate(blocks):
                b.pins += 1
                if b.data is not None:
                    self.resident.move_to_end((b.policy, b.ordinal))
                    values.append(memoryview(b.data))
                    self.metrics["ram_read_bytes"] += b.size
                else:
                    assert b.dfs
                    values.append(None)
                    positions.append(i)
                    ranges.append((b.ordinal * b.size, b.size))
                    self.metrics["dfs_read_bytes"] += b.size
            self.metrics["get_count"] += len(blocks)
        try:
            if ranges:
                data = (io or self.io).io(g.fd, ranges, read=True)
                for i, value in zip(positions, data):
                    values[i] = memoryview(value)
            yield values
        finally:
            with self.lock:
                for v in values:
                    if v is not None:
                        v.release()
                self._account()
                for b in blocks:
                    b.pins -= 1
                g.leases[lease.token] -= 1
                g.io_refs -= 1
                self._collect(lease.policy)

    def snapshot(self):
        with self.lock:
            self._account()
            return {
                "metrics": self.metrics.copy(),
                "resident_bytes": self.used,
                "policies": len(self.policies),
                "fds": sum(g.fd is not None for g in self.policies.values()),
                "dirty_bytes": sum(b.size for b in self.resident.values() if not b.dfs),
                "pinned_bytes": sum(b.size for b in self.resident.values() if b.pins),
                "budget": self.budget,
                "mode": self.mode,
            }
