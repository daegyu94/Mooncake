"""Typed policy regions versus conservatively batched object metadata.

An independent architecture prototype, not the production Mooncake protocol.
Trusted writers publish only completed bytes. Lost owners/leases quarantine
regions indefinitely; no timeout pretends that outstanding DFS IO was canceled.
"""

import secrets
import sys
import threading
import time
from dataclasses import dataclass, field


class StoreError(Exception):
    pass


def footprint(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return 0
    seen.add(id(value))
    total = sys.getsizeof(value)
    if isinstance(value, dict):
        total += sum(footprint(k, seen) + footprint(v, seen) for k, v in value.items())
    elif isinstance(value, (tuple, list, set)):
        total += sum(footprint(v, seen) for v in value)
    elif hasattr(value, "__dict__"):
        total += footprint(value.__dict__, seen)
    return total


@dataclass
class Region:
    stride: int
    capacity: int
    layout: str
    writer: int
    frontier: int = 0
    drained: bool = False


@dataclass
class Policy:
    identity: str
    state: str = "active"
    regions: dict = field(default_factory=dict)
    objects: dict = field(default_factory=dict)
    readers: set = field(default_factory=set)


class Coordinator:
    def __init__(self, mode):
        if mode not in ("objects", "regions"):
            raise ValueError(mode)
        self.mode = mode
        self.boot = secrets.randbits(63)
        self.next_policy = self.next_region = self.next_token = 0
        self.policies = {}
        self.lock = threading.RLock()
        self.metrics = {
            "lock_wait_ns": 0,
            "lock_hold_ns": 0,
            "object_inserts": 0,
            "object_lookups": 0,
            "region_lookups": 0,
            "object_deletes": 0,
            "region_deletes": 0,
            "errors": 0,
        }

    def dispatch(self, op, args):
        if op in ("hello", "stats"):
            with self.lock:
                return self._dispatch(op, args)
        start = time.perf_counter_ns()
        with self.lock:
            acquired = time.perf_counter_ns()
            try:
                result = self._dispatch(op, args)
            except (KeyError, IndexError, TypeError, ValueError, StoreError) as error:
                self.metrics["errors"] += 1
                raise StoreError(str(error)) from error
            finally:
                self.metrics["lock_wait_ns"] += acquired - start
                self.metrics["lock_hold_ns"] += time.perf_counter_ns() - acquired
            return result

    def _dispatch(self, op, a):
        if op == "hello":
            return [self.boot, self.mode]
        if a["boot"] != self.boot:
            raise StoreError("incarnation mismatch")
        if op == "stats":
            return {
                "metrics": self.metrics.copy(),
                "policies": len(self.policies),
                "regions": sum(len(p.regions) for p in self.policies.values()),
                "objects": sum(len(p.objects) for p in self.policies.values()),
                "readers": sum(len(p.readers) for p in self.policies.values()),
                "metadata_bytes": footprint(self.policies),
            }
        if op == "create":
            self.next_policy += 1
            self.policies[self.next_policy] = Policy(a["identity"])
            return self.next_policy
        p = self.policies[a["policy"]]
        if op == "revoke":
            p.state = "retired"
            return True
        if op == "collect":
            if (
                p.state != "retired"
                or p.readers
                or any(not r.drained for r in p.regions.values())
            ):
                return False
            self.metrics["object_deletes"] += len(p.objects)
            self.metrics["region_deletes"] += len(p.regions)
            del self.policies[a["policy"]]
            return True
        if op == "release":
            p.readers.remove(a["lease"])
            return True
        if op == "open":
            if p.state != "active" or p.identity != a["identity"]:
                raise StoreError("policy is not readable for these weights")
            self.next_token += 1
            p.readers.add(self.next_token)
            return self.next_token
        if op == "allocate":
            if p.state != "active" or a["stride"] <= 0 or a["capacity"] <= 0:
                raise StoreError("invalid allocation")
            if a["stride"] * a["capacity"] > (1 << 50):
                raise StoreError("region size overflow")
            self.next_region += 1
            self.next_token += 1
            p.regions[self.next_region] = Region(
                a["stride"], a["capacity"], a["layout"], self.next_token
            )
            return [self.next_region, self.next_token]
        if op in ("publish", "drained"):
            r = p.regions[a["region"]]
            if r.writer != a["writer"]:
                raise StoreError("writer ownership mismatch")
            if op == "drained":
                r.drained = True
                return True
            if p.state != "active" or r.drained:
                raise StoreError("late publication")
            start, end = a["start"], a["end"]
            if start != r.frontier or not start < end <= r.capacity:
                raise StoreError("frontier hole or overflow")
            if self.mode == "objects":
                entries = a["entries"]
                if len(entries) != end - start:
                    raise StoreError("missing object metadata")
                for index, (slot, offset, size) in enumerate(entries, start):
                    if (slot, offset, size) != (index, index * r.stride, r.stride):
                        raise StoreError("invalid typed object geometry")
                for slot, offset, size in entries:
                    p.objects[(a["region"], slot)] = (offset, size)
                self.metrics["object_inserts"] += len(entries)
            r.frontier = end
            return end
        if op in ("lookup", "describe"):
            if a["lease"] not in p.readers:
                raise StoreError("reader lease missing")
            # An already pinned reader may complete after revoke. A new open
            # cannot obtain a lease, and a changed weight identity cannot open.
            if op == "describe":
                self.metrics["region_lookups"] += len(a["regions"])
                return [
                    [
                        rid,
                        p.regions[rid].stride,
                        p.regions[rid].capacity,
                        p.regions[rid].frontier,
                        p.regions[rid].layout,
                    ]
                    for rid in a["regions"]
                ]
            if self.mode != "objects":
                raise StoreError("typed regions do not materialize object metadata")
            self.metrics["object_lookups"] += len(a["keys"])
            return [p.objects[tuple(key)] for key in a["keys"]]
        raise StoreError("unknown operation")


class CompletionFrontier:
    """Bounded-by-inflight interval tracker, not a per-object completion bitmap."""

    def __init__(self, capacity):
        self.capacity, self.reserved, self.committed = capacity, 0, 0
        self.pending = {}
        self.failed = False

    def reserve(self, count):
        if self.failed or count <= 0 or self.reserved + count > self.capacity:
            raise StoreError("cannot reserve")
        start = self.reserved
        self.reserved += count
        self.pending[start] = [count, False]
        return start

    def complete(self, start, count, success=True):
        if start not in self.pending or self.pending[start] != [count, False]:
            raise StoreError("unknown or duplicate completion")
        self.pending[start][1] = True
        if not success:
            self.failed = True
        if self.failed:
            return self.committed
        while self.committed in self.pending and self.pending[self.committed][1]:
            count, _ = self.pending.pop(self.committed)
            self.committed += count
        return self.committed
