"""Explicit continuation handles: (boot, policy, region, ordinal, layout).

A caller must retain the immutable weight identity for a ReaderSession. Closing
means all submitted data IO has drained. A lost session is not reclaimed by TTL.
General content-hash discovery requires an additional directory, not modeled as
free or replaced by an ordinal lookup.
"""

import threading
from contextlib import contextmanager

from coordinator import CompletionFrontier, StoreError, footprint


class Writer:
    def __init__(self, rpc, policy, stride, capacity, layout="kv-layout-v1"):
        self.rpc, self.policy = rpc, policy
        self.stride, self.capacity, self.layout = stride, capacity, layout
        self.region, self.token = rpc.call(
            "allocate", policy=policy, stride=stride, capacity=capacity, layout=layout
        )
        self.completion = CompletionFrontier(capacity)
        self.published = 0
        self.closed = False

    def reserve(self, count):
        if self.closed:
            raise StoreError("writer is drained")
        return self.completion.reserve(count)

    def completed(self, start, count, success=True):
        if self.closed:
            raise StoreError("writer is drained")
        end = self.completion.complete(start, count, success)
        if end <= self.published:
            return
        args = {
            "policy": self.policy,
            "region": self.region,
            "writer": self.token,
            "start": self.published,
            "end": end,
        }
        if self.rpc.mode == "objects":
            args["entries"] = [
                [slot, slot * self.stride, self.stride]
                for slot in range(self.published, end)
            ]
        self.rpc.call("publish", **args)
        self.published = end

    def drained(self):
        if self.closed:
            return
        if any(not ready for _, ready in self.completion.pending.values()):
            raise StoreError("cannot release writer with pending completion")
        # Seal locally before RPC, including an ambiguous/lost reply.
        self.closed = True
        # Every reserved batch, including failed IO, must report completion.
        self.rpc.call(
            "drained", policy=self.policy, region=self.region, writer=self.token
        )


class ReaderSession:
    def __init__(self, rpc, policy, identity, mode, layout="kv-layout-v1"):
        if mode not in ("objects", "cached", "regions"):
            raise ValueError(mode)
        self.rpc, self.policy, self.identity, self.mode, self.layout = (
            rpc,
            policy,
            identity,
            mode,
            layout,
        )
        self.lease = rpc.call("open", policy=policy, identity=identity)
        self.cache, self.regions = {}, {}
        self.closed = False
        self.pins = 0
        self.lock = threading.RLock()

    @contextmanager
    def pin(self):
        """Keep the read lease through submitted IO completion, not just lookup."""
        with self.lock:
            if self.closed:
                raise StoreError("reader is closed")
            self.pins += 1
        try:
            yield self
        finally:
            with self.lock:
                self.pins -= 1

    def locations(self, region, ordinals, *, identity=None, boot=None):
        if (
            self.closed
            or (identity is not None and identity != self.identity)
            or (boot is not None and boot != self.rpc.boot)
        ):
            raise StoreError("invalid reader/weight/incarnation")
        if not ordinals:
            return []
        if any(type(o) is not int or o < 0 for o in ordinals):
            raise StoreError("negative ordinal")
        if self.mode == "regions":
            needed = max(ordinals, default=-1)
            desc = self.regions.get(region)
            if desc is None or needed >= desc[3]:
                desc = self.rpc.call(
                    "describe", policy=self.policy, lease=self.lease, regions=[region]
                )[0]
                if desc[4] != self.layout:
                    raise StoreError("layout mismatch")
                self.regions[region] = desc
            _, stride, capacity, frontier, _ = desc
            if not needed < min(capacity, frontier):
                raise StoreError("unpublished or out-of-bounds ordinal")
            return [(o * stride, stride) for o in ordinals]
        keys = [(region, o) for o in ordinals]
        missing = (
            keys
            if self.mode == "objects"
            else list(dict.fromkeys(k for k in keys if k not in self.cache))
        )
        if missing:
            locs = self.rpc.call(
                "lookup", policy=self.policy, lease=self.lease, keys=missing
            )
            if self.mode == "objects":
                return [tuple(loc) for loc in locs]
            self.cache.update((key, tuple(loc)) for key, loc in zip(missing, locs))
        return [self.cache[key] for key in keys]

    def bytes(self):
        return footprint([self.cache, self.regions])

    def close(self):
        with self.lock:
            if self.pins:
                raise StoreError("reader has in-flight IO")
            if not self.closed:
                self.closed = True
                # Lost release reply quarantines the region, never unsafe reuse.
                self.rpc.call("release", policy=self.policy, lease=self.lease)
                self.cache.clear()
                self.regions.clear()
