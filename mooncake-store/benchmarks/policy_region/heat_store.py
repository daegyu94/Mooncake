"""Online read admission and bounded reader-drain credit for volatile KV.

Both policies use the same frequency sketch, cost-ranked victims and successful
DFS-read admission. Only adaptive mode adds a credit from completed policy
drains and suppresses it on observed/predicted active reuse. No future trace,
reader deadline or durable receipt is used. This is a single-owner PoC.
"""

import time

from resident_store import ResidentStore


class RecentFrequency:
    """Fixed-size decaying sketch; collisions affect policy, never identity."""

    def __init__(self, width=4096):
        if width < 16 or width & (width - 1):
            raise ValueError("power-of-two sketch width required")
        self.width = width
        self.rows = [bytearray(width) for _ in range(4)]
        self.samples = self.decays = 0

    def _indices(self, key):
        policy, ordinal = key
        value = (policy * 0x9E3779B185EBCA87 + ordinal * 0xC2B2AE3D27D4EB4F) & (
            (1 << 64) - 1
        )
        for shift, salt in zip([13, 23, 33, 43], [17, 29, 43, 71]):
            yield ((value ^ (value >> shift)) * salt) & (self.width - 1)

    def estimate(self, key):
        return min(row[i] for row, i in zip(self.rows, self._indices(key)))

    def record(self, key):
        for row, i in zip(self.rows, self._indices(key)):
            row[i] = min(15, row[i] + 1)
        self.samples += 1
        if self.samples >= self.width * 4:
            for row in self.rows:
                for i, value in enumerate(row):
                    row[i] = value // 2
            self.samples = 0
            self.decays += 1


class HeatResidentStore(ResidentStore):
    """Cost/frequency baseline or lifetime-aware admission on the same budget."""

    def __init__(self, io, root, budget, mode="adaptive", protected_fraction=0.5):
        if mode not in ["heat_lru", "adaptive"]:
            raise ValueError("invalid heat policy")
        super().__init__(io, root, budget, "lru", protected_fraction)
        self.mode = mode
        self.frequency = RecentFrequency()
        self.drain_samples = 0
        self.drain_forecast = None
        self.heat_forecast = None
        self.metrics.update(
            admission_bytes=0,
            admission_count=0,
            admission_rejections=0,
            admission_busy=0,
            admission_failures=0,
            admission_stale_rejections=0,
            admission_dirty_spill_bytes=0,
            active_repeat_reads=0,
            protection_decisions=0,
            protection_guarded=0,
            cold_start_decisions=0,
            forecast_samples=0,
            heat_cpu_ns=0,
        )

    def create(self, identity, stride):
        with self.lock:
            policy = super().create(identity, stride)
            g = self.policies[policy]
            g.active_reads = g.active_repeats = 0
            g.hot_guard = False
            g.retire_progress = None
            g.overlapped = False
            return policy

    def _account(self):
        start, old_clock = time.thread_time_ns(), self.clock
        before = self.metrics["account_cpu_ns"]
        super()._account()
        base_cpu = self.metrics["account_cpu_ns"] - before
        if self.mode == "adaptive":
            candidates = [
                b for b in self.resident.values() if not b.pins and not b.moving
            ]
            protected = self._protected(candidates, count=False)
            size = sum(b.size for b in candidates if (b.policy, b.ordinal) in protected)
            self.metrics["protected_byte_ns"] += size * (self.clock - old_clock)
        self.metrics["account_cpu_ns"] += time.thread_time_ns() - start - base_cpu

    def revoke(self, policy):
        with self.lock:
            g = self.policies[policy]
            if g.state == "active":
                repeat = g.active_repeats / max(1, g.active_reads)
                self.heat_forecast = (
                    repeat
                    if self.heat_forecast is None
                    else 0.5 * self.heat_forecast + 0.5 * repeat
                )
                g.retire_progress = self.metrics["accepted_bytes"]
                g.overlapped = bool(g.leases)
            return super().revoke(policy)

    def _collect(self, policy):
        g = self.policies[policy]
        ready = g.state == "retired" and not (g.leases or g.writers or g.io_refs)
        if ready and g.overlapped:
            delay = self.metrics["accepted_bytes"] - g.retire_progress
            self.drain_forecast = (
                delay
                if self.drain_forecast is None
                else 0.5 * self.drain_forecast + 0.5 * delay
            )
            self.drain_samples += 1
            self.metrics["forecast_samples"] += 1
            g.overlapped = False
        return super()._collect(policy)

    def _observe_read(self, g, blocks):
        start = time.thread_time_ns()
        for b in blocks:
            key = self._frequency_key(b)
            repeated = self.frequency.estimate(key) > 0
            self.frequency.record(key)
            if g.state == "active":
                g.active_reads += 1
                g.active_repeats += repeated
                self.metrics["active_repeat_reads"] += repeated
        # React on observed repetition; the first scan is not future hotness.
        g.hot_guard = g.active_repeats / max(1, g.active_reads) >= 0.25
        self.metrics["heat_cpu_ns"] += time.thread_time_ns() - start

    def _credit(self, g):
        if self.mode != "adaptive" or not self.drain_samples or not self.drain_forecast:
            return 0.0
        if (self.heat_forecast or 0) >= 0.25 or any(
            active.state == "active" and active.hot_guard
            for active in self.policies.values()
        ):
            return 0.0
        age = self.metrics["accepted_bytes"] - g.retire_progress
        # EWMA is past completed byte progress, not a predicted wall deadline.
        return max(0.0, 1.0 - age / (2 * self.drain_forecast))

    def _protected(self, candidates, count=True):
        protected, used = {}, 0
        if self.mode != "adaptive":
            return protected
        if not self.drain_samples:
            self.metrics["cold_start_decisions"] += count
            return protected
        for b in candidates:
            g = self.policies[b.policy]
            if g.state != "retired" or b.dfs or used + b.size > self.protected_limit:
                continue
            credit = self._credit(g)
            if credit:
                protected[(b.policy, b.ordinal)] = credit
                used += b.size
        if count:
            self.metrics["protection_decisions"] += bool(protected)
            self.metrics["protection_guarded"] += not bool(protected)
        return protected

    def _score(self, b, protected):
        key = (b.policy, b.ordinal)
        reuse = max(0, min(8, self.frequency.estimate(self._frequency_key(b))) - 1)
        return (0 if b.dfs else 1) + reuse + protected.get(key, 0)

    def _frequency_key(self, block):
        # Active reuse does not predict reuse after new acquisition is fenced.
        retired = self.policies[block.policy].state == "retired"
        return (block.policy * 2 + retired, block.ordinal)

    def _victim(self):
        start = time.thread_time_ns()
        candidates = [b for b in self.resident.values() if not b.pins and not b.moving]
        self.metrics["victim_scans"] += len(self.resident)
        if not candidates:
            raise BufferError("all resident data operations are pinned")
        protected = self._protected(candidates)
        victim = min(candidates, key=lambda b: self._score(b, protected))
        self.metrics["heat_cpu_ns"] += time.thread_time_ns() - start
        return victim

    def _admit_read(self, g, blocks, positions, data):
        # Cache fill is optional; a successful GET must never wait on a writer.
        if not self.mutation.acquire(blocking=False):
            with self.lock:
                self.metrics["admission_busy"] += len(positions)
            return
        try:
            for pos, value in zip(positions, data):
                b = blocks[pos]
                with self.lock:
                    if self.policies.get(b.policy) is not g or g.state != "active":
                        self.metrics["admission_stale_rejections"] += 1
                        continue
                    if b.data is not None or b.moving:
                        continue
                    freq = self.frequency.estimate(self._frequency_key(b))
                    if b.size > self.budget or freq < 2:
                        self.metrics["admission_rejections"] += 1
                        continue
                    need = max(0, self.used + b.size - self.budget)
                    candidates = [
                        v for v in self.resident.values() if not v.pins and not v.moving
                    ]
                    protected = self._protected(candidates)
                    plan, freed, cost = [], 0, 0.0
                    for victim in sorted(
                        candidates, key=lambda v: self._score(v, protected)
                    ):
                        if freed >= need:
                            break
                        plan.append(victim)
                        freed += victim.size
                        cost += self._score(victim, protected) * victim.size
                    gain = max(0, min(8, freq) - 1) * b.size
                    # Plan the entire replacement before side effects. A large
                    # candidate must not evict a few small victims then give up.
                    # Equal value preserves incumbents instead of cycling a
                    # uniform scan through clean entries. Free space remains
                    # admissible at the existing second-demand threshold.
                    if freed < need or (need and cost >= gain):
                        self.metrics["admission_rejections"] += 1
                        continue
                failed = False
                for victim in plan:
                    with self.lock:
                        if self.policies.get(b.policy) is not g or g.state != "active":
                            failed = True
                            break
                        before = self.metrics["spill_bytes"]
                    try:
                        self._spill(victim)
                    except OSError:
                        self.metrics["admission_failures"] += 1
                        failed = True
                        break
                    self.metrics["admission_dirty_spill_bytes"] += (
                        self.metrics["spill_bytes"] - before
                    )
                with self.lock:
                    if self.policies.get(b.policy) is not g or g.state != "active":
                        self.metrics["admission_stale_rejections"] += 1
                        continue
                    if failed or self.used + b.size > self.budget:
                        self.metrics["admission_rejections"] += 1
                        continue
                    self._account()
                    b.data = bytes(value)  # Immutable bytes are retained.
                    self.resident[(b.policy, b.ordinal)] = b
                    self.used += b.size
                    self.metrics["peak_resident_bytes"] = max(
                        self.metrics["peak_resident_bytes"], self.used
                    )
                    self.metrics["admission_count"] += 1
                    self.metrics["admission_bytes"] += b.size
        finally:
            self.mutation.release()

    def snapshot(self):
        with self.lock:
            result = super().snapshot()
            result["predictor"] = {
                "completed_drains": self.drain_samples,
                "drain_forecast_bytes": self.drain_forecast,
                "repeat_heat_ewma": self.heat_forecast,
                "sketch_bytes": sum(map(len, self.frequency.rows)),
                "sketch_decays": self.frequency.decays,
                "credit_source": "past completed drains and observed reads only",
            }
            return result
