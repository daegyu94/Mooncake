import pytest
from clients import ReaderSession, Writer
from coordinator import CompletionFrontier, Coordinator, StoreError
from rpc import RPC, Peer


class LocalRPC:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.boot, self.mode = self.call("hello")

    def call(self, op, **args):
        if op != "hello":
            args["boot"] = self.boot
        return self.coordinator.dispatch(op, args)


def setup(mode="regions", capacity=16):
    c = Coordinator(mode)
    rpc = LocalRPC(c)
    policy = rpc.call("create", identity="weights-A/layout-A")
    writer = Writer(rpc, policy, 448 << 10, capacity)
    return c, rpc, policy, writer


def test_frontier_holes_failure_and_bounds():
    f = CompletionFrontier(8)
    a, b = f.reserve(4), f.reserve(4)
    assert f.complete(b, 4) == 0
    assert f.complete(a, 4) == 8
    assert not f.pending
    with pytest.raises(StoreError):
        f.reserve(1)
    with pytest.raises(StoreError):
        f.complete(a, 4)
    f = CompletionFrontier(8)
    a, b = f.reserve(4), f.reserve(4)
    assert f.complete(a, 4, False) == 0
    assert f.failed and f.complete(b, 4) == 0
    assert all(done for _, done in f.pending.values())
    with pytest.raises(StoreError):
        f.reserve(1)


@pytest.mark.parametrize("mode", ["objects", "cached", "regions"])
def test_lookup_cache_and_policy_fence(mode):
    c, rpc, p, w = setup("regions" if mode == "regions" else "objects")
    start = w.reserve(8)
    w.completed(start, 8)
    reader = ReaderSession(rpc, p, "weights-A/layout-A", mode)
    assert reader.locations(w.region, [7, 0, 7]) == [
        (7 * w.stride, w.stride),
        (0, w.stride),
        (7 * w.stride, w.stride),
    ]
    with pytest.raises(StoreError):
        reader.locations(w.region, [8])
    with pytest.raises(StoreError):
        reader.locations(w.region, [0], identity="weights-B/layout-A")
    with pytest.raises(StoreError):
        reader.locations(w.region, [0], boot=rpc.boot + 1)
    rpc.call("revoke", policy=p)
    with pytest.raises(StoreError):
        ReaderSession(rpc, p, "weights-A/layout-A", mode)
    assert reader.locations(w.region, [0]) == [(0, w.stride)]
    assert not rpc.call("collect", policy=p)
    reader.close()
    assert not rpc.call("collect", policy=p)  # writer still owns the region
    w.drained()
    assert rpc.call("collect", policy=p)
    assert not c.policies
    with pytest.raises(StoreError):
        reader.locations(w.region, [0])


def test_inflight_writer_after_revoke_is_quarantined():
    _, rpc, p, w = setup()
    first, second = w.reserve(4), w.reserve(4)
    rpc.call("revoke", policy=p)
    assert not rpc.call("collect", policy=p)
    w.completed(second, 4)
    with pytest.raises(StoreError):
        w.drained()
    with pytest.raises(StoreError):
        w.completed(first, 4)  # publication fenced
    assert not rpc.call("collect", policy=p)
    w.drained()  # both IO batches completed; only now may physical GC run
    assert rpc.call("collect", policy=p)


def test_failure_does_not_advance_visibility_or_release_live_io():
    _, rpc, p, w = setup()
    first, second = w.reserve(4), w.reserve(4)
    w.completed(first, 4, False)
    with pytest.raises(StoreError):
        w.drained()
    w.completed(second, 4)
    reader = ReaderSession(rpc, p, "weights-A/layout-A", "regions")
    with pytest.raises(StoreError):
        reader.locations(w.region, [0])
    reader.close()
    w.drained()
    rpc.call("revoke", policy=p)
    assert rpc.call("collect", policy=p)


def test_incremental_frontier_and_distinct_writers():
    _, rpc, p, w = setup()
    other = Writer(rpc, p, w.stride, 16)
    reader = ReaderSession(rpc, p, "weights-A/layout-A", "regions")
    for expected in [4, 8]:
        start = w.reserve(4)
        w.completed(start, 4)
        assert reader.locations(w.region, [expected - 1]) == [
            ((expected - 1) * w.stride, w.stride)
        ]
    with pytest.raises(StoreError):
        reader.locations(other.region, [0])
    with pytest.raises(StoreError):
        rpc.call(
            "publish", policy=p, region=w.region, writer=other.token, start=8, end=9
        )
    with pytest.raises(StoreError):
        rpc.call("publish", policy=p, region=w.region, writer=w.token, start=9, end=10)
    reader.close()
    w.drained()
    other.drained()
    rpc.call("revoke", policy=p)
    assert rpc.call("collect", policy=p)


def test_layout_restart_identity_and_atomic_object_publish():
    _, rpc, p, w = setup("objects")
    with pytest.raises(StoreError):
        rpc.call(
            "publish",
            policy=p,
            region=w.region,
            writer=w.token,
            start=0,
            end=2,
            entries=[[0, 0, w.stride], [1, 999, w.stride]],
        )
    assert rpc.call("stats")["objects"] == 0
    assert rpc.call("stats")["metrics"]["object_inserts"] == 0
    fresh = Coordinator("regions")
    with pytest.raises(StoreError):
        fresh.dispatch(
            "open", {"boot": rpc.boot, "policy": p, "identity": "weights-A/layout-A"}
        )
    _, rpc, p, w = setup()
    w.completed(w.reserve(1), 1)
    reader = ReaderSession(rpc, p, "weights-A/layout-A", "regions", layout="different")
    with pytest.raises(StoreError):
        reader.locations(w.region, [0])


def test_real_tcp_cache_baseline_and_region_payload():
    for server_mode, read_mode in [("objects", "cached"), ("regions", "regions")]:
        peer = Peer(server_mode)
        rpc = RPC(peer.address)
        try:
            p = rpc.call("create", identity="A")
            w = Writer(rpc, p, 65536, 16)
            w.completed(w.reserve(16), 16)
            w.drained()
            r = ReaderSession(rpc, p, "A", read_mode)
            expected = [(k * 65536, 65536) for k in range(16)]
            assert r.locations(w.region, list(range(16))) == expected
            before = rpc.stats["rpcs"]
            assert r.locations(w.region, list(range(16))) == expected
            assert rpc.stats["rpcs"] == before  # both baselines really cache
            r.close()
            rpc.call("revoke", policy=p)
            assert rpc.call("collect", policy=p)
            stats = rpc.call("stats")
            assert stats["transport"]["rpcs"] == rpc.stats["rpcs"]
            assert stats["transport"]["rx_bytes"] == rpc.stats["tx_bytes"]
            assert stats["transport"]["tx_bytes"] == rpc.stats["rx_bytes"]
        finally:
            rpc.close()
            peer.close()


def test_sealed_writer_and_read_pin():
    _, rpc, p, w = setup()
    w.completed(w.reserve(1), 1)
    w.drained()
    with pytest.raises(StoreError):
        w.reserve(1)
    with pytest.raises(StoreError):
        w.completed(1, 1)
    r = ReaderSession(rpc, p, "weights-A/layout-A", "regions")
    with r.pin():
        rpc.call("revoke", policy=p)
        with pytest.raises(StoreError):
            r.close()
        assert not rpc.call("collect", policy=p)
        assert r.locations(w.region, [0]) == [(0, w.stride)]
    r.close()
    assert rpc.call("collect", policy=p)
    with pytest.raises(StoreError), r.pin():
        pass
