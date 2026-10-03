"""Actual TCP metadata benchmark; no KV data IO or trainer is simulated as real."""

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from clients import ReaderSession, Writer
from rpc import RPC, Peer


def timed(fn):
    wall, cpu = time.perf_counter_ns(), time.process_time_ns()
    result = fn()
    return result, {
        "wall_ns": time.perf_counter_ns() - wall,
        "cpu_ns": time.process_time_ns() - cpu,
    }


def run(mode, count, concurrency, writers, reuse, pattern):
    peer = Peer("regions" if mode == "regions" else "objects")
    connections = [RPC(peer.address) for _ in range(9)]
    ctl = connections[0]
    try:
        p = ctl.call("create", identity="weights-A/layout-v1")
        count_per = count // writers
        ws = [Writer(ctl, p, 448 << 10, count_per) for _ in range(writers)]
        readers = [
            ReaderSession(c, p, "weights-A/layout-v1", mode) for c in connections[1:]
        ]

        def publish():
            for w in ws:
                for start in range(0, count_per, 256):
                    n = min(256, count_per - start)
                    w.completed(w.reserve(n), n)
                w.drained()

        _, put = timed(publish)
        retained = ctl.call("stats")
        tasks = [[] for _ in readers]
        seq = 0
        for w in ws:
            keys = list(range(0, count_per, 1 if pattern == "dense" else 64))
            for start in range(0, len(keys), 64):
                tasks[seq % len(tasks)].append((w.region, keys[start : start + 64]))
                seq += 1

        def read(index):
            r = readers[index]
            for rid, ordinals in tasks[index]:
                with r.pin():
                    found = r.locations(rid, ordinals)
                    assert found == [(o * (448 << 10), 448 << 10) for o in ordinals]

        passes = []
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            for _ in range(reuse):
                _, elapsed = timed(lambda: list(pool.map(read, range(len(readers)))))
                passes.append(elapsed)
        cache_bytes = sum(r.bytes() for r in readers)
        _, revoke = timed(lambda: ctl.call("revoke", policy=p))

        def reclaim():
            for r in readers:
                r.close()
            assert ctl.call("collect", policy=p)

        _, gc = timed(reclaim)
        final = ctl.call("stats")
        assert final["objects"] == final["regions"] == final["policies"] == 0
        assert final["transport"]["rpcs"] == sum(c.stats["rpcs"] for c in connections)
        assert final["transport"]["rx_bytes"] == sum(
            c.stats["tx_bytes"] for c in connections
        )
        assert final["transport"]["tx_bytes"] == sum(
            c.stats["rx_bytes"] for c in connections
        )
        return {
            "mode": mode,
            "count": count,
            "concurrency": concurrency,
            "writers": writers,
            "reuse": reuse,
            "pattern": pattern,
            "put": put,
            "passes": passes,
            "revoke": revoke,
            "gc": gc,
            "server": final,
            "retained": retained,
            "client_descriptor_bytes": cache_bytes,
            "requested_objects": sum(len(k) for ts in tasks for _, k in ts) * reuse,
        }
    finally:
        for c in connections:
            c.close()
        peer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    cases = [
        (256, 1, 1, 1, "dense"),
        (8192, 1, 1, 1, "dense"),
        (65536, 1, 1, 1, "dense"),
        (8192, 8, 4, 4, "dense"),
        (8192, 8, 4, 4, "sparse"),
        (8192, 1, 1, 1, "sparse"),
        (8192, 1, 1, 8, "dense"),
        (8192, 8, 4, 1, "dense"),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        for repeat in range(args.repeat):
            modes = ["cached", "regions", "objects"]
            modes = modes[repeat % 3 :] + modes[: repeat % 3]
            for case in cases:
                for mode in modes:
                    row = run(mode, *case)
                    row["repeat"] = repeat
                    out.write(json.dumps(row) + "\n")
                    out.flush()
                print(repeat, case, flush=True)


if __name__ == "__main__":
    main()
