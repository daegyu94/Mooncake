"""Measure legacy reset RPCs against an exclusive TCP DRAM Store.

Completed entries contain real bytes. Pending entries are held native PUT
sessions, not an artificial sleep or a simulated Master. Never use a shared
Master: remove_all deletes its entire default-tenant key space.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import urlopen


def port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def rss(pid):
    fields = Path(f"/proc/{pid}/status").read_text().splitlines()
    return {
        line.split(":")[0]: int(line.split()[1])
        for line in fields
        if line.startswith(("VmRSS:", "VmHWM:"))
    }


def ok(results, count):
    assert len(results) == count, (len(results), count)
    assert all(x == 0 for x in results), results[:10]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--entries", type=int, nargs="+", default=[1000, 10000, 100000, 1000000]
    )
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--payload", type=int, default=64)
    parser.add_argument("--pending", type=int, nargs="+", default=[0, 32])
    parser.add_argument("--methods", nargs="+", default=["all", "regex"])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--probe-empty", action="store_true")
    parser.add_argument("--max-master-rss-mib", type=int, default=16384)
    args = parser.parse_args()
    assert args.repeats > 0 and args.batch > 0 and args.payload > 0
    assert set(args.methods) <= {"all", "regex"}
    args.output.mkdir(parents=True, exist_ok=False)
    import mooncake.store as sdk

    master_path = args.master.resolve()
    provenance = {
        "argv": vars(args) | {"master": str(master_path), "output": str(args.output)},
        "sdk": sdk.__file__,
        "master_sha256": hashlib.sha256(master_path.read_bytes()).hexdigest(),
        "sdk_sha256": hashlib.sha256(Path(sdk.__file__).read_bytes()).hexdigest(),
        "kernel": os.uname().release,
        "cpu": Path("/proc/cpuinfo")
        .read_text()
        .split("model name\t: ")[1]
        .splitlines()[0],
        "transport": "tcp loopback, one client, RF1, DRAM only, HA disabled",
        "pending_kind": "native reserved PUT sessions; transfer and commit after reset",
    }
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2, default=str) + "\n"
    )
    payload = bytes(i % 251 for i in range(args.payload))
    buffer = ctypes.create_string_buffer(payload)
    addr = ctypes.addressof(buffer)
    rpc, metrics = port(), port()
    with (args.output / "master.log").open("w") as log, (
        args.output / "samples.jsonl"
    ).open("w") as raw:
        master = subprocess.Popen(
            [str(master_path), f"--port={rpc}", f"--metrics_port={metrics}"],
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        store = None
        try:
            deadline = time.monotonic() + 30
            while True:
                assert master.poll() is None, "Master exited"
                try:
                    urlopen(f"http://127.0.0.1:{metrics}/metrics", timeout=0.5).close()
                    break
                except OSError:
                    assert time.monotonic() < deadline, "Master readiness timeout"
                    time.sleep(0.1)
            store = sdk.MooncakeDistributedStore()
            segment_bytes = max(
                512 * 1024**2, max(args.entries) * max(64, args.payload) * 4
            )
            assert (
                store.setup(
                    f"127.0.0.1:{port()}",
                    "P2PHANDSHAKE",
                    segment_bytes,
                    32 * 1024**2,
                    "tcp",
                    "",
                    f"127.0.0.1:{rpc}",
                )
                == 0
            )
            if args.probe_empty:
                result = {
                    "put_return": store.put("empty-object", b""),
                    "exists": store.is_exist("empty-object"),
                    "removed": store.remove_all(force=True),
                }
                result["put_from_return"] = store.put_from("empty-from", addr, 0)
                result["put_from_exists"] = store.is_exist("empty-from")
                assert (
                    result["exists"]
                    == result["removed"]
                    == result["put_from_exists"]
                    == 0
                ), result
                (args.output / "empty-probe.json").write_text(
                    json.dumps(result, indent=2) + "\n"
                )
                print(json.dumps(result), flush=True)
                return
            assert store.register_buffer(addr, args.payload) == 0

            def fill(keys):
                start = time.perf_counter()
                for j in range(0, len(keys), args.batch):
                    if (
                        j % (args.batch * 64) == 0
                        and rss(master.pid)["VmRSS"] > args.max_master_rss_mib * 1024
                    ):
                        raise MemoryError(
                            "BLOCKED: exclusive Master exceeded experiment RSS budget"
                        )
                    chunk = keys[j : j + args.batch]
                    ok(
                        store.batch_put_from(
                            chunk, [addr] * len(chunk), [args.payload] * len(chunk)
                        ),
                        len(chunk),
                    )
                return time.perf_counter() - start

            def check(keys, value):
                for j in range(0, len(keys), args.batch):
                    chunk = keys[j : j + args.batch]
                    actual = store.batch_is_exist(chunk)
                    assert actual == [value] * len(chunk), (value, actual[:10])

            for n in args.entries:
                for method in args.methods:
                    targets = (
                        [n]
                        if method == "all" or args.pilot
                        else sorted({1000, min(10000, n), min(100000, n), n})
                    )
                    for d in targets:
                        for pending_count in args.pending:
                            assert 0 <= pending_count < d <= n
                            assert store.remove_all(force=True) >= 0
                            target = [
                                f"policy-target@{i:012d}"
                                for i in range(d - pending_count)
                            ]
                            held = [
                                f"policy-target@pending{i:012d}"
                                for i in range(pending_count)
                            ]
                            foreign = [f"policy-foreign@{i:012d}" for i in range(n - d)]
                            foreign_fill = fill(foreign)
                            for repeat in range(-1, args.repeats):
                                fill_s = fill(target)
                                if held:
                                    ok(
                                        store.batch_put_session_start(
                                            held, [args.payload] * len(held)
                                        ),
                                        len(held),
                                    )
                                # Check actual bytes before timing; existence checks are separate RPCs.
                                for key in target[:: max(1, len(target) // 3)][:3]:
                                    assert store.get(key) == payload
                                before = rss(master.pid)
                                wall0, cpu0 = (
                                    time.perf_counter_ns(),
                                    time.process_time_ns(),
                                )
                                deleted = (
                                    store.remove_all(force=True)
                                    if method == "all"
                                    else store.remove_by_regex(
                                        "^policy-target@", force=True
                                    )
                                )
                                cpu_s = (time.process_time_ns() - cpu0) / 1e9
                                elapsed_s = (time.perf_counter_ns() - wall0) / 1e9
                                assert deleted == len(target), (deleted, len(target))
                                # Full enumeration on the first and last measured repeat, samples otherwise.
                                full = repeat in (0, args.repeats - 1)
                                check(
                                    target
                                    if full
                                    else target[:: max(1, len(target) // 8)],
                                    0,
                                )
                                check(
                                    foreign
                                    if full
                                    else foreign[:: max(1, len(foreign) // 8)],
                                    1,
                                )
                                late_s = 0.0
                                if held:
                                    late0 = time.perf_counter()
                                    completed = (
                                        store.batch_put_from_multi_buffer_ranges(
                                            held,
                                            [[addr]] * len(held),
                                            [[args.payload]] * len(held),
                                            [[0]] * len(held),
                                        )
                                    )
                                    assert completed == [args.payload] * len(
                                        held
                                    ), completed
                                    ok(store.batch_put_session_end(held), len(held))
                                    late_s = time.perf_counter() - late0
                                    check(held, 1)
                                    assert store.get(held[0]) == payload
                                    # Successful reset permitted these old-policy writes to survive.
                                    assert store.remove_by_regex(
                                        "^policy-target@pending", force=True
                                    ) == len(held)
                                row = {
                                    "entries": n,
                                    "target": d,
                                    "pending": pending_count,
                                    "method": method,
                                    "repeat": repeat,
                                    "deleted": deleted,
                                    "reset_s": elapsed_s,
                                    "client_cpu_s": cpu_s,
                                    "fill_s": fill_s,
                                    "foreign_fill_s": foreign_fill,
                                    "late_commit_s": late_s,
                                    "full_enumeration": full,
                                    "master_before_kib": before,
                                    "master_after_kib": rss(master.pid),
                                    "client_kib": rss(os.getpid()),
                                }
                                raw.write(json.dumps(row) + "\n")
                                raw.flush()
                                if repeat == 0 or repeat == args.repeats - 1:
                                    print(json.dumps(row), flush=True)
            assert store.remove_all(force=True) >= 0
            (args.output / "final.prom").write_bytes(
                urlopen(f"http://127.0.0.1:{metrics}/metrics", timeout=5).read()
            )
        finally:
            if store is not None:
                store.close()
            master.terminate()
            try:
                master.wait(timeout=15)
            except subprocess.TimeoutExpired:
                master.kill()
                master.wait()


if __name__ == "__main__":
    main()
