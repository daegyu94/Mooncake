#!/usr/bin/env python3
"""Owned Master, four native TCP clients, real 3FS only; no existing data reset.

A uses upstream object metadata (MEMORY1+DFS1). B is DFS-only policy regions.
DFS layout/IO and logical work are equal; MEMORY replica cost is a confounder.
Trace is mandatory to verify native payload bytes. Instrumented time is diagnostic.
"""

# ruff: noqa: B023
# Phase closures are joined before any enclosing trial variable changes.
import argparse
import ctypes
import hashlib
import importlib.util
import json
import os
import resource
import socket
import subprocess
import time
import urllib.request
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import policy_region_native as native


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n")


def counters(port):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
        lines = r.read().decode().splitlines()
    out = {}
    for line in lines:
        if line.startswith("master_"):
            key, val = line.split()[:2]
            key = key.split("{")[0]
            if key.endswith(("_requests_total", "_items_total")):
                out[key] = out.get(key, 0) + float(val)
    return out


def stats(client):
    s = client.stats()
    return {k: getattr(s, k) for k in dir(s) if not k.startswith("_")}


def proc(pid):
    values = Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()
    return {
        "cpu_s": (int(values[11]) + int(values[12])) / os.sysconf("SC_CLK_TCK"),
        "rss_bytes": int(values[21]) * os.sysconf("SC_PAGE_SIZE"),
    }


def difference(before, after):
    return {k: v - before.get(k, 0) for k, v in after.items() if v != before.get(k, 0)}


def run(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.mount.is_dir() or not os.environ.get("LD_PRELOAD"):
        raise ValueError("existing 3FS mount and USRBIO trace preload required")
    for port in [
        args.port,
        args.metrics,
        *range(args.client_port, args.client_port + 4),
    ]:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", port))
    args.output.mkdir(parents=True)
    os.environ["AGENT_RL_USRBIO_TRACE"] = str(args.output)
    namespace = args.mount / "verl-lab" / ("policy-region-" + uuid.uuid4().hex)
    env = os.environ.copy()
    env.update(
        {
            "MOONCAKE_POLICY_REGIONS": "1",
            "MOONCAKE_ENABLE_DFS": "1",
            "MOONCAKE_DFS_ROOT_DIR": str(namespace),
            "MOONCAKE_DFS_ALLOCATOR": "shard",
            "MOONCAKE_DFS_SHARD_COUNT": "4",
            "MOONCAKE_DFS_SHARD_CAPACITY": str(256 * 1024**2),
            "MOONCAKE_DFS_ALIGNMENT": str(1024**2),
            "MOONCAKE_DFS_SINGLE_TENANT": "true",
            "MOONCAKE_DFS_FS_ADAPTER": "hf3fs",
            "MOONCAKE_DFS_EVICTION_ENABLED": "false",
            "MOONCAKE_DFS_DEFERRED_FREE_SECONDS": "0",
            "MOONCAKE_LOCAL_HOSTNAME": "127.0.0.1",
            "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES": str(64 * 1024**2),
        }
    )
    os.environ.update(env)
    spec = importlib.util.spec_from_file_location(
        "poc_policy_region", args.store_source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = []
    clients = []
    with (args.output / "master.log").open("wb") as log:
        master = subprocess.Popen(
            [
                str(args.master),
                f"--port={args.port}",
                f"--metrics_port={args.metrics}",
                "--enable_offload=true",
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            for attempt in range(100):
                if master.poll() is not None:
                    raise RuntimeError("owned Master exited; inspect master.log")
                try:
                    counters(args.metrics)
                    break
                except OSError:
                    time.sleep(0.2)
            else:
                raise TimeoutError("owned Master readiness")
            for w in range(4):
                clients.append(
                    native.Client(
                        f"127.0.0.1:{args.port}",
                        f"127.0.0.1:{args.client_port + w}",
                        True,
                        64 * 1024**2,
                    )
                )
            dump(
                args.output / "manifest.json",
                {
                    "validation_level": "four native TCP CPU clients + actual Hf3fsAdapter/USRBIO",
                    "gpu_or_trainer": False,
                    "master_pid": master.pid,
                    "client_pid": os.getpid(),
                    "namespace": str(namespace),
                    "incarnation_root": clients[0].info().fsdir,
                    "master_sha256": hashlib.sha256(
                        args.master.read_bytes()
                    ).hexdigest(),
                    "module": native.__file__,
                    "module_sha256": hashlib.sha256(
                        Path(native.__file__).read_bytes()
                    ).hexdigest(),
                    "store_source": str(args.store_source),
                    "store_sha256": hashlib.sha256(
                        args.store_source.read_bytes()
                    ).hexdigest(),
                    "runtime_maps": [
                        line
                        for line in Path("/proc/self/maps").read_text().splitlines()
                        if any(
                            s in line
                            for s in ("policy_region_native", "hf3fs", "usrbio_trace")
                        )
                    ],
                    "environment": {
                        k: v
                        for k, v in env.items()
                        if k.startswith(("MOONCAKE_", "HF3FS_", "LD_", "AGENT_RL_"))
                    },
                    "repeats": args.repeats,
                    "objects_per_worker": args.objects,
                    "batch": args.batch,
                    "A": "object API MEMORY1+DFS1, forced DFS restore",
                    "B": "DFS-only policy regions; same allocated memory capacity unused",
                },
            )
            with ThreadPoolExecutor(max_workers=4) as pool:

                def phase(name, operation):
                    before = counters(args.metrics)
                    region_before = stats(clients[0])
                    m0 = proc(master.pid)
                    c0 = resource.getrusage(resource.RUSAGE_SELF)
                    start = time.monotonic_ns()
                    values = list(pool.map(operation, range(4)))
                    end = time.monotonic_ns()
                    c1 = resource.getrusage(resource.RUSAGE_SELF)
                    m1 = proc(master.pid)
                    row = {
                        "phase": name,
                        "start_ns": start,
                        "end_ns": end,
                        "wall_ms": (end - start) / 1e6,
                        "client_cpu_s": c1.ru_utime
                        + c1.ru_stime
                        - c0.ru_utime
                        - c0.ru_stime,
                        "master_cpu_s": m1["cpu_s"] - m0["cpu_s"],
                        "master_rss_bytes": m1["rss_bytes"],
                        "client_rss_bytes": proc(os.getpid())["rss_bytes"],
                        "normal_master": difference(before, counters(args.metrics)),
                        "regions": difference(region_before, stats(clients[0])),
                    }
                    phases.append(row)
                    return values

                for repeat in range(args.repeats):
                    for size in args.sizes:
                        # Alternate ordering to expose drift without changing logical work.
                        for variant in ["A", "B"] if repeat % 2 == 0 else ["B", "A"]:
                            policy = (
                                f"trial-{repeat}-{size}-{variant}-{uuid.uuid4().hex}"
                            )
                            names = [
                                [f"{policy}:w{w}:k{i}" for i in range(args.objects)]
                                for w in range(4)
                            ]
                            bufs = [
                                [
                                    ctypes.create_string_buffer(size)
                                    for i in range(args.objects)
                                ]
                                for w in range(4)
                            ]
                            pointers = [
                                [[ctypes.addressof(b)] for b in bs] for bs in bufs
                            ]
                            sizes = [
                                [[size] for i in range(args.objects)] for w in range(4)
                            ]
                            expected = []
                            for w in range(4):
                                hashes = []
                                for i, b in enumerate(bufs[w]):
                                    payload = hashlib.sha256(
                                        f"{policy}/{w}/{i}".encode()
                                    ).digest() * (size // 32)
                                    ctypes.memmove(ctypes.addressof(b), payload, size)
                                    hashes.append(hashlib.sha256(payload).hexdigest())
                                expected.append(hashes)
                            stores = (
                                [
                                    module.PolicyRegionStore(
                                        policy,
                                        slots=args.objects,
                                        cache_keys=128,
                                        native=c,
                                    )
                                    for c in clients
                                ]
                                if variant == "B"
                                else clients
                            )
                            phases = []
                            phase(
                                "precheck",
                                lambda w: (
                                    stores[w].batch_is_exist(names[w])
                                    if variant == "B"
                                    else clients[w].object_exists(names[w])
                                ),
                            )

                            def put(w):
                                out = []
                                for begin in range(0, args.objects, args.batch):
                                    end = begin + args.batch
                                    if variant == "B":
                                        out += stores[w].batch_put_from_multi_buffers(
                                            names[w][begin:end],
                                            pointers[w][begin:end],
                                            sizes[w][begin:end],
                                            None,
                                        )
                                    else:
                                        out += clients[w].object_put(
                                            names[w][begin:end],
                                            pointers[w][begin:end],
                                            sizes[w][begin:end],
                                        )
                                assert out == [0] * args.objects, out

                            phase("put", put)

                            def get(w, ids):
                                source = (w + 1) % 4
                                dst = [pointers[w][i] for i in ids]
                                for p in dst:
                                    ctypes.memset(p[0], 0, size)
                                keys = [names[source][i] for i in ids]
                                lengths = [sizes[w][i] for i in ids]
                                got = (
                                    stores[w].batch_get_into_multi_buffers(
                                        keys, dst, lengths
                                    )
                                    if variant == "B"
                                    else clients[w].object_get(keys, dst, lengths)
                                )
                                assert got == [size] * len(ids), got
                                for i in ids:
                                    assert (
                                        hashlib.sha256(bufs[w][i].raw).hexdigest()
                                        == expected[source][i]
                                    )

                            def probe(w):
                                source = (w + 1) % 4
                                got = (
                                    stores[w].batch_is_exist(names[source])
                                    if variant == "B"
                                    else clients[w].object_exists(names[source])
                                )
                                assert got == [1] * args.objects, got

                            phase("positive_probe", probe)
                            phase("dense_get", lambda w: get(w, range(args.objects)))
                            phase("sparse_get", lambda w: get(w, [0, args.objects - 1]))
                            if variant == "B":
                                phase(
                                    "transition",
                                    lambda w: stores[w].transition_policy(
                                        policy + "/next"
                                    ),
                                )
                                deadline = time.monotonic() + 5
                                while (
                                    stats(clients[0])["retained_bytes"]
                                    and time.monotonic() < deadline
                                ):
                                    time.sleep(0.02)
                                assert stats(clients[0])["retained_bytes"] == 0
                                for st in stores:
                                    assert (
                                        st.batch_is_exist(names[0])
                                        == [0] * args.objects
                                    )
                                # Do not close the shared native clients between trials.
                                for c in clients:
                                    c.drain()
                            else:
                                phase(
                                    "transition",
                                    lambda w: (
                                        clients[w].object_reset() if w == 0 else None
                                    ),
                                )
                            results.append(
                                {
                                    "repeat": repeat,
                                    "size": size,
                                    "variant": variant,
                                    "phases": phases,
                                    "logical_write_bytes": 4 * args.objects * size,
                                    "logical_read_bytes": 4 * (args.objects + 2) * size,
                                    "region_state": stats(clients[0]),
                                }
                            )
                            dump(args.output / "results.json", results)
                            print(
                                json.dumps(
                                    {
                                        "repeat": repeat,
                                        "size": size,
                                        "variant": variant,
                                        "correct": True,
                                    }
                                ),
                                flush=True,
                            )
                # Native uncertainty/fencing tests: errors are injected after
                # actual 3FS completion or after a real Master publication ACK.
                test_policy = "fault-" + uuid.uuid4().hex
                c = clients[0]
                c.join(test_policy)
                b = ctypes.create_string_buffer(65536)
                ctypes.memset(ctypes.addressof(b), 91, 65536)
                ptr = [[ctypes.addressof(b)]]
                for fault in ("completion", "publication_ack"):
                    r = c.reserve(test_policy, uuid.uuid4().hex, 65536, 2)
                    c.inject_fault(fault)
                    try:
                        c.put(r, [fault], ptr, [[65536]])
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("fault did not fail closed")
                    v = c.acquire(test_policy, [fault])
                    assert v.objects[0].found == (fault == "publication_ack")
                    if v.read_id:
                        assert c.get(v, [fault], ptr, [[65536]]) == [65536]
                        try:
                            c.get(v, ["unbound"], ptr, [[65536]])
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError("unbound key accepted")
                        c.release(v.read_id)
                        try:
                            c.get(v, [fault], ptr, [[65536]])
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError("released view accepted")
                    try:
                        c.put(r, ["overwrite"], ptr, [[65536]])
                    except RuntimeError:
                        pass
                    else:
                        raise AssertionError("uncertain writer reused")
                    c.close_region(r.id)
                c.leave(test_policy)
                deadline = time.monotonic() + 5
                while stats(c)["retained_bytes"] and time.monotonic() < deadline:
                    time.sleep(0.02)
                assert stats(c)["retained_bytes"] == 0
                dump(
                    args.output / "fault-tests.json",
                    {
                        "completion_uncertainty": "PASS",
                        "publication_ack_loss": "PASS",
                        "unbound_key": "PASS",
                        "released_read": "PASS",
                        "uncertain_writer_reuse": "PASS",
                    },
                )
                # Long-run generation test uses all four actual RPC clients.
                for epoch in range(args.epochs):
                    p = f"long-{epoch}-{uuid.uuid4().hex}"
                    for c in clients:
                        c.join(p)
                    for w, c in enumerate(clients):
                        b = ctypes.create_string_buffer(65536)
                        ctypes.memset(ctypes.addressof(b), w + 1, 65536)
                        r = c.reserve(p, uuid.uuid4().hex, 65536, 1)
                        assert c.put(
                            r, [f"{p}/w{w}"], [[ctypes.addressof(b)]], [[65536]]
                        ) == [0]
                        c.close_region(r.id)
                    for c in clients:
                        c.leave(p)
                    deadline = time.monotonic() + 5
                    while (
                        stats(clients[0])["retained_bytes"]
                        and time.monotonic() < deadline
                    ):
                        time.sleep(0.02)
                    assert stats(clients[0])["retained_bytes"] == 0
                dump(
                    args.output / "long-run.json",
                    {"epochs": args.epochs, "final": stats(clients[0])},
                )
                if args.gpu:
                    import torch

                    device = torch.device("cuda", args.gpu_device)
                    payload = torch.arange(65536, device=device, dtype=torch.int32).to(
                        torch.uint8
                    )
                    destination = torch.zeros_like(payload)
                    torch.cuda.synchronize(device)
                    c = clients[0]
                    p = "gpu-" + uuid.uuid4().hex
                    c.join(p)
                    c.register_buffer(payload.data_ptr(), payload.numel())
                    c.register_buffer(destination.data_ptr(), destination.numel())
                    r = c.reserve(p, uuid.uuid4().hex, 65536, 1)
                    assert c.put(r, [p], [[payload.data_ptr()]], [[65536]]) == [0]
                    view = c.acquire(p, [p])
                    assert c.get(view, [p], [[destination.data_ptr()]], [[65536]]) == [
                        65536
                    ]
                    torch.cuda.synchronize(device)
                    assert torch.equal(payload, destination)
                    c.release(view.read_id)
                    c.close_region(r.id)
                    c.leave(p)
                    dump(
                        args.output / "gpu.json",
                        {
                            "correctness": "PASS",
                            "device": str(device),
                            "name": torch.cuda.get_device_name(device),
                            "bytes": 65536,
                            "torch": torch.__version__,
                            "gpu_d2h_and_h2d": True,
                            "trainer": False,
                            "copy_reduction_claim": False,
                        },
                    )

        finally:
            for c in clients:
                c.close()
            master.terminate()
            try:
                master.wait(timeout=10)
            except subprocess.TimeoutExpired:
                master.kill()
                master.wait()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--master", type=Path, required=True)
    p.add_argument("--mount", type=Path, required=True)
    p.add_argument("--store-source", type=Path, required=True)
    p.add_argument("--port", type=int, default=54151)
    p.add_argument("--metrics", type=int, default=19403)
    p.add_argument("--client-port", type=int, default=54301)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--objects", type=int, default=8)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--epochs", type=int, default=128)
    p.add_argument("--gpu", action="store_true")
    p.add_argument("--gpu-device", type=int, default=1)
    p.add_argument(
        "--sizes", type=int, nargs="+", default=[65536, 458752, 1048576, 4194304]
    )
    run(p.parse_args())
