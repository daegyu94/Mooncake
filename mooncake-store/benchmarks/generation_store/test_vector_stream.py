"""POSIX/C-ABI stub tests; no native3FS performance is measured here."""

import os
import random
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from generation_store import NativeIO
from stream_store import PlannedTailStore, VectorTailStore, plan_adjacent
from vector_io import VectorIO, allocate_output


@pytest.fixture(scope="session")
def stub_library(tmp_path_factory):
    source = Path(__file__).parent
    sdk = Path(os.environ.get("HF3FS_SDK", ""))
    if not (sdk / "hf3fs_usrbio.h").is_file() or not shutil.which("gcc"):
        pytest.skip("HF3FS_SDK header and gcc required for ABI-stub tests")
    library = tmp_path_factory.mktemp("vector-stub") / "vector_stub.so"
    subprocess.run(
        [
            "gcc",
            "-O2",
            "-g",
            "-shared",
            "-fPIC",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I" + str(sdk),
            str(source / "vector_bridge.c"),
            str(source / "vector_stub.c"),
            "-o",
            str(library),
        ],
        check=True,
    )
    return library


@pytest.fixture
def io(stub_library, tmp_path):
    result = VectorIO(stub_library, tmp_path, depth=4, slot_bytes=32)
    result.lib.vs_stub_config(-1, -1, 0, 0)
    yield result
    result.close()


def test_reordered_completions_gather_scatter_duplicates(io, tmp_path):
    fd = io.open(tmp_path / "vector.data")
    try:
        io.write_parts(
            fd, [(0, [(b"abc", 0, 3), (b"defghi", 0, 6)]), (9, [(b"jkl", 0, 3)])]
        )
        outputs = [allocate_output(5), allocate_output(5), allocate_output(2)]
        io.read_into(
            fd,
            [(0, 9), (9, 3)],
            [[(0, 2, 5, 0)], [(0, 2, 5, 0)], [(1, 1, 2, 0)]],
            outputs,
        )
        assert outputs == [b"cdefg", b"cdefg", b"kl"]
        assert io.stats["write_bytes"] == io.stats["read_bytes"] == 12
        assert io.stats["gather_copy_bytes"] == io.stats["scatter_copy_bytes"] == 12
        assert io.lib.vs_stub_drained() == 4
    finally:
        io.close_fd(fd)


@pytest.mark.parametrize(
    "short,prep,submit,expected", [(1, -1, 0, 2), (-1, 1, 0, 1), (-1, -1, 1, 2)]
)
def test_error_drains_prepared_and_does_not_publish_scatter(
    io, tmp_path, short, prep, submit, expected
):
    fd = io.open(tmp_path / "error.data")
    try:
        io.write_parts(fd, [(0, [(b"abcdefgh", 0, 8)])])
        outputs = [io.copy_parts([(b"????", 0, 4)]) for _ in range(2)]
        io.lib.vs_stub_config(short, prep, submit, 0)
        with pytest.raises(OSError):
            io.read_into(
                fd, [(0, 4), (4, 4)], [[(0, 0, 4, 0)], [(1, 0, 4, 0)]], outputs
            )
        assert outputs == [b"????", b"????"]
        assert io.lib.vs_stub_prepared() == io.lib.vs_stub_drained() == expected
        io.lib.vs_stub_config(-1, -1, 0, 0)
        assert io.io(fd, [(0, 8)], read=True) == [b"abcdefgh"]
    finally:
        io.close_fd(fd)


def test_unknown_completion_fails_closed_by_process_abort(stub_library, tmp_path):
    code = """
import resource
from pathlib import Path
from vector_io import VectorIO
resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
io = VectorIO(LIBRARY, MOUNT, depth=2, slot_bytes=32)
fd = io.open(Path(MOUNT) / 'abort.data')
io.lib.vs_stub_config(-1, -1, 0, 1)
io.write_parts(fd, [(0, [(b'abcd', 0, 4)])])
""".replace("LIBRARY", repr(str(stub_library))).replace("MOUNT", repr(str(tmp_path)))
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).parent,
        capture_output=True,
        check=False,
    )
    assert result.returncode == -6


def test_no_gap_adjacency_and_sparse_byte_conservation():
    locations = [(24, 8), (0, 8), (8, 8), (24, 8)]
    ranges, pieces, unique = plan_adjacent(locations, 16)
    assert ranges == [(0, 16), (24, 8)]
    assert sum(size for _, size in ranges) == unique == 24
    assert pieces == [[(1, 0, 8, 0)], [(0, 0, 8, 0)], [(0, 8, 8, 0)], [(1, 0, 8, 0)]]
    with pytest.raises(ValueError):
        plan_adjacent([(0, 16), (8, 16)], 32)


def test_residency_api_exact_and_adjacent_plans(io, tmp_path):
    fd = io.open(tmp_path / "residency.data")
    try:
        io.io(fd, [(0, b"abcdefghijklmnop")])
        ranges = [(0, 4), (4, 4), (8, 4), (12, 4)]
        before = dict(io.stats)
        assert io.io(fd, ranges, read=True) == [b"abcd", b"efgh", b"ijkl", b"mnop"]
        assert io.stats["requests"] - before["requests"] == 4
        before = dict(io.stats)
        io.read_plan = "adjacent"
        assert io.io(fd, ranges, read=True) == [b"abcd", b"efgh", b"ijkl", b"mnop"]
        assert io.stats["requests"] - before["requests"] == 1
        assert io.stats["read_bytes"] - before["read_bytes"] == 16
        assert io.io(fd, [(0, 4), (0, 4)], read=True) == [b"abcd", b"abcd"]
        with pytest.raises(ValueError):
            io.read_into(fd, [(0, 4)], [], [allocate_output(4)])
    finally:
        io.close_fd(fd)


@pytest.mark.parametrize("cls", [PlannedTailStore, VectorTailStore])
def test_random_variable_objects_cross_frontier_exact_order(io, tmp_path, cls):
    channel = NativeIO(io.lib._name, tmp_path, 4, 32) if cls is PlannedTailStore else io
    try:
        store = cls(channel, tmp_path / "store", 10000, chunk_bytes=16, tail_budget=64)
        g = store.create()
        rng = random.Random(713)
        values = [bytes([i % 251]) * rng.randrange(1, 99) for i in range(50)]
        for i, value in enumerate(values):
            store.append(g, [value])
            keys = [rng.randrange(i + 1) for _ in range(7)]
            assert store.get(g, keys) == [values[key] for key in keys]
            assert store.resident_tail_bytes <= 64
        store.seal(g)
        store.flush(g)
        assert store.get(g, list(range(50))) == values
        store.invalidate(g)
        assert store.gc() == 1
        assert (
            not store.generations
            and not store.retired
            and store.resident_tail_bytes == 0
        )
        assert store.metrics["accepted_bytes"] == store.metrics["flushed_bytes"]
    finally:
        if channel is not io:
            channel.close()


def test_mutable_input_ownership_suffix_charge_and_retired_lease(io, tmp_path):
    store = VectorTailStore(
        io, tmp_path / "ownership", 1000, chunk_bytes=16, tail_budget=8
    )
    g = store.create()
    mutable = bytearray(b"a" * 19)
    store.append(g, [mutable])
    mutable[:] = b"z" * 19
    assert store.get(g, [0]) == [b"a" * 19]
    generation = store.acquire(g)
    store.invalidate(g)
    assert store.gc() == 0
    assert store.read_lease(generation, [0]) == [b"a" * 19]
    with pytest.raises(KeyError):
        store.get(g, [0])
    assert store.metrics["ownership_copy_bytes"] == 19
    assert store.metrics["suffix_compact_copy_bytes"] == 3
    assert (
        sum(len(value) for value, _, _ in generation.tail.parts)
        == len(generation.tail)
        == 3
    )
    store.release(generation)
    assert store.resident_tail_bytes == 0 and store.gc() == 1


def test_failed_append_reserves_capacity_and_fences_all_reads(io, tmp_path):
    store = VectorTailStore(io, tmp_path / "failed", 8, chunk_bytes=4, tail_budget=8)
    g = store.create()
    io.lib.vs_stub_config(0, -1, 0, 0)
    with pytest.raises(OSError):
        store.append(g, [b"aaaa", b"bbbb"])
    assert store.generations[g].count == 0 and store.generations[g].cursor >= 4
    with pytest.raises(ValueError):
        store.get(g, [0])
    new = store.create()
    with pytest.raises(BufferError):
        store.append(new, [b"12345"])
    store.invalidate(g)
    assert store.gc() == 1
    io.lib.vs_stub_config(-1, -1, 0, 0)
    store.append(new, [b"12345"])
    assert store.get(new, [0]) == [b"12345"]
    store.invalidate(new)
    store.gc()


def test_read_snapshot_operation_pin_survives_release_and_invalidate(io, tmp_path):
    store = VectorTailStore(io, tmp_path / "race", 1000, chunk_bytes=16, tail_budget=64)
    g = store.create()
    store.append(g, [b"a" * 19])
    lease = store.acquire(g)
    entered, resume = threading.Event(), threading.Event()
    original = io.read_into

    def blocked(*args, **kwargs):
        entered.set()
        assert resume.wait(5)
        return original(*args, **kwargs)

    io.read_into = blocked
    with ThreadPoolExecutor(1) as pool:
        result = pool.submit(store.read_lease, lease, [0])
        assert entered.wait(5)
        store.invalidate(g)
        store.release(lease)
        assert store.gc() == 0
        resume.set()
        assert result.result(5) == [b"a" * 19]
    assert store.resident_tail_bytes == 0 and store.gc() == 1


@pytest.mark.parametrize("cls", [PlannedTailStore, VectorTailStore])
@pytest.mark.parametrize("budget", [4, 64])
def test_hundred_epoch_pressure_and_reader_draining(io, tmp_path, cls, budget):
    channel = NativeIO(io.lib._name, tmp_path, 4, 32) if cls is PlannedTailStore else io
    try:
        store = cls(channel, tmp_path / "long", 128, chunk_bytes=16, tail_budget=budget)
        old = None
        for epoch in range(128):
            if old:
                store.invalidate(old[0])
            gid = store.create()
            payload = bytes([epoch % 251]) * 19
            store.append(gid, [payload])
            lease = store.acquire(gid)
            assert store.get(gid, [0]) == [payload]
            if old:
                assert store.read_lease(old[1], [0]) == [old[2]]
                store.release(old[1])
                assert store.gc() == 1
            assert store.resident_tail_bytes <= budget
            old = gid, lease, payload
        store.invalidate(old[0])
        store.release(old[1])
        assert store.gc() == 1
        assert (
            not store.generations
            and not store.retired
            and store.resident_tail_bytes == 0
        )
        assert (
            store.metrics["accepted_bytes"]
            == store.metrics["flushed_bytes"] + store.metrics["discarded_tail_bytes"]
        )
    finally:
        if channel is not io:
            channel.close()
