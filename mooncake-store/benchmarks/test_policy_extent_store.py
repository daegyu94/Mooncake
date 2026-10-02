"""Local ownership/fence checks; these tests perform no 3FS I/O."""

import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import pytest

from agent_rl_dfs_bench import Buffer
from policy_extent_store import PolicyExtentStore


def test_fence_waits_for_inflight_put_then_removes_only_owned_extents():
    client = MagicMock()
    entered, release, switching = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def put(keys, *_):
        entered.set()
        assert release.wait(5)
        return [0] * len(keys)

    client.batch_put_from_multi_buffers.side_effect = put
    client.batch_remove.side_effect = lambda keys, **_: [0] * len(keys)
    store = PolicyExtentStore(client, None, "owned", 2)
    buffers = [Buffer(128, "cpu"), Buffer(128, "cpu")]

    def advance():
        switching.set()
        return store.advance_policy(1)

    with ThreadPoolExecutor(2) as pool:
        writer = pool.submit(store.put, 0, ["a", "b"], buffers)
        assert entered.wait(5)
        fence = pool.submit(advance)
        assert switching.wait(5)
        assert not fence.done()
        client.batch_remove.assert_not_called()
        release.set()
        writer.result(timeout=5)
        old = fence.result(timeout=5)
    assert len(old) == 1 and old[0].startswith("owned/policy-0/")
    client.batch_remove.assert_called_once_with(old, force=True)
    client.remove_all.assert_not_called()
    with pytest.raises(ValueError, match="policy epoch"):
        store.get(0, ["a"], buffers[:1])
    with pytest.raises(KeyError):
        store.get(1, ["a"], buffers[:1])
    store.put(1, ["a"], buffers[:1])
    assert store.index["a"].extent.startswith("owned/policy-1/")


def test_failed_extent_is_not_published():
    client = MagicMock()
    client.batch_put_from_multi_buffers.return_value = [-5]
    store = PolicyExtentStore(client, None, "owned", 2)
    with pytest.raises(RuntimeError, match="extent put"):
        store.put(0, ["a"], [Buffer(128, "cpu")])
    assert not store.index
    assert len(store.extents) == 1


def test_fence_waits_for_inflight_restore_before_reclaim():
    client = MagicMock()
    client.batch_put_from_multi_buffers.return_value = [0]
    client.register_buffer.return_value = 0
    client.batch_remove.return_value = [0]
    entered, release, switching = (
        threading.Event(),
        threading.Event(),
        threading.Event(),
    )

    def restore(keys, pointers, sizes):
        import ctypes

        entered.set()
        assert release.wait(5)
        ctypes.memset(pointers[0][0], 37, sizes[0][0])
        return [sizes[0][0]]

    client.batch_get_into_multi_buffers.side_effect = restore
    store = PolicyExtentStore(client, None, "owned", 2)
    source = [Buffer(128, "cpu"), Buffer(128, "cpu")]
    destination = Buffer(128, "cpu")
    store.put(0, ["a", "b"], source)

    def advance():
        switching.set()
        return store.advance_policy(1)

    with ThreadPoolExecutor(2) as pool:
        reader = pool.submit(store.get, 0, ["b"], [destination])
        assert entered.wait(5)
        fence = pool.submit(advance)
        assert switching.wait(5)
        assert not fence.done()
        client.batch_remove.assert_not_called()
        release.set()
        reader.result(timeout=5)
        fence.result(timeout=5)
    import ctypes

    assert ctypes.string_at(destination.address, 128) == bytes([37]) * 128
    assert store.epoch == 1 and not store.index
    store.close()
