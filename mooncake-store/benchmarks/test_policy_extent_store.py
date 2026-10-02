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
