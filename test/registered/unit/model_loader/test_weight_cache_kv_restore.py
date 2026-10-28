"""Weight cache KV restore: daemon session protocol, directory adoption and
commit/evict ordering (CPU)."""

import os
import select
import socket
import subprocess
import threading
from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.mem_cache.allocator.paged import PagedTokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import MatchPrefixParams
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import ComponentType
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.weight_cache import kv_restore
from sglang.srt.weight_cache.protocol import recv_msg, send_msg
from sglang.srt.weight_cache.transport import _recv_fd, _send_fd
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.tree_slots import tree_slots_of

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

PAGE = 4
KEY = {"model_path": "m", "tp_rank": 0}
DESC = ((8, 2), "bfloat16", 32)


@pytest.fixture(autouse=True)
def _no_peer_process(monkeypatch):
    # A socketpair's peer is this test process, which never exits; tests that
    # need a writer process pin a child with _pin_child.
    monkeypatch.setattr(kv_restore, "_pin_peer_process", lambda conn: None)
    monkeypatch.setenv("SGLANG_WEIGHT_CACHE_KV_RESTORE_UNKNOWN_WRITER_GRACE_S", "0")
    # As in an engine with KV restore on: radix caches use the Python TreeCore.
    monkeypatch.setenv("SGLANG_WEIGHT_CACHE_KV_RESTORE", "1")


def _session(store):
    engine, daemon = socket.socketpair()
    send_msg(engine, {"type": "kv_attach", "key": KEY})
    attach = recv_msg(daemon)
    thread = threading.Thread(target=store.serve_session, args=(daemon, attach))
    thread.start()
    return engine, thread, recv_msg(engine)


def _deposit(engine, payload: bytes):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload)
    os.close(write_fd)
    send_msg(engine, {"type": "kv_deposit", "desc": DESC})
    _send_fd(engine, read_fd, 0)
    os.close(read_fd)


def _end(engine, thread):
    engine.close()
    thread.join(timeout=10)


def test_pool_and_directory_outlive_the_engine_session():
    store = kv_restore.KvRestoreStore()
    engine, thread, reply = _session(store)
    assert reply == {"status": "fresh"}
    _deposit(engine, b"kv0")
    send_msg(engine, {"type": "kv_layout_done", "count": 1, "token_capacity": 16})
    pages = [(5, -1, (1, 2, 3, 4), None, None), (6, 5, (5, 6, 7, 8), None, None)]
    send_msg(engine, {"type": "kv_commit", "pages": pages})
    send_msg(engine, {"type": "kv_evict", "page_ids": [6]})

    # While the lease is held, a second engine cannot take the pool.
    other, other_thread, busy = _session(store)
    assert busy == {"status": "busy"}
    _end(other, other_thread)
    _end(engine, thread)  # engine dies: the daemon keeps the pool

    engine, thread, reply = _session(store)
    assert reply["status"] == "restore"
    assert reply["buffers"] == [DESC]
    assert reply["token_capacity"] == 16
    assert reply["directory"] == [pages[0]]
    _, fd = _recv_fd(engine)
    with os.fdopen(fd, "rb") as f:
        assert f.read() == b"kv0"  # the same buffer the first engine deposited
    _end(engine, thread)
    store.close()


def test_engine_dying_before_the_layout_is_sealed_drops_the_pool():
    store = kv_restore.KvRestoreStore()
    engine, thread, _ = _session(store)
    _deposit(engine, b"partial")
    _end(engine, thread)
    engine, thread, reply = _session(store)
    assert reply == {"status": "fresh"}
    _end(engine, thread)
    store.close()


def test_pool_without_buffers_is_never_handed_over():
    # A writer that sealed a layout with no KV buffers holds no KV: the next
    # writer must start fresh, not adopt a directory over unrelated memory.
    store = kv_restore.KvRestoreStore()
    engine, thread, _ = _session(store)
    send_msg(engine, {"type": "kv_layout_done", "count": 0, "token_capacity": 16})
    _end(engine, thread)
    engine, thread, reply = _session(store)
    assert reply == {"status": "fresh"}
    _end(engine, thread)
    store.close()


def _paged(num_pages: int = 16):
    return PagedTokenToKVPoolAllocator(
        size=num_pages * PAGE,
        page_size=PAGE,
        dtype=torch.bfloat16,
        device="cpu",
        kvcache=SimpleNamespace(),
        need_sort=True,
    )


def _cache(allocator):
    return UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=SimpleNamespace(),
            token_to_kv_pool_allocator=allocator,
            page_size=PAGE,
            tree_slots=tree_slots_of((ComponentType.FULL,)),
        )
    )


def _tokens(*pages):
    return [t for p in pages for t in range(p * 10, p * 10 + PAGE)]


def test_directory_tree_is_adopted_with_shared_prefixes():
    # root -> 3 -> 7 -> {9, 11}; page 13's parent is missing (orphan).
    entries = [
        (3, -1, tuple(_tokens(3)), None, None),
        (7, 3, tuple(_tokens(7)), None, None),
        (9, 7, tuple(_tokens(9)), None, None),
        (11, 7, tuple(_tokens(11)), None, None),
        (13, 12, tuple(_tokens(13)), None, None),
    ]
    allocator = _paged()
    cache = _cache(allocator)
    free_before = allocator.available_size()
    adopted = kv_restore.import_directory(cache, entries, PAGE)
    assert adopted == 4 * PAGE
    # The shared prefix is claimed once, not freed as a duplicate.
    assert allocator.available_size() == free_before - 4 * PAGE
    for leaf in (9, 11):
        match = cache.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", _tokens(3, 7, leaf))))
        )
        assert match.device_prefix_len == 3 * PAGE
        path = cache.path_device_indices(match.last_device_node)
        assert (path[::PAGE] // PAGE).tolist() == [3, 7, leaf]


def test_directory_import_adopts_nothing_when_a_page_is_taken():
    # Every directory page is free once the previous writer exited; a taken
    # page means the directory is stale, so nothing is adopted.
    entries = [
        (3, -1, tuple(_tokens(3)), None, None),
        (5, 3, tuple(_tokens(5)), None, None),
        (6, 3, tuple(_tokens(6)), None, None),
    ]
    allocator = _paged()
    cache = _cache(allocator)
    allocator.alloc_pages(torch.tensor([6]))
    free_before = sorted(allocator.get_all_free_pages().tolist())
    assert kv_restore.import_directory(cache, entries, PAGE) == 0
    assert sorted(allocator.get_all_free_pages().tolist()) == free_before
    assert cache.tree_core.root_node.children == {}


def test_directory_import_into_a_smaller_pool_skips_pages_beyond_it():
    # The engine maps the leading 8 pages of a 16-page pool: page 12 and its
    # child are beyond it, so only root -> 3 -> 5 is adopted.
    entries = [
        (3, -1, tuple(_tokens(3)), None, None),
        (5, 3, tuple(_tokens(5)), None, None),
        (12, 3, tuple(_tokens(12)), None, None),
        (13, 12, tuple(_tokens(13)), None, None),
    ]
    allocator = _paged(num_pages=8)
    cache = _cache(allocator)
    free_before = allocator.available_size()
    assert kv_restore.import_directory(cache, entries, PAGE) == 2 * PAGE
    assert allocator.available_size() == free_before - 2 * PAGE
    match = cache.match_prefix(
        MatchPrefixParams(key=RadixKey(array("q", _tokens(3, 5))))
    )
    assert match.device_prefix_len == 2 * PAGE


def test_smaller_pool_fits_the_leading_rows_of_the_restored_buffers():
    # Rows lead, or follow the K/V axis when K and V share one allocation.
    assert kv_restore.fits_restored(((6, 8), "bfloat16", 96), ((9, 8), "bfloat16", 144))
    assert kv_restore.fits_restored(
        ((2, 6, 8), "bfloat16", 192), ((2, 9, 8), "bfloat16", 288)
    )
    assert not kv_restore.fits_restored(
        ((2, 9, 8), "bfloat16", 288), ((2, 6, 8), "bfloat16", 192)
    )
    assert not kv_restore.fits_restored(
        ((2, 6, 4), "bfloat16", 96), ((2, 9, 8), "bfloat16", 288)
    )
    assert not kv_restore.fits_restored(
        ((6, 8), "float16", 96), ((9, 8), "bfloat16", 144)
    )


def test_directory_import_returns_pages_of_duplicate_content():
    # Two chains with the same tokens: the tree keeps one, the other's page
    # goes back to the allocator.
    entries = [
        (3, -1, tuple(_tokens(3)), None, None),
        (5, -1, tuple(_tokens(3)), None, None),
    ]
    allocator = _paged()
    cache = _cache(allocator)
    free_before = allocator.available_size()
    kv_restore.import_directory(cache, entries, PAGE)
    assert allocator.available_size() == free_before - PAGE


class _FakeEvent:
    done = False

    def record(self, stream):
        pass

    def query(self):
        return _FakeEvent.done


def _node(value, tokens, parent):
    node = SimpleNamespace(key=RadixKey(tokens), parent=parent, component_data={})
    node.component_data = [SimpleNamespace(value=None) for _ in ComponentType]
    node.component_data[ComponentType.FULL].value = (
        None if value is None else torch.tensor(value)
    )
    return node


def test_commit_waits_for_the_forward_and_eviction_cancels_it(monkeypatch):
    # A chunked prefill can be inserted while its forward still runs; its pages
    # must not reach the daemon until that forward finished, and an eviction
    # in between must win.
    monkeypatch.setattr(torch.cuda, "Event", _FakeEvent)
    sent = []
    listener = kv_restore.DirectoryListener(
        send=sent.append, page_size=PAGE, stream=None
    )
    listener._stream = object()  # record events, flush by hand
    root = _node([], [], None)
    a = _node(list(range(8, 16)), _tokens(2, 3), root)  # pages 2, 3
    b = _node(list(range(20, 24)), _tokens(5), a)  # page 5
    _FakeEvent.done = False
    listener.on_device_store(a)
    listener.on_device_store(b)
    listener._flush_locked()
    assert sent == []
    listener.on_device_remove(b)
    _FakeEvent.done = True
    listener._flush_locked()
    assert sent[0] == {"type": "kv_evict", "page_ids": [5]}
    assert sent[1]["type"] == "kv_commit"
    assert [p[:2] for p in sent[1]["pages"]] == [(2, -1), (3, 2)]
    assert len(sent) == 2


def _attach(store, **extra):
    engine, daemon = socket.socketpair()
    send_msg(engine, {"type": "kv_attach", "key": KEY, **extra})
    attach = recv_msg(daemon)
    thread = threading.Thread(target=store.serve_session, args=(daemon, attach))
    thread.start()
    return engine, thread


def _writer_with_pool(store):
    writer, writer_thread = _attach(store)
    assert recv_msg(writer) == {"status": "fresh"}
    _deposit(writer, b"kv0")
    send_msg(writer, {"type": "kv_layout_done", "count": 1, "token_capacity": 16})
    return writer, writer_thread


def _pin_child(monkeypatch):
    """The next session's writer process is a child the test controls."""
    child = subprocess.Popen(["sleep", "600"])
    pidfd = os.pidfd_open(child.pid)
    pins = iter([pidfd])
    monkeypatch.setattr(kv_restore, "_pin_peer_process", lambda conn: next(pins, None))
    return child


def _replied(sock, timeout_s):
    return bool(select.select([sock], [], [], timeout_s)[0])


def test_restart_waits_for_an_exiting_writer_instead_of_failing(monkeypatch):
    # A writer restarted right after its predecessor disconnected must get the
    # pool once that process is gone, not "busy".
    store = kv_restore.KvRestoreStore()
    child = _pin_child(monkeypatch)
    writer, writer_thread = _writer_with_pool(store)
    writer.close()
    restarted, restarted_thread = _attach(store)
    assert not _replied(restarted, 0.3)
    child.kill()
    child.wait()
    assert recv_msg(restarted)["status"] == "restore"
    writer_thread.join(timeout=10)
    _end(restarted, restarted_thread)
    store.close()


def test_restart_after_a_writer_that_never_exits_starts_fresh(monkeypatch):
    # Fail closed: the restarted writer must not adopt a pool whose last
    # writer may still be running; it gets a fresh one instead.
    monkeypatch.setattr(kv_restore, "WRITER_EXIT_TIMEOUT_S", 0.3)
    store = kv_restore.KvRestoreStore()
    child = _pin_child(monkeypatch)
    writer, writer_thread = _writer_with_pool(store)
    writer.close()
    restarted, restarted_thread = _attach(store)
    assert recv_msg(restarted) == {"status": "fresh"}
    writer_thread.join(timeout=10)
    _end(restarted, restarted_thread)
    child.kill()
    child.wait()
    store.close()


def test_tp_ranks_agree_on_imported_entries(monkeypatch):
    # Each rank's daemon may have seen a different last commit; every rank
    # must import the same entries.
    a = (3, -1, (1,), None, None)
    b = (4, 3, (2,), None, None)
    ranks = [[a, b], [a]]
    group = SimpleNamespace(world_size=2, cpu_group=None)
    monkeypatch.setattr(kv_restore, "_tp_group", lambda: group)

    def all_gather_object(out, obj, group=None):
        out[:] = [sorted(e) for e in ranks]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    for entries in ranks:
        assert kv_restore._agree_across_tp(entries) == [a]
