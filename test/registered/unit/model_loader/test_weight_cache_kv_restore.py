"""Weight cache KV restore: daemon session protocol, directory adoption and
commit/evict ordering (CPU)."""

import os
import socket
import threading
from array import array
from types import SimpleNamespace

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

register_cpu_ci(est_time=6, suite="base-a-test-cpu")

PAGE = 4
KEY = {"model_path": "m", "tp_rank": 0}
DESC = ((8, 2), "bfloat16", 32)


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
            tree_components=(ComponentType.FULL,),
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


def _attach(store, role, **extra):
    engine, daemon = socket.socketpair()
    send_msg(engine, {"type": "kv_attach", "key": KEY, "role": role, **extra})
    attach = recv_msg(daemon)
    thread = threading.Thread(target=store.serve_session, args=(daemon, attach))
    thread.start()
    return engine, thread


def test_standby_maps_the_writers_pool_and_promotes_after_writer_eof():
    # A standby attaching first waits for the writer's deposit, then gets the
    # same buffers; it can only promote once the writer's session is gone,
    # and it inherits the directory plus the writer's uncommitted live pages.
    store = kv_restore.KvRestoreStore()
    standby, standby_thread = _attach(store, "standby", wait_s=10)
    writer, writer_thread = _attach(store, "writer")
    assert recv_msg(writer) == {"status": "fresh"}
    _deposit(writer, b"kv0")
    send_msg(writer, {"type": "kv_layout_done", "count": 1, "token_capacity": 16})
    reply = recv_msg(standby)
    assert reply == {"status": "standby", "buffers": [DESC], "token_capacity": 16}
    _, fd = _recv_fd(standby)
    with os.fdopen(fd, "rb") as f:
        assert f.read() == b"kv0"

    page = (2, -1, (1, 2, 3, 4), None, None)
    send_msg(writer, {"type": "kv_live", "add": [2, 3, 9], "remove": []})
    send_msg(writer, {"type": "kv_live", "add": [4], "remove": [3]})
    send_msg(writer, {"type": "kv_commit", "pages": [page]})
    send_msg(standby, {"type": "kv_promote", "timeout_s": 0.2})
    assert recv_msg(standby) == {"status": "timeout"}  # writer still alive

    _end(writer, writer_thread)
    send_msg(standby, {"type": "kv_promote", "timeout_s": 10})
    promoted = recv_msg(standby)
    assert promoted["status"] == "promoted"
    assert promoted["directory"] == [page]
    assert promoted["live"] == [4, 9]  # live minus committed
    assert promoted["writer_gone_s"] >= 0

    # The promoted standby holds the lease now.
    other, other_thread = _attach(store, "writer")
    assert recv_msg(other) == {"status": "busy"}
    _end(other, other_thread)
    _end(standby, standby_thread)
    store.close()


def test_standby_with_a_different_layout_is_refused():
    store = kv_restore.KvRestoreStore()
    writer, writer_thread = _attach(store, "writer")
    recv_msg(writer)
    _deposit(writer, b"kv0")
    send_msg(writer, {"type": "kv_layout_done", "count": 1, "token_capacity": 16})
    engine, daemon = socket.socketpair()
    send_msg(engine, {"type": "kv_attach", "key": {"other": 1}, "role": "standby"})
    thread = threading.Thread(
        target=store.serve_session, args=(daemon, recv_msg(daemon))
    )
    thread.start()
    assert recv_msg(engine) == {"status": "mismatch"}
    _end(engine, thread)
    _end(writer, writer_thread)
    store.close()


def _standby_session(cache):
    session = kv_restore.KvRestoreSession("unused", KEY, page_size=PAGE, role="standby")
    session._mode = "standby"
    session._tree_cache = cache
    allocator = cache.token_to_kv_pool_allocator
    session._held = allocator.alloc(allocator.available_size())
    return session


def test_promotion_quarantines_old_writer_pages_until_the_grace_ends(monkeypatch):
    # The old writer's uncommitted pages may still be written by its GPU work:
    # after promotion they must not be allocatable until the grace ends, and
    # the release is requested once.
    now = [100.0]
    monkeypatch.setattr(kv_restore.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(kv_restore, "_tp_group", lambda: None)
    allocator = _paged()
    cache = _cache(allocator)
    session = _standby_session(cache)
    assert allocator.available_size() == 0  # a standby holds every page

    engine, daemon = socket.socketpair()
    session._conn = engine
    entries = [(3, -1, tuple(_tokens(3)), None, None)]
    send_msg(
        daemon,
        {
            "status": "promoted",
            "directory": entries,
            "live": [5, 9],
            "writer_gone_s": 0.5,
        },
    )
    adopted, quarantined = session.promote(timeout_s=5, grace_s=2.0)
    assert (adopted, quarantined) == (1, 2)
    assert recv_msg(daemon) == {"type": "kv_promote", "timeout_s": 5}
    assert allocator.available_size() == (16 - 3) * PAGE
    assert allocator.alloc_pages(torch.tensor([5])) is None  # quarantined
    assert session.promote(timeout_s=5, grace_s=2.0) == (1, 2)  # idempotent

    now[0] += 1.0
    assert kv_restore._session is None
    assert not session.due_release()
    now[0] += 0.6  # 2.0 grace - 0.5 already gone
    assert session.due_release()
    assert not session.due_release()  # requested once
    assert session.release_quarantine() == 2
    assert allocator.available_size() == (16 - 1) * PAGE
    engine.close()
    daemon.close()


def test_live_pages_report_frees_with_the_next_allocation():
    # Allocations are sent at once (before the KV is written); frees ride the
    # next allocation, which they must precede so a reused page stays live.
    sent = []
    listener = kv_restore.LivePageListener(send=sent.append)
    allocator = _paged()
    allocator.page_listener = listener
    first = allocator.alloc(2 * PAGE)
    allocator.free(first)
    assert len(sent) == 1
    second = allocator.alloc(PAGE)
    assert sent[1]["remove"] == sorted(sent[0]["add"])
    assert sent[1]["add"] == (second[::PAGE] // PAGE).tolist()
    allocator.clear()
    assert sent[-1]["reset"] is True


def test_tp_ranks_agree_on_imported_entries_and_quarantine(monkeypatch):
    # Each rank's daemon may have seen a different last commit; every rank
    # must import the same entries and quarantine the same pages.
    a = (3, -1, (1,), None, None)
    b = (4, 3, (2,), None, None)
    ranks = [([a, b], [7]), ([a], [8])]
    group = SimpleNamespace(world_size=2, cpu_group=None)
    monkeypatch.setattr(kv_restore, "_tp_group", lambda: group)

    def all_gather_object(out, obj, group=None):
        out[:] = [(sorted(e), sorted(l)) for e, l in ranks]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    for entries, live in ranks:
        assert kv_restore._agree_across_tp(entries, live) == ([a], [7, 8])
