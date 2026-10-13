"""Weight cache KV restore: daemon session protocol, pool lifetime, directory
adoption and commit/evict ordering (CPU)."""

import os
import socket
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

register_cpu_ci(est_time=8, suite="base-a-test-cpu")

PAGE = 4
KEY = {"model_path": "m", "tp_rank": 0, "page_size": PAGE}
DESC = ((8, 2), "bfloat16", 32)


# ===== Daemon side =====


def _session(store, key=KEY, role=None):
    engine, daemon = socket.socketpair()
    attach = {"type": "kv_attach", "key": key}
    if role is not None:
        attach["role"] = role
    send_msg(engine, attach)
    thread = threading.Thread(
        target=store.serve_session, args=(daemon, recv_msg(daemon))
    )
    thread.start()
    return engine, thread, recv_msg(engine)


def _deposit(engine, payload: bytes):
    read_fd, write_fd = os.pipe()
    os.write(write_fd, payload)
    os.close(write_fd)
    send_msg(engine, {"type": "kv_deposit", "desc": DESC})
    _send_fd(engine, read_fd, 0)
    os.close(read_fd)


def _seal(engine, payload=b"kv0"):
    _deposit(engine, payload)
    send_msg(engine, {"type": "kv_layout_done", "count": 1, "token_capacity": 16})


def _end(engine, thread):
    engine.close()
    thread.join(timeout=10)
    assert not thread.is_alive()


def test_pool_and_directory_outlive_the_engine_session():
    store = kv_restore.KvRestoreStore()
    engine, thread, reply = _session(store)
    assert reply == {"status": "fresh", "replaced": None}
    _seal(engine)
    pages = [(5, -1, (1, 2, 3, 4), None, None), (6, 5, (5, 6, 7, 8), None, None)]
    send_msg(engine, {"type": "kv_commit", "pages": pages})
    send_msg(engine, {"type": "kv_evict", "page_ids": [6]})

    # Single writer: while the lease is held, a second engine gets busy.
    other, other_thread, busy = _session(store)
    assert busy == {"status": "busy"}
    _end(other, other_thread)
    _end(engine, thread)  # the engine exits or dies: the daemon keeps the pool

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
    assert reply == {"status": "fresh", "replaced": None}
    _end(engine, thread)
    store.close()


def test_a_different_key_replaces_the_pool_and_names_the_fields():
    store = kv_restore.KvRestoreStore()
    engine, thread, _ = _session(store)
    _seal(engine)
    _end(engine, thread)
    other_key = dict(KEY, page_size=2 * PAGE, kv_cache_dtype="fp8_e4m3")
    engine, thread, reply = _session(store, key=other_key)
    assert reply == {"status": "fresh", "replaced": ["kv_cache_dtype", "page_size"]}
    _end(engine, thread)
    store.close()


def test_drop_frees_an_unattached_pool_only():
    store = kv_restore.KvRestoreStore()
    assert store.drop() == {"status": "empty"}
    engine, thread, _ = _session(store)
    _seal(engine)
    send_msg(
        engine, {"type": "kv_commit", "pages": [(5, -1, (1, 2, 3, 4), None, None)]}
    )
    assert store.drop() == {"status": "busy"}  # stop the engine first
    assert store.status()["pool"]["leased"]
    _end(engine, thread)
    # A clean or unclean exit keeps the pool; only a drop frees it.
    assert store.status()["pool"] == {
        "bytes": 32,
        "buffers": 1,
        "pages": 1,
        "token_capacity": 16,
        "complete": True,
        "leased": False,
    }
    assert store.drop() == {"status": "dropped", "bytes": 32, "pages": 1}
    assert store.status() == {"status": "ok", "pool": None}
    engine, thread, reply = _session(store)
    assert reply == {"status": "fresh", "replaced": None}
    _seal(engine)
    # Forced: the daemon lets go while the engine keeps its own mapping; the
    # engine's later messages and exit leave nothing behind.
    assert store.drop(force=True)["status"] == "dropped"
    send_msg(
        engine, {"type": "kv_commit", "pages": [(5, -1, (1, 2, 3, 4), None, None)]}
    )
    _end(engine, thread)
    assert store.status()["pool"] is None
    store.close()


def test_a_mismatched_buffer_resets_the_pool_instead_of_remapping_it():
    store = kv_restore.KvRestoreStore()
    engine, thread, _ = _session(store)
    _deposit(engine, b"kv0")
    _deposit(engine, b"kv1")
    send_msg(engine, {"type": "kv_layout_done", "count": 2, "token_capacity": 16})
    send_msg(
        engine, {"type": "kv_commit", "pages": [(5, -1, (1, 2, 3, 4), None, None)]}
    )
    _end(engine, thread)

    engine, thread, reply = _session(store)
    assert reply["status"] == "restore"
    for _ in reply["buffers"]:
        os.close(_recv_fd(engine)[1])
    # Buffer 1 does not match: keep buffer 0 (mapped already), redo the rest.
    send_msg(engine, {"type": "kv_reset", "keep": 1})
    _deposit(engine, b"new1")
    send_msg(engine, {"type": "kv_layout_done", "count": 2, "token_capacity": 8})
    _end(engine, thread)
    pool = store.status()["pool"]
    assert pool["complete"] and pool["pages"] == 0 and pool["token_capacity"] == 8
    engine, thread, reply = _session(store)
    with os.fdopen(_recv_fd(engine)[1], "rb") as f:
        assert f.read() == b"kv0"
    with os.fdopen(_recv_fd(engine)[1], "rb") as f:
        assert f.read() == b"new1"
    _end(engine, thread)
    store.close()


def test_drop_command_talks_to_the_daemon_socket(tmp_path):
    store = kv_restore.KvRestoreStore()
    path = str(tmp_path / "daemon.sock")
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(1)

    def serve_one():
        conn, _ = server.accept()
        with conn:
            assert recv_msg(conn) == {"type": "kv_drop", "force": False}
            send_msg(conn, store.drop())

    thread = threading.Thread(target=serve_one)
    thread.start()
    assert kv_restore.main(["drop", "--socket", path]) == 0
    thread.join(timeout=10)
    server.close()


def test_unknown_roles_are_refused_not_misread():
    store = kv_restore.KvRestoreStore()
    engine, thread, reply = _session(store, role="standby")
    assert reply["status"] == "unsupported"
    _end(engine, thread)
    engine, thread, reply = _session(store)  # the refusal left no pool or lease
    assert reply == {"status": "fresh", "replaced": None}
    _end(engine, thread)
    store.close()


# ===== Engine side =====


def _serve(store, path):
    """A daemon stand-in that runs KV sessions on ``path``."""
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(path)
    server.listen(4)

    def loop():
        while True:
            try:
                conn, _ = server.accept()
            except OSError:
                return
            threading.Thread(
                target=store.serve_session, args=(conn, recv_msg(conn)), daemon=True
            ).start()

    threading.Thread(target=loop, daemon=True).start()
    return server


def test_engine_session_resets_to_fresh_on_a_layout_mismatch(tmp_path):
    store = kv_restore.KvRestoreStore()
    key = dict(KEY, kv_restore_version=kv_restore.KV_RESTORE_VERSION)
    engine, thread, _ = _session(store, key=key)
    _seal(engine)
    send_msg(
        engine, {"type": "kv_commit", "pages": [(5, -1, (1, 2, 3, 4), None, None)]}
    )
    _end(engine, thread)
    path = str(tmp_path / "daemon.sock")
    server = _serve(store, path)
    session = kv_restore.KvRestoreSession(path, KEY, page_size=PAGE)
    session.attach()
    assert session.mode == "restore" and session.restored_bytes == 32
    session._reset_to_fresh("test")
    assert session.mode == "fresh"
    session._send({"type": "kv_layout_done", "count": 0, "token_capacity": 0})
    session._conn.close()
    for _ in range(100):
        if store.status()["pool"] is None or not store.status()["pool"]["leased"]:
            break
        threading.Event().wait(0.05)
    # The old directory is gone: it described the previous layout.
    assert store.status()["pool"]["pages"] == 0
    server.close()
    store.close()


def test_engine_session_falls_back_to_no_restore_when_busy_or_unreachable(tmp_path):
    path = str(tmp_path / "daemon.sock")
    gone = kv_restore.KvRestoreSession(path, KEY, page_size=PAGE)
    gone.attach()
    assert gone.mode == "off"

    store = kv_restore.KvRestoreStore()
    server = _serve(store, path)
    first = kv_restore.KvRestoreSession(path, KEY, page_size=PAGE)
    first.attach()
    assert first.mode == "fresh"
    second = kv_restore.KvRestoreSession(path, KEY, page_size=PAGE)
    second.attach()
    assert second.mode == "off" and second.restored_bytes == 0
    first._conn.close()
    server.close()
    store.close()


def test_tp_ranks_restore_together_or_not_at_all(monkeypatch, tmp_path):
    # TP ranks keep identical allocators and radix caches, so a rank whose
    # predecessor is still alive (busy) turns restore off on every rank.
    from sglang.srt import runtime_context

    store = kv_restore.KvRestoreStore()
    path = str(tmp_path / "daemon.sock")
    server = _serve(store, path)
    session = kv_restore.KvRestoreSession(path, KEY, page_size=PAGE)
    session.attach()
    assert session.mode == "fresh"
    group = SimpleNamespace(world_size=2, cpu_group=None)
    monkeypatch.setattr(
        runtime_context,
        "get_parallel",
        lambda: SimpleNamespace(tp_size=2, tp_group=group),
    )

    def all_gather_object(out, obj, group=None):
        out[:] = [obj, ("off", None)]

    monkeypatch.setattr(torch.distributed, "all_gather_object", all_gather_object)
    kv_restore._agree_on_mode(session)
    assert session.mode == "off" and session._conn is None
    # The daemon saw the session end before any deposit: no pool is left.
    other = kv_restore.KvRestoreSession(path, KEY, page_size=PAGE)
    for _ in range(100):
        other.attach()
        if other.mode == "fresh":
            break
        threading.Event().wait(0.05)
    assert other.mode == "fresh"
    other._conn.close()
    server.close()
    store.close()


def test_fits_restored_allows_fewer_rows_only():
    assert kv_restore.fits_restored(((4, 2), "bfloat16", 16), DESC)
    assert not kv_restore.fits_restored(((16, 2), "bfloat16", 64), DESC)
    assert not kv_restore.fits_restored(((8, 4), "bfloat16", 64), DESC)
    assert not kv_restore.fits_restored(((8, 2), "float16", 32), DESC)


# ===== Directory adoption =====


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
            tree_core_backend="python",
        )
    )


def _tokens(*pages):
    return [t for p in pages for t in range(p * 10, p * 10 + PAGE)]


def _entry(page, parent, token_page=None):
    return (
        page,
        parent,
        tuple(_tokens(page if token_page is None else token_page)),
        None,
        None,
    )


def test_directory_tree_is_adopted_with_shared_prefixes():
    # root -> 3 -> 7 -> {9, 11}; page 13's parent is missing (orphan).
    entries = [_entry(3, -1), _entry(7, 3), _entry(9, 7), _entry(11, 7), _entry(13, 12)]
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
    # What the daemon is told after adoption is exactly the adopted tree.
    assert sorted(kv_restore.directory_of(cache, PAGE)) == sorted(entries[:4])


def test_directory_import_skips_only_the_leaves_with_a_taken_page():
    # root -> 3 -> {5, 6}; page 6 is already in use, so only 3 -> 5 is adopted.
    entries = [_entry(3, -1), _entry(5, 3), _entry(6, 3)]
    allocator = _paged()
    cache = _cache(allocator)
    allocator.alloc_pages(torch.tensor([6]))
    free_before = allocator.available_size()
    assert kv_restore.import_directory(cache, entries, PAGE) == 2 * PAGE
    assert allocator.available_size() == free_before - 2 * PAGE
    assert 6 not in allocator.get_all_free_pages().tolist()
    assert sorted(kv_restore.directory_of(cache, PAGE)) == entries[:2]


def test_pages_the_import_did_not_keep_leave_the_daemon_directory():
    # Two chains with the same tokens: the tree keeps one, the other's page
    # goes back to the allocator and must not stay in the daemon's directory,
    # or a later restore would adopt a page this engine has rewritten.
    entries = [_entry(3, -1), _entry(5, -1, token_page=3), _entry(9, 5)]
    allocator = _paged()
    cache = _cache(allocator)
    free_before = allocator.available_size()
    kv_restore.import_directory(cache, entries, PAGE)
    assert allocator.available_size() == free_before - 2 * PAGE
    directory = kv_restore.directory_of(cache, PAGE)
    kept = {e[0] for e in directory}
    assert len(kept) == 2 and 9 in kept and not {3, 5} <= kept
    # Page 9 hangs off whichever duplicate the tree kept.
    (nine,) = [e for e in directory if e[0] == 9]
    assert nine[1] in kept


class _FakeEvent:
    done = False

    def record(self, stream):
        pass

    def query(self):
        return _FakeEvent.done


def _node(value, tokens, parent):
    node = SimpleNamespace(
        key=RadixKey(array("q", tokens)), parent=parent, component_data={}
    )
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


# ===== Unsupported configurations fail at startup =====


def test_unsupported_configurations_fail_with_a_clear_error():
    allocator = _paged()
    cache = _cache(allocator)
    allocator.get_kvcache = lambda: SimpleNamespace()  # not an MHA/MLA pool
    with pytest.raises(RuntimeError, match="standard MHA and MLA KV pools"):
        kv_restore._check_supported(cache, set())
    cache.tree_core = SimpleNamespace()  # e.g. the Rust TreeCore
    with pytest.raises(RuntimeError, match="needs the Python TreeCore"):
        kv_restore._check_supported(cache, set())
    cache.disable = True
    with pytest.raises(RuntimeError, match="radix cache"):
        kv_restore._check_supported(cache, set())


def test_kv_restore_selects_the_python_tree_core(monkeypatch):
    from sglang.srt.environ import envs
    from sglang.srt.mem_cache.unified_cache import tree_core_registry

    monkeypatch.setenv("SGLANG_WEIGHT_CACHE_KV_RESTORE", "1")
    assert envs.SGLANG_WEIGHT_CACHE_KV_RESTORE.get()
    params = SimpleNamespace(enable_session_radix_cache=False)
    assert tree_core_registry.resolve_tree_core_backend("rust", params) == "python"
