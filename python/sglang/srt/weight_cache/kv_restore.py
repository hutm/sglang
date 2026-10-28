# SPDX-License-Identifier: Apache-2.0
"""KV restore: the weight cache daemon keeps an engine's KV cache alive.

Opt-in with ``SGLANG_WEIGHT_CACHE_KV_RESTORE=1`` on engines that load weights
from a weight cache daemon. The engine allocates its KV buffers as exportable
CUDA VMM memory and hands the daemon their POSIX fds; holding an fd keeps the
memory alive after the engine exits. The engine also streams a page-level
directory of its radix cache. A later engine with the same weights and KV
layout maps the same buffers and adopts the directory into its radix cache,
so it starts with the previous engine's prefix cache instead of a cold one.

Session protocol on one persistent connection (the open connection is the
writer lease; only one engine writes to a pool at a time):

    engine -> {"type": "kv_attach", "key": {...}}
    daemon -> {"status": "restore", "buffers": [desc...], "directory": [...],
               "token_capacity": n} followed by one fd per buffer, or
              {"status": "fresh"} / {"status": "busy"}
    fresh:  engine -> {"type": "kv_deposit", "desc": desc} + fd   (per buffer)
            engine -> {"type": "kv_layout_done", "count": n, "token_capacity": t}
    then:   engine -> {"type": "kv_commit", "pages": [...]}
            engine -> {"type": "kv_evict", "page_ids": [...]}
            engine -> {"type": "kv_clear"}

A writer's lease ends only when its process is gone, not when its connection
closes: the socket closes early in exit(), while the dying process's kernels
may still store into the pool until the driver has torn its context down. The
daemon pins the writer's process with a pidfd at attach and, after the
connection closes, waits for that process to exit before another engine may
write. This needs the daemon to see the
engine's PID namespace; otherwise it waits a fixed
SGLANG_WEIGHT_CACHE_KV_RESTORE_UNKNOWN_WRITER_GRACE_S after the disconnect. So the next writer only ever sees
committed pages that nothing else can still write, and every other page is
free. A writer that does not exit within the bound fails closed: the pool is
dropped, never handed over.
"""

from __future__ import annotations

import logging
import os
import select
import socket
import struct
import threading
import time
from array import array
from typing import Any, Callable, Dict, List, Optional, Tuple

import msgspec
import torch

from sglang.srt.weight_cache.protocol import recv_msg, send_msg
from sglang.srt.weight_cache.transport import ShareableAllocation, _recv_fd, _send_fd

logger = logging.getLogger(__name__)

KV_RESTORE_VERSION = 1
NO_PARENT_PAGE = -1

# (shape, dtype name, nbytes)
BufferDesc = Tuple[Tuple[int, ...], str, int]
# (page_id, parent_page_id, token_ids, extra_key, cache_salt)
PageEntry = Tuple[int, int, Tuple[int, ...], Optional[str], Optional[str]]

# How long the daemon waits for a disconnected writer's process to exit
# before it gives up on the pool; arbitrary, far above a SIGKILLed engine's
# teardown (~0.15 s measured on B200).
WRITER_EXIT_TIMEOUT_S = 30.0
# How long a restarting writer waits for a predecessor that is still connected
# to disconnect (it may be exiting); arbitrary.
LIVE_WRITER_WAIT_S = 1.0


def buffer_desc(shape, dtype: torch.dtype, nbytes: int) -> BufferDesc:
    return (tuple(int(s) for s in shape), str(dtype).replace("torch.", ""), nbytes)


# ===== Daemon side =====


def _pin_peer_process(conn: socket.socket) -> Optional[int]:
    """A pidfd of the connected process, or None when it is not visible here.
    A pidfd keeps naming that process even after its pid is reused."""
    try:
        creds = conn.getsockopt(
            socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")
        )
        pid = struct.unpack("3i", creds)[0]
        return os.pidfd_open(pid) if pid > 0 else None
    except (OSError, AttributeError):
        return None


def _wait_process_exit(pidfd: Optional[int], timeout_s: float) -> bool:
    """Whether the pinned process exited (its context is torn down) in time."""
    if pidfd is None:
        # Another PID namespace: only a fixed wait after the disconnect works.
        from sglang.srt.environ import envs

        time.sleep(envs.SGLANG_WEIGHT_CACHE_KV_RESTORE_UNKNOWN_WRITER_GRACE_S.get())
        return True
    poller = select.poll()
    poller.register(pidfd, select.POLLIN)
    return bool(poller.poll(timeout_s * 1000))


class _Session:
    """One engine connection and, when visible, its process."""

    def __init__(self, conn: socket.socket):
        self.conn = conn
        self.pidfd = _pin_peer_process(conn)
        # Set (under the store's lock) once the connection has closed.
        self.disconnected = False

    def close(self) -> None:
        self.conn.close()
        if self.pidfd is not None:
            os.close(self.pidfd)


class _KvPool(msgspec.Struct):
    key: Dict[str, Any]
    fds: List[int] = []
    descs: List[BufferDesc] = []
    complete: bool = False
    token_capacity: int = 0
    directory: Dict[int, PageEntry] = {}
    # Holds the lease from attach until its process has exited.
    writer: Optional[_Session] = None


class KvRestoreStore:
    """The daemon's KV pool: buffer fds, page directory and the writer lease.

    Holds plain fds only; the daemon makes no CUDA calls for KV.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pool: Optional[_KvPool] = None

    def serve_session(self, conn: socket.socket, attach: Dict[str, Any]) -> None:
        """Run one engine's session until it disconnects. Owns ``conn``."""
        session = _Session(conn)
        try:
            pool = self._attach(session, attach)
            if pool is not None:
                self._session_loop(session, pool)
        except (ConnectionError, OSError) as e:
            logger.info(f"[KvRestore] session ended: {e}")
        finally:
            session.close()

    def _wait_for_exiting_writer(self, pool: _KvPool) -> bool:
        """Wait (holding ``_cond``) for a predecessor that is disconnecting or
        exiting; a writer still connected after LIVE_WRITER_WAIT_S keeps the pool."""
        deadline = time.monotonic() + LIVE_WRITER_WAIT_S
        while pool.writer is not None and self._pool is pool:
            if pool.writer.disconnected:
                # _end_lease releases or drops the pool within WRITER_EXIT_TIMEOUT_S.
                self._cond.wait()
                continue
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            self._cond.wait(timeout=remaining)
        return True

    def _attach(self, session: _Session, attach: Dict[str, Any]) -> Optional[_KvPool]:
        conn, key = session.conn, attach["key"]
        with self._cond:
            pool = self._pool
            if pool is not None and pool.writer is not None:
                if not self._wait_for_exiting_writer(pool):
                    send_msg(conn, {"status": "busy"})
                    return None
                pool = self._pool
            if pool is not None and pool.complete and pool.key == key:
                pool.writer = session
                send_msg(conn, {"status": "restore", **self._handover(pool)})
                self._send_fds(conn, pool)
                logger.info(
                    f"[KvRestore] restoring {len(pool.fds)} KV buffers and "
                    f"{len(pool.directory)} cached pages"
                )
                return pool
            if pool is not None:
                self._close_pool(pool)
            pool = _KvPool(key=key, writer=session)
            self._pool = pool
            send_msg(conn, {"status": "fresh"})
            return pool

    @staticmethod
    def _handover(pool: _KvPool) -> Dict[str, Any]:
        return {
            "buffers": list(pool.descs),
            "token_capacity": pool.token_capacity,
            "directory": list(pool.directory.values()),
        }

    @staticmethod
    def _send_fds(conn: socket.socket, pool: _KvPool) -> None:
        for index, fd in enumerate(pool.fds):
            _send_fd(conn, fd, index)

    def _session_loop(self, session: _Session, pool: _KvPool) -> None:
        conn = session.conn
        conn.settimeout(None)
        try:
            while True:
                msg = recv_msg(conn)
                with self._cond:
                    self._apply(session, pool, msg)
        finally:
            conn.close()
            with self._cond:
                session.disconnected = True
                self._cond.notify_all()
            self._end_lease(session, pool)

    def _apply(self, session: _Session, pool: _KvPool, msg: Dict[str, Any]) -> None:
        kind = msg["type"]
        if kind == "kv_commit":
            for entry in msg["pages"]:
                pool.directory[entry[0]] = tuple(entry)
        elif kind == "kv_evict":
            for page_id in msg["page_ids"]:
                pool.directory.pop(page_id, None)
        elif kind == "kv_clear":
            pool.directory.clear()
        elif kind == "kv_deposit":
            _, fd = _recv_fd(session.conn)
            pool.fds.append(fd)
            pool.descs.append(tuple(msg["desc"]))
        elif kind == "kv_layout_done":
            # A pool with no buffers holds no KV; never hand it over.
            pool.complete = msg["count"] == len(pool.fds) > 0
            pool.token_capacity = msg["token_capacity"]
            self._cond.notify_all()
        else:
            raise ValueError(f"unknown KV restore message {kind!r}")

    def _end_lease(self, session: _Session, pool: _KvPool) -> None:
        """The writer disconnected: keep its lease until its process exited."""
        t0 = time.monotonic()
        exited = _wait_process_exit(session.pidfd, WRITER_EXIT_TIMEOUT_S)
        with self._cond:
            if pool.writer is session:
                pool.writer = None
            if not exited:
                logger.error(
                    f"[KvRestore] writer still running {WRITER_EXIT_TIMEOUT_S:.0f} s "
                    f"after disconnecting; dropping its KV pool instead of handing "
                    f"it over"
                )
            else:
                logger.info(
                    f"[KvRestore] writer gone {(time.monotonic() - t0) * 1e3:.1f} ms "
                    f"after disconnecting; {len(pool.directory)} cached pages"
                )
            if not exited or not pool.complete:
                # Died while depositing: the layout is unusable.
                self._close_pool(pool)
                if self._pool is pool:
                    self._pool = None
            self._cond.notify_all()

    @staticmethod
    def _close_pool(pool: _KvPool) -> None:
        for fd in pool.fds:
            os.close(fd)
        pool.fds = []
        pool.directory = {}

    def close(self) -> None:
        with self._cond:
            if self._pool is not None:
                self._close_pool(self._pool)
                self._pool = None
            self._cond.notify_all()


# ===== Engine side =====


def fits_restored(wanted: BufferDesc, restored: BufferDesc) -> bool:
    """Whether a KV buffer ``wanted`` can be the leading part of ``restored``
    along one dimension: the token or page rows (the first dimension, or the
    second when K and V share one allocation)."""
    (shape, dtype, _), (rshape, rdtype, _) = wanted, restored
    if dtype != rdtype or len(shape) != len(rshape):
        return False
    differ = [i for i, (n, r) in enumerate(zip(shape, rshape)) if n != r]
    return not differ or (len(differ) == 1 and shape[differ[0]] < rshape[differ[0]])


def _nbytes(shape, dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty((), dtype=dtype).element_size()


def _view(buf: ShareableAllocation, shape, dtype: torch.dtype) -> torch.Tensor:
    return buf.tensor(0, _nbytes(shape, dtype)).view(dtype).view(shape)


class KvRestoreSession:
    """Engine side: one per scheduler process, bound to its GPU's daemon."""

    def __init__(self, socket_path: str, key: Dict[str, Any], page_size: int):
        self._socket_path = socket_path
        self._key = dict(key, kv_restore_version=KV_RESTORE_VERSION)
        self._page_size = page_size
        self._conn: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._mode: Optional[str] = None  # "restore" | "fresh" | "off"
        self._restore_descs: List[BufferDesc] = []
        self._restore_fds: List[int] = []
        self._restore_token_capacity: Optional[int] = None
        self._directory: List[PageEntry] = []
        self._buffers: List[ShareableAllocation] = []
        self._tree_cache = None
        self._forward_stream: Optional[torch.cuda.Stream] = None

    @property
    def mode(self) -> Optional[str]:
        return self._mode

    @property
    def restored_bytes(self) -> int:
        """KV bytes the daemon already holds for this engine (0 unless restoring)."""
        return sum(desc[2] for desc in self._restore_descs)

    @property
    def restored_token_capacity(self) -> Optional[int]:
        """The restored pool's token capacity; the engine must size its pool to it."""
        return self._restore_token_capacity

    def _send(self, msg: Dict[str, Any]) -> None:
        # The scheduler thread and the commit flusher share the connection.
        with self._send_lock:
            send_msg(self._conn, msg)

    def attach(self) -> None:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        # The daemon may wait for an exiting predecessor (WRITER_EXIT_TIMEOUT_S).
        conn.settimeout(60 + WRITER_EXIT_TIMEOUT_S)
        try:
            conn.connect(self._socket_path)
            send_msg(conn, {"type": "kv_attach", "key": self._key})
            reply = recv_msg(conn)
        except OSError as e:
            conn.close()
            logger.warning(f"[KvRestore] daemon unavailable, KV restore off: {e}")
            self._mode = "off"
            return
        status = reply["status"]
        if status == "busy":
            conn.close()
            logger.warning("[KvRestore] another engine holds the KV pool; restore off")
            self._mode = "off"
            return
        if status == "restore":
            self._restore_token_capacity = reply["token_capacity"]
            self._restore_descs = [tuple(d) for d in reply["buffers"]]
            self._directory = [tuple(e) for e in reply["directory"]]
            self._restore_fds = [_recv_fd(conn)[1] for _ in self._restore_descs]
        self._mode = status
        self._conn = conn
        logger.info(
            f"[KvRestore] attached ({status}): {len(self._restore_descs)} buffers, "
            f"{len(self._directory)} cached pages"
        )

    def allocate(self, shape, dtype: torch.dtype, device) -> torch.Tensor:
        """Allocate the next KV buffer: the restored one, or a new shared one."""
        if self._mode == "off":
            return torch.zeros(shape, dtype=dtype, device=device)
        device_id = torch.device(device).index
        if device_id is None:
            device_id = torch.cuda.current_device()
        desc = buffer_desc(shape, dtype, _nbytes(shape, dtype))
        index = len(self._buffers)
        if self._mode == "restore":
            if index >= len(self._restore_descs) or not fits_restored(
                desc, self._restore_descs[index]
            ):
                raise RuntimeError(
                    f"[KvRestore] KV layout differs from the daemon's pool at "
                    f"buffer {index}: {desc}; restart the daemon or disable "
                    f"SGLANG_WEIGHT_CACHE_KV_RESTORE"
                )
            fd = self._restore_fds[index]
            try:
                restored_shape, _, restored_nbytes = self._restore_descs[index]
                size = ShareableAllocation.aligned_size(restored_nbytes, device_id)
                buf = ShareableAllocation.import_fd(fd, size, device_id)
            finally:
                os.close(fd)
            self._buffers.append(buf)
            # A smaller pool uses the leading rows/pages of the restored buffer.
            return _view(buf, restored_shape, dtype)[tuple(slice(n) for n in shape)]
        buf = ShareableAllocation.create(desc[2], device_id)
        self._buffers.append(buf)
        tensor = _view(buf, shape, dtype)
        tensor.zero_()
        fd = buf.export_fd()
        try:
            self._send({"type": "kv_deposit", "desc": desc})
            _send_fd(self._conn, fd, index)
        finally:
            os.close(fd)
        return tensor

    def finish(self, tree_cache, forward_stream: Optional[torch.cuda.Stream]) -> None:
        """Seal the layout, adopt the restored directory and start streaming
        directory updates from ``tree_cache``."""
        if self._mode in (None, "off"):
            return
        from sglang.srt.mem_cache.unified_cache.unified_tree_core import (
            UnifiedTreeCore,
        )
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        if not isinstance(tree_cache, UnifiedRadixCache) or tree_cache.disable:
            raise RuntimeError(
                "[KvRestore] needs the radix cache (UnifiedRadixCache) enabled; "
                "unset SGLANG_WEIGHT_CACHE_KV_RESTORE"
            )
        if not isinstance(tree_cache.tree_core, UnifiedTreeCore):
            # Other cores do not report per-node device stores/removes, so the
            # directory would silently stay empty.
            raise RuntimeError(
                f"[KvRestore] needs the Python TreeCore, got "
                f"{type(tree_cache.tree_core).__name__}"
            )
        if self._mode == "restore" and len(self._buffers) != len(self._restore_descs):
            raise RuntimeError(
                f"[KvRestore] the engine allocated {len(self._buffers)} KV buffers "
                f"but the daemon's pool has {len(self._restore_descs)}"
            )
        self._tree_cache = tree_cache
        self._forward_stream = forward_stream
        allocator = tree_cache.token_to_kv_pool_allocator
        if self._mode == "fresh":
            self._send(
                {
                    "type": "kv_layout_done",
                    "count": len(self._buffers),
                    "token_capacity": allocator.size,
                }
            )
        self._conn.settimeout(None)
        if self._mode == "restore":
            self._adopt(self._directory)
            self._directory = []
        self._start_streaming()

    def _start_streaming(self) -> None:
        self._tree_cache.tree_core.kv_events.placement_listener = DirectoryListener(
            send=self._send,
            page_size=self._page_size,
            stream=self._forward_stream,
        )

    def _adopt(self, directory: List[PageEntry]) -> int:
        """Import the directory entries every TP rank has; returns the number
        of adopted pages."""
        t0 = time.perf_counter()
        entries = _agree_across_tp(directory)
        t1 = time.perf_counter()
        pages = import_directory(self._tree_cache, entries, self._page_size)
        pages //= self._page_size
        logger.info(
            f"[KvRestore] adopted {pages} of {len(directory)} cached pages "
            f"(agree {(t1 - t0) * 1e3:.1f} ms, "
            f"import {(time.perf_counter() - t1) * 1e3:.1f} ms)"
        )
        return pages


def _tp_group():
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return parallel.tp_group if parallel.tp_size > 1 else None


def _agree_across_tp(entries: List[PageEntry]) -> List[PageEntry]:
    """TP ranks keep identical radix caches and allocators, but each rank's
    daemon may have seen a different last commit: import only the entries
    every rank has, identically on every rank."""
    group = _tp_group()
    if group is None:
        return entries
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(
        gathered, sorted(entries), group=group.cpu_group
    )
    common = set(gathered[0])
    for rank_entries in gathered[1:]:
        common &= set(rank_entries)
    return [e for e in gathered[0] if e in common]


def import_directory(tree_cache, entries: List[PageEntry], page_size: int) -> int:
    """Adopt a page directory into ``tree_cache``; returns the number of
    tokens adopted. Each leaf's root-to-leaf path is imported once, reusing
    the pages earlier leaves already adopted. Pages whose chain to the root
    is incomplete are skipped, as are pages beyond this engine's (smaller)
    pool. All pages are claimed in one allocator call and copied to the
    device once, so the import does not sync per leaf."""
    from sglang.srt.mem_cache.radix_cache import RadixKey

    allocator = tree_cache.token_to_kv_pool_allocator
    # Page 0 is the padding page; a smaller pool uses the leading pages.
    last_page = allocator.size // page_size
    paths = _leaf_paths([e for e in entries if 0 < e[0] <= last_page])
    if not paths:
        return 0
    pages = list(dict.fromkeys(e[0] for path in paths for e in path))
    if allocator.alloc_pages(torch.tensor(pages, dtype=torch.int64)) is None:
        # Every directory page is free when the previous writer has exited.
        logger.warning(
            "[KvRestore] some cached pages are not free; adopting none of them"
        )
        return 0
    flat = torch.tensor([e[0] for path in paths for e in path], dtype=torch.int64).to(
        allocator.device
    )
    adopted_pages: set = set()
    adopted = offset = 0
    for path in paths:
        cached = 0
        while cached < len(path) and path[cached][0] in adopted_pages:
            cached += 1
        page_ids = flat[offset : offset + len(path)]
        offset += len(path)
        tokens: List[int] = []
        for e in path:
            tokens.extend(e[2])
        key = RadixKey(array("q", tokens), extra_key=path[0][3], cache_salt=path[0][4])
        tree_cache.import_prefix(
            key, page_ids, cached_prefix_pages=cached, claimed=True
        )
        adopted_pages.update(e[0] for e in path)
        adopted += (len(path) - cached) * page_size
    return adopted


def _leaf_paths(entries: List[PageEntry]) -> List[List[PageEntry]]:
    """Root-to-leaf page paths whose chain to the root is complete; a path
    never repeats an earlier path's pages except as its leading prefix."""
    by_id = {e[0]: e for e in entries}
    has_child = {e[1] for e in entries}
    paths = []
    for leaf in by_id:
        if leaf in has_child:
            continue
        path: List[PageEntry] = []
        page_id, seen = leaf, set()
        while page_id != NO_PARENT_PAGE:
            if page_id not in by_id or page_id in seen:
                path = []
                break
            seen.add(page_id)
            path.append(by_id[page_id])
            page_id = by_id[page_id][1]
        if path:
            path.reverse()
            paths.append(path)
    return paths


class DirectoryListener:
    """Turns the radix cache's device store/remove events into directory
    commits and evictions on the session.

    A store can precede the forward that writes its pages (overlap scheduling
    stashes a chunked prefill while that chunk still runs), so commits wait
    for an event recorded on the forward stream at store time. Evictions are
    sent at once and drop matching commits still waiting.
    """

    _FLUSH_INTERVAL_S = 0.005

    def __init__(
        self,
        *,
        send: Callable[[Dict[str, Any]], None],
        page_size: int,
        stream: Optional[torch.cuda.Stream],
    ):
        self._send = send
        self._page_size = page_size
        self._stream = stream
        self._lock = threading.Lock()
        self._pending: List[Tuple[Optional[torch.cuda.Event], List[PageEntry]]] = []
        if stream is not None:
            threading.Thread(
                target=self._flush_loop, name="kv-restore-commit", daemon=True
            ).start()

    def _page_ids(self, value: torch.Tensor) -> List[int]:
        # One sync per event; only paid with KV restore on.
        return (value[:: self._page_size] // self._page_size).tolist()

    @staticmethod
    def _device_value(node) -> Optional[torch.Tensor]:
        from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

        return node.component_data[ComponentType.FULL].value

    def on_device_store(self, node) -> None:
        key = node.key
        value = self._device_value(node)
        if value is None or key.is_bigram:
            return
        page_ids = self._page_ids(value)
        parent = node.parent
        parent_page = NO_PARENT_PAGE
        if parent is not None and parent.parent is not None:
            parent_value = self._device_value(parent)
            if parent_value is None or len(parent_value) == 0:
                return  # parent not on device: chain to the root is broken
            parent_page = int(parent_value[-1]) // self._page_size
        tokens = list(key.token_ids[: len(key)])
        pages = []
        for i, page_id in enumerate(page_ids):
            chunk = tuple(tokens[i * self._page_size : (i + 1) * self._page_size])
            pages.append((page_id, parent_page, chunk, key.extra_key, key.cache_salt))
            parent_page = page_id
        event = None
        if self._stream is not None:
            event = torch.cuda.Event()
            event.record(self._stream)
        with self._lock:
            self._pending.append((event, pages))
            if event is None:
                self._flush_locked()

    def on_device_remove(self, node) -> None:
        value = self._device_value(node)
        if value is None:
            return
        page_ids = self._page_ids(value)
        evicted = set(page_ids)
        with self._lock:
            self._pending = [
                (event, [p for p in pages if p[0] not in evicted])
                for event, pages in self._pending
            ]
            self._send({"type": "kv_evict", "page_ids": page_ids})

    def on_all_cleared(self) -> None:
        with self._lock:
            self._pending = []
            self._send({"type": "kv_clear"})

    def _flush_locked(self) -> None:
        ready: List[PageEntry] = []
        while self._pending:
            event, pages = self._pending[0]
            if event is not None and not event.query():
                break
            ready.extend(pages)
            self._pending.pop(0)
        if ready:
            self._send({"type": "kv_commit", "pages": ready})

    def _flush_loop(self) -> None:
        import time

        while True:
            time.sleep(self._FLUSH_INTERVAL_S)
            try:
                with self._lock:
                    self._flush_locked()
            except OSError as e:
                logger.warning(f"[KvRestore] directory stream closed: {e}")
                return


# ===== Engine wiring =====

_session: Optional[KvRestoreSession] = None


def maybe_start_session(*, socket_path: str, cache_config: Dict[str, Any]) -> int:
    """Called by the weight cache client after it mapped the daemon's weights,
    before KV sizing. Returns the KV bytes the daemon already holds for this
    engine, which KV sizing must count as available."""
    global _session
    from sglang.srt.environ import envs
    from sglang.srt.runtime_context import (
        get_exec,
        get_memory,
        get_parallel,
        get_schedule,
    )

    if not envs.SGLANG_WEIGHT_CACHE_KV_RESTORE.get() or _session is not None:
        # One session per process: a draft model loaded from the same daemon
        # shares the target's pool, whose KV buffers include the draft's.
        return 0
    # HiCache demotes device KV without a device-remove notice before the slots
    # are freed, a memory saver releases KV under the daemon's feet, and KV
    # cache sharding allocates KV-region scratch outside the pool's buffers.
    if (
        get_memory().enable_hierarchical_cache
        or get_exec().features.enable_memory_saver
        or get_parallel().enable_kv_cache_sharding
    ):
        logger.warning(
            "[KvRestore] not supported with hierarchical cache, memory saver or "
            "KV cache sharding; off"
        )
        return 0
    _session = KvRestoreSession(
        socket_path, cache_config, page_size=get_schedule().page_size
    )
    _session.attach()
    return _session.restored_bytes


def current_session() -> Optional[KvRestoreSession]:
    return _session


def restored_bytes() -> int:
    """KV bytes the daemon holds that this engine maps instead of allocating."""
    return 0 if _session is None else _session.restored_bytes


def restored_token_capacity() -> Optional[int]:
    """Token capacity KV sizing must use so the restored buffers fit exactly."""
    return None if _session is None else _session.restored_token_capacity
