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

    engine -> {"type": "kv_attach", "key": {...}, "role": r, "wait_s": t}
    writer: daemon -> {"status": "restore", "buffers": [desc...], "directory": [...],
                       "live": [...], "writer_gone_s": s, "token_capacity": n}
                      followed by one fd per buffer, or
                      {"status": "fresh"} / {"status": "busy"}
    standby: daemon -> {"status": "standby", "buffers": [...], "token_capacity": n}
                       + fds (waits up to wait_s for a writer to deposit), or
                       {"status": "mismatch"} / {"status": "timeout"}
    fresh:  engine -> {"type": "kv_deposit", "desc": desc} + fd   (per buffer)
            engine -> {"type": "kv_layout_done", "count": n, "token_capacity": t}
    writer: engine -> {"type": "kv_commit", "pages": [...]}
            engine -> {"type": "kv_evict", "page_ids": [...]}
            engine -> {"type": "kv_clear"}
            engine -> {"type": "kv_live", "add": [...], "remove": [...], "reset": b}
    standby: engine -> {"type": "kv_promote", "timeout_s": t}
             daemon -> {"status": "promoted", "directory": [...], "live": [...],
                        "writer_gone_s": s} (the session is now the writer), or
                       {"status": "timeout"} (still a standby)

A standby maps the pool but never writes to it: its allocator holds every page
until promotion. Promotion waits until no writer session is alive (the
orchestrator fenced the old writer; its EOF may lag). The live set is the
pages a writer allocated but has not freed, reported before the forward that
writes them runs; the next writer quarantines the uncommitted ones until the
old writer is gone for a grace period, in case its GPU work is still running.

A page entry is ``(page_id, parent_page_id, token_ids, extra_key,
cache_salt)``; ``parent_page_id`` is -1 for a page at the root. Commits are
sent when the radix cache inserts the pages, after the forward that wrote
them has finished. Evictions are sent before the pages can be reallocated,
and stream ordering makes the daemon apply them even if the engine dies
right after.
"""

from __future__ import annotations

import logging
import os
import socket
import threading
import time
from array import array
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import msgspec
import torch

from sglang.srt.weight_cache.protocol import recv_msg, send_msg
from sglang.srt.weight_cache.transport import _recv_fd, _send_fd

logger = logging.getLogger(__name__)

KV_RESTORE_VERSION = 1
NO_PARENT_PAGE = -1

# (shape, dtype name, nbytes)
BufferDesc = Tuple[Tuple[int, ...], str, int]
# (page_id, parent_page_id, token_ids, extra_key, cache_salt)
PageEntry = Tuple[int, int, Tuple[int, ...], Optional[str], Optional[str]]


def buffer_desc(shape, dtype: torch.dtype, nbytes: int) -> BufferDesc:
    return (tuple(int(s) for s in shape), str(dtype).replace("torch.", ""), nbytes)


# ===== Daemon side =====


class _KvPool(msgspec.Struct):
    key: Dict[str, Any]
    fds: List[int] = []
    descs: List[BufferDesc] = []
    complete: bool = False
    token_capacity: int = 0
    directory: Dict[int, PageEntry] = {}
    leased: bool = False
    # Pages the current writer allocated and has not freed.
    live: Set[int] = set()
    # The last writer's live pages and when its session ended (monotonic).
    last_writer_live: Set[int] = set()
    last_writer_eof: Optional[float] = None


class KvRestoreStore:
    """The daemon's KV pool: buffer fds, page directory, live pages and the
    writer lease.

    Holds plain fds only; the daemon makes no CUDA calls for KV.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._pool: Optional[_KvPool] = None

    def serve_session(self, conn: socket.socket, attach: Dict[str, Any]) -> None:
        """Run one engine's session until it disconnects. Owns ``conn``."""
        try:
            if attach.get("role", "writer") == "standby":
                pool = self._attach_standby(conn, attach)
                if pool is not None:
                    self._standby_loop(conn, pool)
            else:
                pool = self._attach(conn, attach["key"])
                if pool is not None:
                    self._session_loop(conn, pool)
        except (ConnectionError, OSError) as e:
            logger.info(f"[KvRestore] session ended: {e}")
        finally:
            conn.close()

    def _attach(self, conn: socket.socket, key: Dict[str, Any]) -> Optional[_KvPool]:
        with self._cond:
            pool = self._pool
            if pool is not None and pool.leased:
                send_msg(conn, {"status": "busy"})
                return None
            if pool is not None and pool.complete and pool.key == key:
                pool.leased = True
                send_msg(conn, {"status": "restore", **self._handover(pool)})
                self._send_fds(conn, pool)
                logger.info(
                    f"[KvRestore] restoring {len(pool.fds)} KV buffers and "
                    f"{len(pool.directory)} cached pages"
                )
                return pool
            if pool is not None:
                self._close_pool(pool)
            pool = _KvPool(key=key, leased=True)
            self._pool = pool
            send_msg(conn, {"status": "fresh"})
            return pool

    def _attach_standby(
        self, conn: socket.socket, attach: Dict[str, Any]
    ) -> Optional[_KvPool]:
        """Map-only attach: wait for a complete pool with this key."""
        key = attach["key"]
        deadline = time.monotonic() + attach.get("wait_s", 1800.0)
        with self._cond:
            while True:
                pool = self._pool
                if pool is not None and pool.complete:
                    if pool.key != key:
                        send_msg(conn, {"status": "mismatch"})
                        return None
                    send_msg(
                        conn,
                        {
                            "status": "standby",
                            "buffers": list(pool.descs),
                            "token_capacity": pool.token_capacity,
                        },
                    )
                    self._send_fds(conn, pool)
                    logger.info(f"[KvRestore] standby mapped {len(pool.fds)} buffers")
                    return pool
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    send_msg(conn, {"status": "timeout"})
                    return None
                self._cond.wait(timeout=remaining)

    def _standby_loop(self, conn: socket.socket, pool: _KvPool) -> None:
        conn.settimeout(None)
        while True:
            msg = recv_msg(conn)
            if msg["type"] != "kv_promote":
                raise ValueError(f"standby sent {msg['type']!r}")
            if self._promote(conn, pool, msg["timeout_s"]):
                break
        self._session_loop(conn, pool)

    def _promote(self, conn: socket.socket, pool: _KvPool, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while pool.leased or self._pool is not pool:
                remaining = deadline - time.monotonic()
                if self._pool is not pool or remaining <= 0:
                    send_msg(conn, {"status": "timeout"})
                    return False
                self._cond.wait(timeout=remaining)
            pool.leased = True
            send_msg(conn, {"status": "promoted", **self._handover(pool)})
            logger.info(
                f"[KvRestore] standby promoted with {len(pool.directory)} cached "
                f"pages; {len(pool.last_writer_live)} pages were live"
            )
            return True

    @staticmethod
    def _handover(pool: _KvPool) -> Dict[str, Any]:
        """Directory and quarantine inputs for a new writer of ``pool``."""
        gone = None
        if pool.last_writer_eof is not None:
            gone = time.monotonic() - pool.last_writer_eof
        return {
            "buffers": list(pool.descs),
            "directory": list(pool.directory.values()),
            "live": sorted(pool.last_writer_live - pool.directory.keys()),
            "writer_gone_s": gone,
            "token_capacity": pool.token_capacity,
        }

    @staticmethod
    def _send_fds(conn: socket.socket, pool: _KvPool) -> None:
        for index, fd in enumerate(pool.fds):
            _send_fd(conn, fd, index)

    def _session_loop(self, conn: socket.socket, pool: _KvPool) -> None:
        conn.settimeout(None)
        pool.live = set()
        try:
            while True:
                msg = recv_msg(conn)
                kind = msg["type"]
                with self._cond:
                    if kind == "kv_commit":
                        for entry in msg["pages"]:
                            pool.directory[entry[0]] = tuple(entry)
                    elif kind == "kv_evict":
                        for page_id in msg["page_ids"]:
                            pool.directory.pop(page_id, None)
                    elif kind == "kv_clear":
                        pool.directory.clear()
                    elif kind == "kv_live":
                        if msg.get("reset"):
                            pool.live.clear()
                        pool.live.difference_update(msg["remove"])
                        pool.live.update(msg["add"])
                    elif kind == "kv_deposit":
                        _, fd = _recv_fd(conn)
                        pool.fds.append(fd)
                        pool.descs.append(tuple(msg["desc"]))
                    elif kind == "kv_layout_done":
                        pool.complete = msg["count"] == len(pool.fds)
                        pool.token_capacity = msg["token_capacity"]
                        self._cond.notify_all()
                    else:
                        raise ValueError(f"unknown KV restore message {kind!r}")
        finally:
            with self._cond:
                pool.leased = False
                pool.last_writer_live = pool.live
                pool.live = set()
                pool.last_writer_eof = time.monotonic()
                if not pool.complete:
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


class _SharedBuffer:
    """One exportable device allocation (POSIX fd handle) mapped here."""

    def __init__(self, nbytes: int, device_id: int, fd: Optional[int] = None):
        from sglang.srt.utils.cuda_vmm_utils import (
            VmmReservation,
            _get_cuda_driver,
            align_up,
            check_drv,
            get_allocation_granularity,
            import_and_map_alloc,
            make_device_allocation_prop,
        )

        drv = _get_cuda_driver()
        posix_fd = (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        prop = make_device_allocation_prop(device_id, handle_types=int(posix_fd))
        self.size = align_up(max(nbytes, 1), get_allocation_granularity(prop))
        self.device_id = device_id
        if fd is None:
            self._reservation = VmmReservation(self.size, prop, device_id)
            handle = self._reservation.map(0, self.size, retain_handle=True)
            self.base = self._reservation.base
            self.export_fd = lambda: int(
                check_drv(
                    drv.cuMemExportToShareableHandle(handle, posix_fd, 0),
                    "cuMemExportToShareableHandle(POSIX_FD)",
                )
            )
        else:
            self._reservation = None
            self.base = import_and_map_alloc(
                None, fd, self.size, device_id, use_fabric=False, peer_rank=0
            )

    def tensor(self, shape, dtype: torch.dtype) -> torch.Tensor:
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        nbytes = _nbytes(shape, dtype)
        flat = tensor_from_pointer(self.base, nbytes, device_id=self.device_id)
        return flat.view(dtype).view(shape)


def fits_restored(wanted: BufferDesc, restored: BufferDesc) -> bool:
    """Whether a KV buffer ``wanted`` can be the leading rows of ``restored``.
    Every KV layout keeps tokens or pages in the leading dimension."""
    (shape, dtype, _), (rshape, rdtype, _) = wanted, restored
    return (
        dtype == rdtype
        and len(shape) == len(rshape)
        and shape[1:] == rshape[1:]
        and shape[0] <= rshape[0]
    )


def _nbytes(shape, dtype: torch.dtype) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n * torch.empty((), dtype=dtype).element_size()


class KvRestoreSession:
    """Engine side: one per scheduler process, bound to its GPU's daemon.

    ``role`` is "writer" (restore or deposit, then stream the directory) or
    "standby" (map the pool only, hold every page, wait for ``promote``).
    """

    # How long a standby waits for a writer to deposit the pool.
    STANDBY_ATTACH_WAIT_S = 1800.0

    def __init__(
        self,
        socket_path: str,
        key: Dict[str, Any],
        page_size: int,
        role: str = "writer",
    ):
        if role not in ("writer", "standby"):
            raise ValueError(f"unknown KV restore role {role!r}")
        self._socket_path = socket_path
        self._key = dict(key, kv_restore_version=KV_RESTORE_VERSION)
        self._page_size = page_size
        self._role = role
        self._conn: Optional[socket.socket] = None
        self._send_lock = threading.Lock()
        self._mode: Optional[str] = None  # "restore" | "fresh" | "standby" | "off"
        self._restore_descs: List[BufferDesc] = []
        self._restore_fds: List[int] = []
        self._restore_token_capacity: Optional[int] = None
        self._handover: Dict[str, Any] = {}
        self._buffers: List[_SharedBuffer] = []
        self._tree_cache = None
        self._forward_stream: Optional[torch.cuda.Stream] = None
        # Standby: every page, held so nothing writes into the writer's pool.
        self._held: Optional[torch.Tensor] = None
        # The old writer's uncommitted pages, held until it is surely gone.
        self._quarantine: Optional[torch.Tensor] = None
        self._quarantine_deadline: Optional[float] = None
        self._release_requested = False
        self._promoted_pages: Optional[int] = None
        # The daemon's "promoted" reply once this rank holds the lease; a
        # retried promotion (another rank timed out) must not ask again.
        self._promote_reply: Optional[Dict[str, Any]] = None

    @property
    def mode(self) -> Optional[str]:
        return self._mode

    @property
    def restored_bytes(self) -> int:
        """KV bytes the daemon already holds for this engine (0 unless mapping)."""
        return sum(desc[2] for desc in self._restore_descs)

    @property
    def restored_token_capacity(self) -> Optional[int]:
        """The held pool's token capacity; the engine must size its pool to it."""
        return self._restore_token_capacity

    def _send(self, msg: Dict[str, Any]) -> None:
        # The scheduler thread and the commit flusher share the connection.
        with self._send_lock:
            send_msg(self._conn, msg)

    def attach(self) -> None:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        wait_s = self.STANDBY_ATTACH_WAIT_S
        conn.settimeout(60 + (wait_s if self._role == "standby" else 0))
        try:
            conn.connect(self._socket_path)
            send_msg(
                conn,
                {
                    "type": "kv_attach",
                    "key": self._key,
                    "role": self._role,
                    "wait_s": wait_s,
                },
            )
            reply = recv_msg(conn)
        except OSError as e:
            conn.close()
            if self._role == "standby":
                raise RuntimeError(f"[KvRestore] standby cannot reach daemon: {e}")
            logger.warning(f"[KvRestore] daemon unavailable, KV restore off: {e}")
            self._mode = "off"
            return
        status = reply["status"]
        if status in ("mismatch", "timeout"):
            conn.close()
            raise RuntimeError(
                f"[KvRestore] standby attach failed ({status}): no writer pool "
                f"with this weights/KV layout"
            )
        if status == "busy":
            conn.close()
            logger.warning("[KvRestore] another engine holds the KV pool; restore off")
            self._mode = "off"
            return
        if status in ("restore", "standby"):
            self._restore_token_capacity = reply["token_capacity"]
            self._restore_descs = [tuple(d) for d in reply["buffers"]]
            if status == "restore":
                self._handover = reply
            self._restore_fds = [_recv_fd(conn)[1] for _ in self._restore_descs]
        self._mode = status
        self._conn = conn
        logger.info(
            f"[KvRestore] attached ({status}): {len(self._restore_descs)} buffers, "
            f"{len(reply.get('directory', ()))} cached pages"
        )

    def allocate(self, shape, dtype: torch.dtype, device) -> torch.Tensor:
        """Allocate the next KV buffer: the held one, or a new shared one."""
        if self._mode == "off":
            return torch.zeros(shape, dtype=dtype, device=device)
        device_id = torch.device(device).index
        if device_id is None:
            device_id = torch.cuda.current_device()
        desc = buffer_desc(shape, dtype, _nbytes(shape, dtype))
        index = len(self._buffers)
        if self._mode in ("restore", "standby"):
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
                buf = _SharedBuffer(self._restore_descs[index][2], device_id, fd=fd)
            finally:
                os.close(fd)
            self._buffers.append(buf)
            # A smaller pool uses the leading rows/pages of the held buffer.
            return buf.tensor(shape, dtype)
        buf = _SharedBuffer(desc[2], device_id)
        self._buffers.append(buf)
        tensor = buf.tensor(shape, dtype)
        tensor.zero_()
        fd = buf.export_fd()
        try:
            self._send({"type": "kv_deposit", "desc": desc})
            _send_fd(self._conn, fd, index)
        finally:
            os.close(fd)
        return tensor

    def finish(self, tree_cache, forward_stream: Optional[torch.cuda.Stream]) -> None:
        """Seal the layout; a writer adopts the restored directory and starts
        streaming updates, a standby takes every page and waits."""
        if self._mode in (None, "off"):
            return
        from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

        if not isinstance(tree_cache, UnifiedRadixCache) or tree_cache.disable:
            raise RuntimeError(
                "[KvRestore] needs the radix cache (UnifiedRadixCache) enabled; "
                "unset SGLANG_WEIGHT_CACHE_KV_RESTORE"
            )
        if self._mode in ("restore", "standby") and len(self._buffers) != len(
            self._restore_descs
        ):
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
        if self._mode == "standby":
            self._held = allocator.alloc(allocator.available_size())
            return
        if self._mode == "restore":
            pages = self._adopt(self._handover, grace_s=_RESTORE_GRACE_S)
            logger.info(f"[KvRestore] adopted {pages} cached pages")
            self._handover = {}
        self._start_streaming()

    def _start_streaming(self) -> None:
        tree_cache = self._tree_cache
        tree_cache.tree_core.kv_events.placement_listener = DirectoryListener(
            send=self._send,
            page_size=self._page_size,
            stream=self._forward_stream,
        )
        tree_cache.token_to_kv_pool_allocator.page_listener = LivePageListener(
            send=self._send
        )

    def promote(self, timeout_s: float, grace_s: float) -> Tuple[int, int]:
        """Become the writer: adopt the directory, quarantine the old writer's
        uncommitted pages. Returns (adopted pages, quarantined pages).
        Idempotent; collective over the TP group."""
        if self._mode != "standby":
            if self._promoted_pages is None:
                raise RuntimeError(f"[KvRestore] not a standby (mode {self._mode})")
            return self._promoted_pages, self._quarantined_count()
        if self._promote_reply is None:
            self._conn.settimeout(timeout_s + 60)
            self._send({"type": "kv_promote", "timeout_s": timeout_s})
            reply = recv_msg(self._conn)
            if reply["status"] == "promoted":
                self._promote_reply = reply
        reply = self._promote_reply
        if not _all_ranks(reply is not None):
            # A rank that did get the lease keeps it; the orchestrator must
            # retry promotion, which every rank answers idempotently.
            raise RuntimeError(
                "[KvRestore] promotion timed out waiting for the old writer to go"
            )
        self._conn.settimeout(None)
        allocator = self._tree_cache.token_to_kv_pool_allocator
        allocator.free(self._held)
        self._held = None
        self._mode = "restore"
        self._promoted_pages = self._adopt(reply, grace_s=grace_s)
        self._start_streaming()
        return self._promoted_pages, self._quarantined_count()

    def _adopt(self, handover: Dict[str, Any], *, grace_s: float) -> int:
        """Import the agreed directory and quarantine the old writer's live
        pages; returns the number of adopted pages."""
        entries = [tuple(e) for e in handover["directory"]]
        entries, live = _agree_across_tp(entries, handover["live"])
        adopted_tokens = import_directory(self._tree_cache, entries, self._page_size)
        allocator = self._tree_cache.token_to_kv_pool_allocator
        # Pages the import did not claim; the radix cache owns the claimed ones.
        num_pages = allocator.size // self._page_size
        candidates = [p for p in live if 0 < p <= num_pages]
        quarantine = None
        if candidates:
            free = set(allocator.get_all_free_pages().tolist())
            pages = [p for p in candidates if p in free]
            if pages:
                quarantine = allocator.alloc_pages(
                    torch.tensor(pages, dtype=torch.int64)
                )
        gone = handover["writer_gone_s"]
        # Each rank's daemon saw its writer's EOF at a different time.
        wait = _max_across_tp(max(0.0, grace_s - (gone if gone is not None else 0.0)))
        if quarantine is not None and quarantine.numel():
            self._quarantine = quarantine
            self._quarantine_deadline = time.monotonic() + wait
            self._release_requested = False
            logger.info(
                f"[KvRestore] quarantined {self._quarantined_count()} pages of the "
                f"old writer for {wait:.1f}s"
            )
        return adopted_tokens // self._page_size

    @property
    def held_tokens(self) -> int:
        held = 0 if self._held is None else self._held.numel()
        quarantined = 0 if self._quarantine is None else self._quarantine.numel()
        return held + quarantined

    def _quarantined_count(self) -> int:
        if self._quarantine is None:
            return 0
        return self._quarantine.numel() // self._page_size

    def release_quarantine(self) -> int:
        """Return the quarantined pages to the allocator; returns how many."""
        if self._quarantine is None:
            return 0
        count = self._quarantined_count()
        self._tree_cache.token_to_kv_pool_allocator.free(self._quarantine)
        self._quarantine = None
        self._quarantine_deadline = None
        logger.info(f"[KvRestore] released {count} quarantined pages")
        return count

    def due_release(self) -> bool:
        """Whether the grace has passed and no release was requested yet.
        Checked on the rank that pulls requests, so every rank releases in
        the same scheduler step."""
        if (
            self._quarantine_deadline is None
            or self._release_requested
            or time.monotonic() < self._quarantine_deadline
        ):
            return False
        self._release_requested = True
        return True


# A restarted writer cannot tell how long its predecessor's GPU work may
# still run; arbitrary, matches the promotion default.
_RESTORE_GRACE_S = 2.0


def _tp_group():
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    return parallel.tp_group if parallel.tp_size > 1 else None


def _all_ranks(flag: bool) -> bool:
    group = _tp_group()
    if group is None:
        return flag
    flags = [None] * group.world_size
    torch.distributed.all_gather_object(flags, flag, group=group.cpu_group)
    return all(flags)


def _max_across_tp(value: float) -> float:
    group = _tp_group()
    if group is None:
        return value
    values = [None] * group.world_size
    torch.distributed.all_gather_object(values, value, group=group.cpu_group)
    return max(values)


def _agree_across_tp(
    entries: List[PageEntry], live: List[int]
) -> Tuple[List[PageEntry], List[int]]:
    """TP ranks keep identical radix caches and allocators, but each rank's
    daemon may have seen a different last commit. Import only entries every
    rank has, and quarantine every page any rank saw live; both results are
    identical on every rank."""
    group = _tp_group()
    if group is None:
        return entries, list(live)
    mine = (sorted(entries), sorted(live))
    gathered = [None] * group.world_size
    torch.distributed.all_gather_object(gathered, mine, group=group.cpu_group)
    common = set(gathered[0][0])
    every_live: Set[int] = set()
    for rank_entries, rank_live in gathered:
        common &= set(rank_entries)
        every_live.update(rank_live)
    agreed = [e for e in gathered[0][0] if e in common]
    return agreed, sorted(every_live)


class LivePageListener:
    """Reports the writer's allocated pages before the forward that writes
    them runs, and their release (batched with the next allocation; a late
    release only widens a future quarantine)."""

    def __init__(self, *, send: Callable[[Dict[str, Any]], None]):
        self._send = send
        self._released: List[torch.Tensor] = []

    def on_pages_allocated(self, page_ids: torch.Tensor) -> None:
        removed: List[int] = []
        if self._released:
            removed = torch.cat(self._released).tolist()
            self._released = []
        self._send({"type": "kv_live", "add": page_ids.tolist(), "remove": removed})

    def on_pages_released(self, page_ids: torch.Tensor) -> None:
        self._released.append(page_ids.reshape(-1))

    def on_pages_cleared(self) -> None:
        self._released = []
        self._send({"type": "kv_live", "add": [], "remove": [], "reset": True})


def import_directory(tree_cache, entries: List[PageEntry], page_size: int) -> int:
    """Adopt a page directory into ``tree_cache``; returns the number of
    tokens adopted. Each leaf's root-to-leaf path is imported once, reusing
    the pages earlier leaves already adopted. Pages whose chain to the root
    is incomplete are skipped."""
    from sglang.srt.mem_cache.radix_cache import RadixKey

    by_id = {e[0]: e for e in entries}
    has_child = {e[1] for e in entries}
    adopted_pages: set = set()
    adopted = 0
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
        if not path:
            continue
        path.reverse()
        cached = 0
        while cached < len(path) and path[cached][0] in adopted_pages:
            cached += 1
        if cached == len(path):
            continue
        tokens: List[int] = []
        for e in path:
            tokens.extend(e[2])
        key = RadixKey(array("q", tokens), extra_key=path[0][3], cache_salt=path[0][4])
        page_ids = torch.tensor([e[0] for e in path], dtype=torch.int64)
        if tree_cache.import_prefix(key, page_ids, cached_prefix_pages=cached):
            adopted_pages.update(e[0] for e in path)
            adopted += (len(path) - cached) * page_size
    return adopted


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
    from sglang.srt.runtime_context import get_exec, get_memory, get_schedule

    if not envs.SGLANG_WEIGHT_CACHE_KV_RESTORE.get() or _session is not None:
        # One session per process: a draft model loaded from the same daemon
        # shares the target's pool, whose KV buffers include the draft's.
        return 0
    # HiCache demotes device KV without a device-remove notice before the slots
    # are freed, and a memory saver releases KV under the daemon's feet.
    if (
        get_memory().enable_hierarchical_cache
        or get_exec().features.enable_memory_saver
    ):
        logger.warning(
            "[KvRestore] not supported with hierarchical cache or memory saver; off"
        )
        return 0
    _session = KvRestoreSession(
        socket_path,
        cache_config,
        page_size=get_schedule().page_size,
        role=envs.SGLANG_WEIGHT_CACHE_KV_RESTORE_ROLE.get(),
    )
    _session.attach()
    return _session.restored_bytes


def current_session() -> Optional[KvRestoreSession]:
    return _session


def restored_bytes() -> int:
    """KV bytes the daemon holds that this engine maps instead of allocating."""
    return 0 if _session is None else _session.restored_bytes


def held_tokens() -> int:
    """KV tokens KV restore keeps out of the allocator: a standby's pages and
    the old writer's quarantined pages."""
    return 0 if _session is None else _session.held_tokens


def restored_token_capacity() -> Optional[int]:
    """Token capacity KV sizing must use so the restored buffers fit exactly."""
    return None if _session is None else _session.restored_token_capacity


# ===== Scheduler control requests =====


def handle_promote(req):
    """Promote this rank's standby session; runs on every TP rank."""
    from sglang.srt.managers.io_struct import KvRestorePromoteReqOutput

    if _session is None:
        return KvRestorePromoteReqOutput(
            success=False, message="KV restore is not enabled on this engine"
        )
    try:
        adopted, quarantined = _session.promote(req.timeout_s, req.quarantine_grace_s)
    except RuntimeError as e:
        return KvRestorePromoteReqOutput(success=False, message=str(e))
    return KvRestorePromoteReqOutput(
        success=True,
        message="",
        adopted_pages=adopted,
        quarantined_pages=quarantined,
    )


def handle_release_quarantine(req):
    """Release the quarantine; runs on every TP rank in the same step.
    Internal requests (the grace ended) get no reply."""
    from sglang.srt.managers.io_struct import KvRestoreReleaseQuarantineReqOutput

    released = 0 if _session is None else _session.release_quarantine()
    if req.internal:
        return None
    return KvRestoreReleaseQuarantineReqOutput(released_pages=released)


def due_internal_requests() -> List[Any]:
    """Requests the request-pulling rank adds to the broadcast: a quarantine
    release once its grace has passed."""
    if _session is None or not _session.due_release():
        return []
    from sglang.srt.managers.io_struct import KvRestoreReleaseQuarantineReqInput

    return [KvRestoreReleaseQuarantineReqInput(internal=True)]
