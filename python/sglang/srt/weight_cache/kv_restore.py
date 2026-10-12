# SPDX-License-Identifier: Apache-2.0
"""KV restore: the weight cache daemon keeps an engine's KV cache alive.

Opt-in with ``SGLANG_WEIGHT_CACHE_KV_RESTORE=1`` on engines that load weights
from a weight cache daemon (``--weight-cache-mode client``). The engine
allocates its KV buffers as exportable CUDA VMM memory and hands the daemon
their POSIX fds; holding an fd keeps the memory alive after the engine exits.
The engine also streams a page-level directory of its radix cache. The next
engine on the same GPU with the same weights and KV layout maps the same
buffers and adopts the directory into its radix cache, so it starts with its
predecessor's prefix cache instead of an empty one.

Session protocol, on one persistent connection per engine. The open
connection is the writer lease: one engine writes to a pool at a time.

    engine -> {"type": "kv_attach", "key": {...}, "role": "writer"}
    daemon -> {"status": "restore", "buffers": [desc...], "token_capacity": n,
               "directory": [entry...]} followed by one fd per buffer, or
              {"status": "fresh", "replaced": [field...] | None}, or
              {"status": "busy"}  (another engine holds the lease)
    restore, when a KV buffer of this engine does not match the pool:
            engine -> {"type": "kv_reset", "keep": i}  (the pool keeps its first
                      i buffers, which the engine already maps; continue as fresh)
    fresh:  engine -> {"type": "kv_deposit", "desc": desc} + fd   (per buffer)
            engine -> {"type": "kv_layout_done", "count": n, "token_capacity": t}
    then:   engine -> {"type": "kv_commit", "pages": [entry...]}
            engine -> {"type": "kv_evict", "page_ids": [...]}
            engine -> {"type": "kv_clear"}

On their own short connections (``python -m sglang.srt.weight_cache.kv_restore
status|drop``):

    {"type": "kv_status"}           -> {"status": "ok", "pool": {...} | None}
    {"type": "kv_drop", "force": b} -> {"status": "dropped" | "empty" | "busy"}

``kv_drop`` is refused while an engine is attached unless ``force``; a forced
drop only releases the daemon's reference, the attached engine keeps its own.

Messages are dicts and replies carry a "status": a later protocol adds message
types, roles and reply fields without changing these. ``kv_restore_version``
in the key changes only when the meaning of a pool changes.

A directory entry is ``(page_id, parent_page_id, token_ids, extra_key,
cache_salt)``; ``parent_page_id`` is -1 for a page at the root. A page is
committed only after the forward that wrote it has finished, and evicted
before it can be reallocated, so the directory never names a page whose KV
is not what it says.

The pool key is the daemon's weights config, the protocol version, the page
size and the KV dtype; the engine then checks every KV buffer descriptor
(shape, dtype) against the pool. Any mismatch drops the old pool and the
engine starts with a fresh one; a pool is never remapped under another layout.

Lifetime: the pool outlives every engine that used it, including a clean
shutdown. It is freed when the daemon exits, when ``kv_drop`` drops it, or
when an engine with a different layout attaches. A pool whose engine died
before its layout was complete is freed at once.
"""

from __future__ import annotations

import argparse
import logging
import os
import socket
import threading
import time
from array import array
from typing import Any, Callable, Dict, List, Optional, Tuple

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


def key_diff(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    """Names of the pool key fields that differ between ``old`` and ``new``."""
    return sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))


# ===== Daemon side =====


class _KvPool(msgspec.Struct):
    key: Dict[str, Any]
    fds: List[int] = []
    descs: List[BufferDesc] = []
    complete: bool = False
    token_capacity: int = 0
    directory: Dict[int, PageEntry] = {}
    leased: bool = False

    @property
    def nbytes(self) -> int:
        return sum(desc[2] for desc in self.descs)


class KvRestoreStore:
    """The daemon's KV pool: buffer fds, page directory and the writer lease.

    Holds plain fds only; the daemon makes no CUDA calls for KV.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pool: Optional[_KvPool] = None

    def serve_session(self, conn: socket.socket, attach: Dict[str, Any]) -> None:
        """Run one engine's session until it disconnects. Owns ``conn``."""
        try:
            role = attach.get("role", "writer")
            if role != "writer":
                send_msg(conn, {"status": "unsupported", "message": f"role {role!r}"})
                return
            pool = self._attach(conn, attach["key"])
            if pool is not None:
                self._session_loop(conn, pool)
        except (ConnectionError, OSError) as e:
            logger.info(f"[KvRestore] session ended: {e}")
        finally:
            conn.close()

    def _attach(self, conn: socket.socket, key: Dict[str, Any]) -> Optional[_KvPool]:
        with self._lock:
            pool = self._pool
            if pool is not None and pool.leased:
                send_msg(conn, {"status": "busy"})
                return None
            if pool is not None and pool.complete and pool.key == key:
                pool.leased = True
                send_msg(
                    conn,
                    {
                        "status": "restore",
                        "buffers": list(pool.descs),
                        "token_capacity": pool.token_capacity,
                        "directory": list(pool.directory.values()),
                    },
                )
                for index, fd in enumerate(pool.fds):
                    _send_fd(conn, fd, index)
                logger.info(
                    f"[KvRestore] restoring {len(pool.fds)} KV buffers "
                    f"({pool.nbytes / (1 << 30):.2f} GiB) and "
                    f"{len(pool.directory)} cached pages"
                )
                return pool
            replaced = None
            if pool is not None:
                replaced = key_diff(pool.key, key)
                logger.warning(
                    f"[KvRestore] dropping the KV pool of a different engine "
                    f"configuration (differs in: {', '.join(replaced)}); "
                    f"{len(pool.directory)} cached pages are lost"
                )
                self._close_pool(pool)
            pool = _KvPool(key=key, leased=True)
            self._pool = pool
            send_msg(conn, {"status": "fresh", "replaced": replaced})
            return pool

    def _session_loop(self, conn: socket.socket, pool: _KvPool) -> None:
        conn.settimeout(None)
        try:
            while True:
                msg = recv_msg(conn)
                kind = msg["type"]
                fd = _recv_fd(conn)[1] if kind == "kv_deposit" else None
                with self._lock:
                    if kind == "kv_commit":
                        for entry in msg["pages"]:
                            pool.directory[entry[0]] = tuple(entry)
                    elif kind == "kv_evict":
                        for page_id in msg["page_ids"]:
                            pool.directory.pop(page_id, None)
                    elif kind == "kv_clear":
                        pool.directory.clear()
                    elif kind == "kv_deposit":
                        pool.fds.append(fd)
                        pool.descs.append(tuple(msg["desc"]))
                    elif kind == "kv_reset":
                        keep = msg["keep"]
                        for fd in pool.fds[keep:]:
                            os.close(fd)
                        pool.fds, pool.descs = pool.fds[:keep], pool.descs[:keep]
                        pool.directory = {}
                        pool.complete = False
                        logger.warning(
                            "[KvRestore] the engine's KV layout differs from the "
                            "pool; dropped its cache and continuing as a fresh pool"
                        )
                    elif kind == "kv_layout_done":
                        pool.complete = msg["count"] == len(pool.fds)
                        pool.token_capacity = msg["token_capacity"]
                    else:
                        raise ValueError(f"unknown KV restore message {kind!r}")
        finally:
            with self._lock:
                pool.leased = False
                if self._pool is not pool:
                    self._close_pool(pool)  # dropped with force while attached
                elif not pool.complete:
                    # Died while depositing: the layout is unusable.
                    logger.info(
                        "[KvRestore] engine left before its KV pool was complete"
                    )
                    self._close_pool(pool)
                    self._pool = None

    def status(self) -> Dict[str, Any]:
        with self._lock:
            pool = self._pool
            if pool is None:
                return {"status": "ok", "pool": None}
            return {
                "status": "ok",
                "pool": {
                    "bytes": pool.nbytes,
                    "buffers": len(pool.fds),
                    "pages": len(pool.directory),
                    "token_capacity": pool.token_capacity,
                    "complete": pool.complete,
                    "leased": pool.leased,
                },
            }

    def drop(self, force: bool = False) -> Dict[str, Any]:
        """Free the pool. Refused while an engine is attached unless ``force``;
        the attached engine then keeps its own mapping until it exits."""
        with self._lock:
            pool = self._pool
            if pool is None:
                return {"status": "empty"}
            if pool.leased and not force:
                return {"status": "busy"}
            reply = {
                "status": "dropped",
                "bytes": pool.nbytes,
                "pages": len(pool.directory),
            }
            self._close_pool(pool)
            self._pool = None
        logger.info(f"[KvRestore] dropped the KV pool on request: {reply}")
        return reply

    @staticmethod
    def _close_pool(pool: _KvPool) -> None:
        for fd in pool.fds:
            os.close(fd)
        pool.fds = []
        pool.directory = {}

    def close(self) -> None:
        with self._lock:
            if self._pool is not None:
                self._close_pool(self._pool)
                self._pool = None


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
    Every supported KV layout keeps tokens or pages in the leading dimension."""
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


def _restorable_buffers(kvcache) -> Optional[List[torch.Tensor]]:
    """The KV buffers of a pool type KV restore supports, else None: the
    standard MHA and MLA pools, whose buffers hold all of a page's KV state."""
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, MLATokenToKVPool

    if type(kvcache) is MHATokenToKVPool:
        if kvcache.is_quantized_kv_cache or kvcache.post_capture_active:
            return None
        return list(kvcache.k_buffer) + list(kvcache.v_buffer)
    if type(kvcache) is MLATokenToKVPool:
        return list(kvcache.kv_buffer)
    return None


class KvRestoreSession:
    """Engine side: one per scheduler process, bound to its GPU's daemon."""

    def __init__(self, socket_path: str, key: Dict[str, Any], page_size: int):
        self._socket_path = socket_path
        self._key = dict(key, kv_restore_version=KV_RESTORE_VERSION)
        self._page_size = page_size
        self._conn: Optional[socket.socket] = None
        # The scheduler thread and the commit flusher share the connection.
        self._send_lock = threading.Lock()
        self._mode: Optional[str] = None  # "restore" | "fresh" | "off"
        self._restore_descs: List[BufferDesc] = []
        self._restore_fds: List[int] = []
        self._restore_token_capacity: Optional[int] = None
        self._directory: List[PageEntry] = []
        self._buffers: List[_SharedBuffer] = []

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
        with self._send_lock:
            send_msg(self._conn, msg)

    def attach(self) -> None:
        conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        conn.settimeout(60)
        try:
            conn.connect(self._socket_path)
            send_msg(conn, {"type": "kv_attach", "key": self._key, "role": "writer"})
            reply = recv_msg(conn)
        except OSError as e:
            conn.close()
            logger.warning(f"[KvRestore] daemon unavailable, KV restore off: {e}")
            self._mode = "off"
            return
        status = reply["status"]
        if status not in ("restore", "fresh"):
            conn.close()
            if status == "busy":
                logger.warning(
                    "[KvRestore] another engine is attached to this GPU's KV pool; "
                    "KV restore off for this engine"
                )
            else:
                logger.warning(f"[KvRestore] daemon refused the session: {reply}")
            self._mode = "off"
            return
        if status == "restore":
            self._restore_token_capacity = reply["token_capacity"]
            self._restore_descs = [tuple(d) for d in reply["buffers"]]
            self._directory = [tuple(e) for e in reply["directory"]]
            self._restore_fds = [_recv_fd(conn)[1] for _ in self._restore_descs]
        elif reply.get("replaced"):
            logger.warning(
                f"[KvRestore] the daemon held a KV pool for a different engine "
                f"configuration (differs in: {', '.join(reply['replaced'])}); it "
                f"was dropped and this engine starts with an empty KV cache"
            )
        self._mode = status
        self._conn = conn
        logger.info(
            f"[KvRestore] attached ({status}): {len(self._restore_descs)} buffers, "
            f"{len(self._directory)} cached pages"
        )

    def _reset_to_fresh(self, reason: str) -> None:
        """Turn a restore into a fresh pool: the daemon keeps the buffers this
        engine already maps, drops the rest and the directory, and this engine
        deposits its remaining buffers. Never remaps a mismatched pool."""
        keep = len(self._buffers)
        logger.warning(
            f"[KvRestore] the daemon's KV pool does not match this engine's KV "
            f"layout ({reason}); dropping the restored cache and starting fresh"
        )
        self._send({"type": "kv_reset", "keep": keep})
        for fd in self._restore_fds[keep:]:
            os.close(fd)
        self._restore_fds = []
        self._directory = []
        self._mode = "fresh"

    def detach(self) -> None:
        """Give up the session before any KV buffer was allocated: the
        engine runs without restore, and a restored pool stays with the
        daemon for a later engine."""
        for fd in self._restore_fds:
            if fd >= 0:
                os.close(fd)
        if self._conn is not None:
            self._conn.close()
            self._conn = None
        self._restore_fds, self._restore_descs, self._directory = [], [], []
        self._restore_token_capacity = None
        self._mode = "off"

    def allocate(self, shape, dtype: torch.dtype, device) -> torch.Tensor:
        """Allocate the next KV buffer: the restored one, or a new shared one."""
        if self._mode == "off":
            return torch.zeros(shape, dtype=dtype, device=device)
        device_id = torch.device(device).index
        if device_id is None:
            device_id = torch.cuda.current_device()
        desc = buffer_desc(shape, dtype, _nbytes(shape, dtype))
        index = len(self._buffers)
        if self._mode == "restore" and not (
            index < len(self._restore_descs)
            and fits_restored(desc, self._restore_descs[index])
        ):
            held = (
                self._restore_descs[index] if index < len(self._restore_descs) else None
            )
            self._reset_to_fresh(
                f"KV buffer {index} (shape, dtype, bytes) = {desc} does not fit the "
                f"pool's {held}"
            )
        if self._mode == "restore":
            fd = self._restore_fds[index]
            self._restore_fds[index] = -1
            try:
                buf = _SharedBuffer(self._restore_descs[index][2], device_id, fd=fd)
            finally:
                os.close(fd)
            self._buffers.append(buf)
            # A smaller pool uses the leading rows/pages of the restored buffer.
            return buf.tensor(shape, dtype)
        buf = _SharedBuffer(desc[2], device_id)
        self._buffers.append(buf)
        tensor = buf.tensor(shape, dtype)
        tensor.zero_()
        fd = buf.export_fd()
        try:
            with self._send_lock:
                send_msg(self._conn, {"type": "kv_deposit", "desc": desc})
                _send_fd(self._conn, fd, index)
        finally:
            os.close(fd)
        return tensor

    def finish(self, tree_cache, forward_stream: Optional[torch.cuda.Stream]) -> None:
        """Seal the layout, adopt the restored directory and start streaming
        directory updates from ``tree_cache``."""
        if self._mode in (None, "off"):
            return
        _check_supported(tree_cache, {buf.base for buf in self._buffers})
        if self._mode == "restore" and len(self._buffers) != len(self._restore_descs):
            self._reset_to_fresh(
                f"{len(self._buffers)} KV buffers, the pool has "
                f"{len(self._restore_descs)}"
            )
        if self._mode == "restore" and not _all_tp_ranks(True):
            self._reset_to_fresh("another TP rank's pool did not match")
        elif self._mode == "fresh" and self._restore_descs:
            _all_tp_ranks(False)  # this rank reset; the others must too
        if self._mode == "fresh":
            self._send(
                {
                    "type": "kv_layout_done",
                    "count": len(self._buffers),
                    "token_capacity": tree_cache.token_to_kv_pool_allocator.size,
                }
            )
        self._conn.settimeout(None)
        if self._mode == "restore":
            t0 = time.perf_counter()
            entries = _agree_across_tp(self._directory)
            adopted = import_directory(tree_cache, entries, self._page_size)
            # The daemon's directory becomes exactly this engine's tree: pages
            # that were not adopted (a broken chain, a duplicate the insert
            # freed, a rank disagreement) are free here and will be rewritten.
            committed = directory_of(tree_cache, self._page_size)
            self._send({"type": "kv_clear"})
            if committed:
                self._send({"type": "kv_commit", "pages": committed})
            logger.info(
                f"[KvRestore] adopted {adopted // self._page_size} of "
                f"{len(self._directory)} cached pages ({adopted} tokens) in "
                f"{(time.perf_counter() - t0) * 1e3:.1f} ms"
            )
            self._directory = []
        tree_cache.tree_core.kv_events.placement_listener = DirectoryListener(
            send=self._send,
            page_size=self._page_size,
            stream=forward_stream,
        )


def _check_supported(tree_cache, session_buffer_bases) -> None:
    """Fail at startup on a configuration whose KV the directory cannot
    describe, instead of restoring a wrong or empty cache."""
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    hint = "unset SGLANG_WEIGHT_CACHE_KV_RESTORE to run this configuration"
    if not isinstance(tree_cache, UnifiedRadixCache) or tree_cache.disable:
        raise RuntimeError(
            f"[KvRestore] needs the radix cache (UnifiedRadixCache) enabled; {hint}"
        )
    if not isinstance(tree_cache.tree_core, UnifiedTreeCore):
        # Other cores do not report per-node device stores/removes, so the
        # directory would silently stay empty.
        raise RuntimeError(
            f"[KvRestore] needs the Python TreeCore, which reports page "
            f"placements; got {type(tree_cache.tree_core).__name__}. {hint}"
        )
    if tuple(tree_cache.tree_components) != (ComponentType.FULL,):
        raise RuntimeError(
            f"[KvRestore] supports full-attention KV only; this cache has "
            f"components {[c.name for c in tree_cache.tree_components]}. {hint}"
        )
    kvcache = tree_cache.token_to_kv_pool_allocator.get_kvcache()
    buffers = _restorable_buffers(kvcache)
    if buffers is None or any(
        t.data_ptr() not in session_buffer_bases for t in buffers
    ):
        raise RuntimeError(
            f"[KvRestore] supports the standard MHA and MLA KV pools without KV "
            f"scale buffers; this engine uses {type(kvcache).__name__}. {hint}"
        )


def _all_tp_ranks(flag: bool) -> bool:
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.tp_size <= 1:
        return flag
    flags = [None] * parallel.tp_group.world_size
    torch.distributed.all_gather_object(flags, flag, group=parallel.tp_group.cpu_group)
    return all(flags)


def _agree_across_tp(entries: List[PageEntry]) -> List[PageEntry]:
    """TP ranks keep identical radix caches and allocators, but each rank's
    daemon may have received a different last commit before the previous
    engine died. Keep only the entries every rank has."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.tp_size <= 1:
        return entries
    group = parallel.tp_group
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
    is incomplete are skipped. All pages are claimed in one allocator call
    and copied to the device once, so the import does not sync per leaf."""
    paths = _leaf_paths(entries)
    if not paths:
        return 0
    allocator = tree_cache.token_to_kv_pool_allocator
    pages = list(dict.fromkeys(e[0] for path in paths for e in path))
    if allocator.alloc_pages(torch.tensor(pages, dtype=torch.int64)) is None:
        return _import_paths_one_by_one(tree_cache, paths, page_size)
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
        key = _path_key(path)
        tree_cache.import_prefix(
            key, page_ids, cached_prefix_pages=cached, claimed=True
        )
        adopted_pages.update(e[0] for e in path)
        adopted += (len(path) - cached) * page_size
    return adopted


def _path_key(path: List[PageEntry]):
    from sglang.srt.mem_cache.radix_cache import RadixKey

    tokens = array("q")
    for e in path:
        tokens.extend(e[2])
    return RadixKey(tokens, extra_key=path[0][3], cache_salt=path[0][4])


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


def _import_paths_one_by_one(tree_cache, paths, page_size: int) -> int:
    """Fallback when some page is not free: claim per leaf, skip failures."""
    adopted_pages: set = set()
    adopted = 0
    for path in paths:
        cached = 0
        while cached < len(path) and path[cached][0] in adopted_pages:
            cached += 1
        if cached == len(path):
            continue
        page_ids = torch.tensor([e[0] for e in path], dtype=torch.int64)
        if tree_cache.import_prefix(
            _path_key(path), page_ids, cached_prefix_pages=cached
        ):
            adopted_pages.update(e[0] for e in path)
            adopted += (len(path) - cached) * page_size
    return adopted


def _device_value(node) -> Optional[torch.Tensor]:
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType

    return node.component_data[ComponentType.FULL].value


def _node_entries(node, page_ids: List[int], parent_page: int, page_size: int):
    key = node.key
    tokens = list(key.token_ids[: len(key)])
    pages = []
    for i, page_id in enumerate(page_ids):
        chunk = tuple(tokens[i * page_size : (i + 1) * page_size])
        pages.append((page_id, parent_page, chunk, key.extra_key, key.cache_salt))
        parent_page = page_id
    return pages


def directory_of(tree_cache, page_size: int) -> List[PageEntry]:
    """Directory entries of every device page in ``tree_cache``, parents
    first; reads all page ids with one device-to-host copy."""
    root = tree_cache.tree_core.root_node
    nodes, stack = [], list(root.children.values())
    while stack:
        node = stack.pop()
        value = _device_value(node)
        if value is None or len(value) == 0 or node.key.is_bigram:
            continue  # not on device: its subtree's chain to the root is broken
        nodes.append(node)
        stack.extend(node.children.values())
    if not nodes:
        return []
    flat = torch.cat([_device_value(n)[::page_size] for n in nodes]).tolist()
    entries: List[PageEntry] = []
    last_page: Dict[int, int] = {}
    offset = 0
    for node in nodes:  # depth-first, so a parent precedes its children
        count = len(_device_value(node)[::page_size])
        page_ids = [index // page_size for index in flat[offset : offset + count]]
        offset += count
        parent = node.parent
        parent_page = (
            NO_PARENT_PAGE if parent is None or parent is root else last_page[parent.id]
        )
        entries.extend(_node_entries(node, page_ids, parent_page, page_size))
        last_page[node.id] = page_ids[-1]
    return entries


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

    def on_device_store(self, node) -> None:
        value = _device_value(node)
        if value is None or node.key.is_bigram:
            return
        parent = node.parent
        parent_page = NO_PARENT_PAGE
        if parent is not None and parent.parent is not None:
            parent_value = _device_value(parent)
            if parent_value is None or len(parent_value) == 0:
                return  # parent not on device: chain to the root is broken
            parent_page = int(parent_value[-1]) // self._page_size
        pages = _node_entries(node, self._page_ids(value), parent_page, self._page_size)
        event = None
        if self._stream is not None:
            event = torch.cuda.Event()
            event.record(self._stream)
        with self._lock:
            self._pending.append((event, pages))
            if event is None:
                self._flush_locked()

    def on_device_remove(self, node) -> None:
        value = _device_value(node)
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
        get_model,
        get_parallel,
        get_schedule,
    )

    if not envs.SGLANG_WEIGHT_CACHE_KV_RESTORE.get() or _session is not None:
        # One session per process: a draft model loaded from the same daemon
        # shares the target's pool, whose KV buffers include the draft's.
        return 0
    off_reason = None
    if get_model().weight_cache_mode != "client":
        # An engine-spawned daemon exits with its engine: nothing to restore.
        off_reason = (
            "needs a standalone daemon (python -m sglang.srt.weight_cache.daemon) "
            "and --weight-cache-mode client"
        )
    elif get_memory().enable_hierarchical_cache:
        # HiCache demotes device KV without a device-remove notice.
        off_reason = "is not supported with the hierarchical cache"
    elif get_exec().features.enable_memory_saver:
        # A memory saver would release KV under the daemon.
        off_reason = "is not supported with the memory saver"
    if off_reason is not None:
        logger.warning(f"[KvRestore] {off_reason}; KV restore is off")
        return 0
    parallel = get_parallel()
    if parallel.pp_size > 1 or parallel.attn_dp_size > 1:
        raise RuntimeError(
            "[KvRestore] supports tensor parallelism only (no pipeline or DP "
            "attention); unset SGLANG_WEIGHT_CACHE_KV_RESTORE"
        )
    page_size = get_schedule().page_size
    # Weights identity (the daemon's config) plus the KV layout choices that
    # change what a page holds without changing every buffer's shape.
    key = dict(
        cache_config,
        page_size=page_size,
        kv_cache_dtype=str(get_model().kv_cache_dtype),
    )
    _session = KvRestoreSession(socket_path, key, page_size=page_size)
    _session.attach()
    _agree_on_mode(_session)
    return _session.restored_bytes


def _agree_on_mode(session: KvRestoreSession) -> None:
    """Every TP rank restores the same pool size, or none restores: ranks must
    keep identical allocators and radix caches. One rank can differ, e.g. when
    its predecessor is still alive (busy) or its daemon was restarted."""
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    if parallel.tp_size <= 1:
        return
    mine = (session.mode, session.restored_token_capacity)
    gathered = [None] * parallel.tp_group.world_size
    torch.distributed.all_gather_object(
        gathered, mine, group=parallel.tp_group.cpu_group
    )
    if all(g == gathered[0] for g in gathered):
        return
    logger.warning(
        f"[KvRestore] TP ranks attached differently ({gathered}); KV restore "
        f"is off on every rank of this engine"
    )
    session.detach()


def current_session() -> Optional[KvRestoreSession]:
    return _session


def restored_bytes() -> int:
    """KV bytes the daemon holds that this engine maps instead of allocating."""
    return 0 if _session is None else _session.restored_bytes


def restored_token_capacity() -> Optional[int]:
    """Token capacity KV sizing must use so the restored buffers fit exactly."""
    return None if _session is None else _session.restored_token_capacity


# ===== Command line: drop a retained pool =====


def _request(socket_path: str, msg: Dict[str, Any], timeout_s: float = 30.0):
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout_s)
        conn.connect(socket_path)
        send_msg(conn, msg)
        return recv_msg(conn)


def pool_status(socket_path: str) -> Dict[str, Any]:
    """The KV pool the daemon at ``socket_path`` holds: {"status": "ok",
    "pool": None or {"bytes", "buffers", "pages", "token_capacity",
    "complete", "leased"}}."""
    return _request(socket_path, {"type": "kv_status"})


def drop_pool(socket_path: str, force: bool = False) -> Dict[str, Any]:
    """Ask the daemon at ``socket_path`` to free its KV pool. Returns its
    reply: "dropped" (with "bytes" and "pages"), "empty", or "busy" when an
    engine is attached and not ``force``."""
    return _request(socket_path, {"type": "kv_drop", "force": force})


def _visible_gpu_socket_paths() -> List[str]:
    from sglang.srt.weight_cache.protocol import get_socket_path

    return [
        get_socket_path(str(torch.cuda.get_device_properties(i).uuid))
        for i in range(torch.cuda.device_count())
    ]


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sglang.srt.weight_cache.kv_restore",
        description="Inspect or free the KV pools weight cache daemons keep "
        "for KV restore (SGLANG_WEIGHT_CACHE_KV_RESTORE).",
    )
    parser.add_argument("command", choices=["status", "drop"])
    parser.add_argument(
        "--socket",
        action="append",
        help="daemon socket path (repeatable); default: the daemons of all "
        "visible GPUs",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="drop: also when an engine is attached (it keeps its own mapping)",
    )
    args = parser.parse_args(argv)
    failed = False
    for path in args.socket or _visible_gpu_socket_paths():
        try:
            if args.command == "status":
                reply = pool_status(path)
            else:
                reply = drop_pool(path, force=args.force)
        except OSError as e:
            reply = {"status": f"unreachable ({e})"}
        failed |= reply["status"] not in ("ok", "dropped", "empty")
        print(f"{path}: {reply}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
