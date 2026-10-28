# SPDX-License-Identifier: Apache-2.0
"""Pluggable tensor transport backends for weight_cache."""

from __future__ import annotations

import array
import logging
import os
import socket
import struct
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from sglang.srt.utils import MultiprocessingSerializer

from .protocol import send_msg

logger = logging.getLogger(__name__)

TORCH_IPC_BACKEND = "torch_ipc"
VMM_FD_BACKEND = "vmm_fd"

_FD_INDEX_STRUCT = struct.Struct("<Q")


def _send_fd(sock: socket.socket, fd: int, index: int) -> None:
    payload = _FD_INDEX_STRUCT.pack(index)
    fds = array.array("i", [int(fd)])
    sent = sock.sendmsg(
        [payload], [(socket.SOL_SOCKET, socket.SCM_RIGHTS, fds.tobytes())]
    )
    if sent != len(payload):
        raise RuntimeError(f"sendmsg sent {sent} bytes, expected {len(payload)}")


def _recv_fd(sock: socket.socket) -> Tuple[int, int]:
    fd_item_size = array.array("i").itemsize
    data, ancdata, _, _ = sock.recvmsg(
        _FD_INDEX_STRUCT.size, socket.CMSG_SPACE(fd_item_size)
    )
    if len(data) != _FD_INDEX_STRUCT.size:
        raise RuntimeError(
            f"received truncated fd header: {len(data)} < {_FD_INDEX_STRUCT.size}"
        )
    index = _FD_INDEX_STRUCT.unpack(data)[0]
    fds = array.array("i")
    for level, cmsg_type, cmsg_data in ancdata:
        if level == socket.SOL_SOCKET and cmsg_type == socket.SCM_RIGHTS:
            fds.frombytes(cmsg_data[: len(cmsg_data) - (len(cmsg_data) % fd_item_size)])
    if len(fds) != 1:
        for fd in fds:
            os.close(fd)
        raise RuntimeError(f"expected one fd, got {len(fds)}")
    return int(index), int(fds[0])


class ShareableAllocation:
    """A CUDA VMM allocation that other processes map by its POSIX fd.

    :meth:`create` makes and maps a new allocation; :meth:`import_fd` maps
    one that another process exported. Each process that maps it holds its
    own reference to the physical memory, which is freed once all of them
    have closed it.
    """

    def __init__(self, base: int, size: int, device_id: int, reservation=None):
        self.base = base
        self.size = size
        self.device_id = device_id
        self._reservation = reservation
        self._handle = None

    @staticmethod
    def _prop_and_size(nbytes: int, device_id: int):
        from sglang.srt.utils.cuda_vmm_utils import (
            align_up,
            get_allocation_granularity,
            make_device_allocation_prop,
        )

        prop = make_device_allocation_prop(
            device_id, handle_types=int(_posix_fd_type())
        )
        return prop, align_up(max(nbytes, 1), get_allocation_granularity(prop))

    @classmethod
    def aligned_size(cls, nbytes: int, device_id: int) -> int:
        """The size :meth:`create` allocates for ``nbytes``."""
        return cls._prop_and_size(nbytes, device_id)[1]

    @classmethod
    def create(cls, nbytes: int, device_id: int) -> ShareableAllocation:
        from sglang.srt.utils.cuda_vmm_utils import VmmReservation

        prop, size = cls._prop_and_size(nbytes, device_id)
        reservation = VmmReservation(size, prop, device_id)
        try:
            handle = reservation.map(0, size, retain_handle=True)
        except BaseException:
            reservation.close()
            raise
        allocation = cls(reservation.base, size, device_id, reservation)
        allocation._handle = handle
        return allocation

    @classmethod
    def import_fd(cls, fd: int, size: int, device_id: int) -> ShareableAllocation:
        """Map an exported allocation of ``size`` bytes; the caller keeps ``fd``."""
        from sglang.srt.utils.cuda_vmm_utils import import_and_map_alloc

        base = import_and_map_alloc(
            None, fd, size, device_id, use_fabric=False, peer_rank=0
        )
        return cls(base, size, device_id)

    def export_fd(self) -> int:
        """A new POSIX fd for the allocation; the caller closes it."""
        from sglang.srt.utils.cuda_vmm_utils import check_drv

        return int(
            check_drv(
                _cuda_driver().cuMemExportToShareableHandle(
                    self._handle, _posix_fd_type(), 0
                ),
                "cuMemExportToShareableHandle(POSIX_FD)",
            )
        )

    def tensor(self, offset: int, nbytes: int) -> torch.Tensor:
        """A uint8 view of ``nbytes`` bytes at ``offset``."""
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        return tensor_from_pointer(self.base + offset, nbytes, device_id=self.device_id)

    def close(self) -> None:
        """Drop this process's mapping and reference."""
        if self._reservation is not None:
            self._reservation.close(release_handles=True)
            self._reservation = None
            self._handle = None
        elif self.base:
            drv = _cuda_driver()
            drv.cuMemUnmap(self.base, self.size)
            drv.cuMemAddressFree(self.base, self.size)
        self.base = 0


def _posix_fd_type():
    drv = _cuda_driver()
    return drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR


class WeightCacheTransportBackend(ABC):
    name: str
    # True when a client's imported tensors stay valid after the exporting
    # daemon exits (the client holds its own reference to the memory).
    mappings_outlive_exporter: bool = False

    @abstractmethod
    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        """Prepare daemon-side entries for all tensors."""

    @abstractmethod
    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
        preloaded_weights_bytes: int = 0,
    ) -> None:
        """Send a successful fetch_state response."""

    @abstractmethod
    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Client-side receive hook after recv_msg."""

    @abstractmethod
    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        """Import a single tensor from one entry."""


class TorchIpcTransportBackend(WeightCacheTransportBackend):
    name = TORCH_IPC_BACKEND

    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        entries: Dict[str, Dict[str, Any]] = {}
        for name, (tensor, is_param) in state_tensors.items():
            entries[name] = {
                "handle": MultiprocessingSerializer.serialize(
                    tensor.data, output_str=True
                ),
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": is_param,
            }
        return entries

    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
        preloaded_weights_bytes: int = 0,
    ) -> None:
        send_msg(
            conn,
            {
                "status": "ok",
                "config": config,
                "entries": entries,
                "pid": pid,
                "transport_backend": self.name,
                "preloaded_weights_bytes": preloaded_weights_bytes,
            },
        )

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        return MultiprocessingSerializer.deserialize(entry["handle"])


class VmmFdTransportBackend(WeightCacheTransportBackend):
    """Share all exported tensors as one CUDA VMM allocation passed by POSIX fd.

    The daemon packs the tensors into a single exportable allocation
    (``cuMemCreate`` with a POSIX file-descriptor handle), re-points its own
    model at that copy, and sends each client one fd plus per-tensor offsets.
    A client imports the handle, maps it, and builds zero-copy tensors over
    the mapping. Unlike legacy CUDA IPC, the client holds its own reference to
    the physical memory, so its mappings stay valid if the daemon exits.
    """

    name = VMM_FD_BACKEND
    mappings_outlive_exporter = True
    # Per-tensor alignment inside the arena; covers every dtype and the
    # alignment kernels assume for weight tensors.
    TENSOR_ALIGNMENT = 512

    def __init__(self):
        self._device_id: Optional[int] = None
        # The daemon's exported arena, or the client's mapping of it.
        self._arena: Optional[ShareableAllocation] = None

    @classmethod
    def can_export_state(
        cls, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> bool:
        from sglang.srt.environ import envs

        if envs.SGLANG_WEIGHT_CACHE_TRANSPORT.get() != VMM_FD_BACKEND:
            return False
        devices = {t.device for t, _ in state_tensors.values()}
        if len(devices) != 1 or next(iter(devices)).type != "cuda":
            logger.warning(
                "[weight_cache] vmm_fd needs every tensor on one CUDA device; "
                "falling back to %s",
                TORCH_IPC_BACKEND,
            )
            return False
        return True

    @classmethod
    def _layout(
        cls, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Tuple[Dict[str, Dict[str, Any]], int]:
        entries: Dict[str, Dict[str, Any]] = {}
        # Copy whole storages, once each, and keep each tensor's stride and
        # storage offset: tied and fused views keep sharing memory, and
        # non-contiguous (e.g. transposed) weights keep the layout kernels expect.
        placed: Dict[int, int] = {}
        offset = 0
        for name, (tensor, is_param) in state_tensors.items():
            storage = tensor.untyped_storage()
            storage_key = storage.data_ptr()
            if storage_key not in placed:
                offset = -(-offset // cls.TENSOR_ALIGNMENT) * cls.TENSOR_ALIGNMENT
                placed[storage_key] = offset
                offset += storage.nbytes()
            entries[name] = {
                "offset": placed[storage_key],
                "storage_nbytes": storage.nbytes(),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "storage_offset": tensor.storage_offset(),
                "dtype": str(tensor.dtype).replace("torch.", ""),
                "is_param": is_param,
            }
        return entries, offset

    def prepare_export(
        self, state_tensors: Mapping[str, Tuple[torch.Tensor, bool]]
    ) -> Dict[str, Dict[str, Any]]:
        entries, used = self._layout(state_tensors)
        device_id = next(iter(state_tensors.values()))[0].device.index
        arena = ShareableAllocation.create(used, device_id)
        try:
            copied = set()
            for name, (tensor, _) in state_tensors.items():
                entry = entries[name]
                if entry["offset"] in copied or not entry["storage_nbytes"]:
                    continue
                copied.add(entry["offset"])
                dst = arena.tensor(entry["offset"], entry["storage_nbytes"])
                src = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                src.set_(tensor.untyped_storage(), 0, (entry["storage_nbytes"],))
                dst.copy_(src)
            torch.cuda.synchronize(device_id)
        except BaseException:
            arena.close()
            raise
        self._device_id = device_id
        self._arena = arena
        return entries

    def tensor_for_entry(self, entry: Dict[str, Any]) -> torch.Tensor:
        """Daemon-side view of an exported tensor inside the shared arena."""
        return self._view(entry)

    def _view(self, entry: Dict[str, Any]) -> torch.Tensor:
        dtype = getattr(torch, entry["dtype"])
        if entry["storage_nbytes"] == 0:
            return torch.empty_strided(
                entry["shape"],
                entry["stride"],
                dtype=dtype,
                device=f"cuda:{self._device_id}",
            )
        storage = self._arena.tensor(entry["offset"], entry["storage_nbytes"])
        return storage.view(dtype).as_strided(
            entry["shape"], entry["stride"], entry["storage_offset"]
        )

    def send_fetch_state_response(
        self,
        conn: socket.socket,
        *,
        config: Dict[str, Any],
        entries: Dict[str, Dict[str, Any]],
        pid: int,
        preloaded_weights_bytes: int = 0,
    ) -> None:
        fd = self._arena.export_fd()
        try:
            send_msg(
                conn,
                {
                    "status": "ok",
                    "config": config,
                    "entries": entries,
                    "pid": pid,
                    "transport_backend": self.name,
                    "preloaded_weights_bytes": preloaded_weights_bytes,
                    "arena_size": self._arena.size,
                },
            )
            _send_fd(conn, fd, 0)
        finally:
            os.close(fd)

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        _, fd = _recv_fd(sock)
        try:
            self._device_id = torch.cuda.current_device()
            self._arena = ShareableAllocation.import_fd(
                fd, int(result["arena_size"]), self._device_id
            )
        finally:
            os.close(fd)
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        if self._arena is None:
            raise RuntimeError("vmm_fd import_tensor before the arena was mapped")
        return self._view(entry)

    def close(self) -> None:
        """Unmap this process's view; physical memory is freed once every
        process that imported or created it has released it."""
        if self._arena is not None:
            self._arena.close()
            self._arena = None


def _cuda_driver():
    from sglang.srt.utils.cuda_vmm_utils import _get_cuda_driver

    return _get_cuda_driver()


def choose_daemon_transport_backend(
    state_tensors: Mapping[str, Tuple[torch.Tensor, bool]],
) -> WeightCacheTransportBackend:
    if VmmFdTransportBackend.can_export_state(state_tensors):
        logger.info("[weight_cache] Using transport backend: %s", VMM_FD_BACKEND)
        return VmmFdTransportBackend()
    logger.info("[weight_cache] Using transport backend: %s", TORCH_IPC_BACKEND)
    return TorchIpcTransportBackend()


def get_client_transport_backend(name: Optional[str]) -> WeightCacheTransportBackend:
    if name in (None, "", TORCH_IPC_BACKEND):
        return TorchIpcTransportBackend()
    if name == VMM_FD_BACKEND:
        return VmmFdTransportBackend()
    raise RuntimeError(f"Unknown weight cache transport backend {name!r}")
