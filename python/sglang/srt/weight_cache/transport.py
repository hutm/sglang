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
        self._handle = None  # daemon: exportable allocation handle
        self._reservation = None  # daemon: arena mapping
        self._client_va: Optional[int] = None
        self._client_size = 0
        self._arena_size = 0

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
        from sglang.srt.utils.cuda_vmm_utils import (
            VmmReservation,
            align_up,
            get_allocation_granularity,
            make_device_allocation_prop,
            tensor_from_pointer,
        )

        drv = _cuda_driver()
        entries, used = self._layout(state_tensors)
        device_id = next(iter(state_tensors.values()))[0].device.index
        prop = make_device_allocation_prop(
            device_id,
            handle_types=int(
                drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
            ),
        )
        size = align_up(max(used, 1), get_allocation_granularity(prop))
        reservation = VmmReservation(size, prop, device_id)
        try:
            handle = reservation.map(0, size, retain_handle=True)
            copied = set()
            for name, (tensor, _) in state_tensors.items():
                entry = entries[name]
                if entry["offset"] in copied or not entry["storage_nbytes"]:
                    continue
                copied.add(entry["offset"])
                dst = tensor_from_pointer(
                    reservation.base + entry["offset"],
                    entry["storage_nbytes"],
                    device_id=device_id,
                )
                src = torch.empty(0, dtype=torch.uint8, device=tensor.device)
                src.set_(tensor.untyped_storage(), 0, (entry["storage_nbytes"],))
                dst.copy_(src)
            torch.cuda.synchronize(device_id)
        except BaseException:
            reservation.close()
            raise
        self._device_id = device_id
        self._handle = handle
        self._reservation = reservation
        self._arena_size = size
        return entries

    def tensor_for_entry(self, entry: Dict[str, Any]) -> torch.Tensor:
        """Daemon-side view of an exported tensor inside the shared arena."""
        return self._view(self._reservation.base, entry)

    def _view(self, base: int, entry: Dict[str, Any]) -> torch.Tensor:
        from sglang.srt.utils.cuda_vmm_utils import tensor_from_pointer

        dtype = getattr(torch, entry["dtype"])
        if entry["storage_nbytes"] == 0:
            return torch.empty_strided(
                entry["shape"],
                entry["stride"],
                dtype=dtype,
                device=f"cuda:{self._device_id}",
            )
        storage = tensor_from_pointer(
            base + entry["offset"], entry["storage_nbytes"], device_id=self._device_id
        )
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
        drv = _cuda_driver()
        posix_fd = (
            drv.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
        from sglang.srt.utils.cuda_vmm_utils import check_drv

        fd = int(
            check_drv(
                drv.cuMemExportToShareableHandle(self._handle, posix_fd, 0),
                "cuMemExportToShareableHandle(POSIX_FD)",
            )
        )
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
                    "arena_size": self._arena_size,
                },
            )
            _send_fd(conn, fd, 0)
        finally:
            os.close(fd)

    def recv_fetch_state_response(
        self, sock: socket.socket, result: Dict[str, Any]
    ) -> Dict[str, Any]:
        from sglang.srt.utils.cuda_vmm_utils import import_and_map_alloc

        _, fd = _recv_fd(sock)
        try:
            self._device_id = torch.cuda.current_device()
            self._client_size = int(result["arena_size"])
            self._client_va = import_and_map_alloc(
                None,
                fd,
                self._client_size,
                self._device_id,
                use_fabric=False,
                peer_rank=0,
            )
        finally:
            os.close(fd)
        return result

    def import_tensor(self, entry: Dict[str, Any]) -> torch.Tensor:
        if self._client_va is None:
            raise RuntimeError("vmm_fd import_tensor before the arena was mapped")
        return self._view(self._client_va, entry)

    def close(self) -> None:
        """Unmap this process's view; physical memory is freed once every
        process that imported or created it has released it."""
        drv = _cuda_driver()
        if self._client_va is not None:
            drv.cuMemUnmap(self._client_va, self._client_size)
            drv.cuMemAddressFree(self._client_va, self._client_size)
            self._client_va = None
        if self._reservation is not None:
            self._reservation.close(release_handles=True)
            self._reservation = None
            self._handle = None


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
