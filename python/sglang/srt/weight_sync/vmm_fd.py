"""Share a flattened weight bucket as a CUDA VMM allocation passed by fd.

For ``update_weights_from_tensor(load_format="flattened_bucket")``. The
trainer copies its flattened bucket into a :class:`SharedFlattenedTensor`,
which pickles to a small handle. When the server unpickles it, it fetches the
allocation's fd from the trainer's private Unix socket (SCM_RIGHTS) and maps
it, holding its own reference to the memory.

PyTorch can share VMM memory itself (expandable segments), but the importer
then pulls the fd with ``pidfd_getfd``, which the default Yama ptrace policy
only allows from an ancestor of the exporter; trainers and servers are usually
siblings or unrelated, so that import fails with EPERM.
"""

import logging
import os
import socket
import struct
import tempfile
import threading
import uuid

import torch

from sglang.srt.weight_cache.transport import ShareableAllocation, _recv_fd, _send_fd

logger = logging.getLogger(__name__)

_PEERCRED = struct.Struct("3i")  # pid, uid, gid


def _peer_uid(conn: socket.socket) -> int:
    creds = conn.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, _PEERCRED.size)
    return _PEERCRED.unpack(creds)[1]


class SharedFlattenedTensor:
    """Trainer side: a uint8 CUDA buffer whose fd is served on a private socket.

    Fill ``tensor`` (or build with :meth:`from_tensor`), put this object in
    place of the flattened tensor in the bucket dict, and ``close()`` it once
    the server has applied the update. Only processes of the same user may
    fetch the fd.
    """

    def __init__(self, nbytes: int, device_id: int):
        self.nbytes = nbytes
        self._allocation = ShareableAllocation.create(nbytes, device_id)
        self.size = self._allocation.size
        self.tensor = self._allocation.tensor(0, nbytes)
        self.transfer_id = uuid.uuid4().hex
        self._dir = tempfile.mkdtemp(prefix="sglang_wt_")
        os.chmod(self._dir, 0o700)
        self.socket_path = os.path.join(self._dir, "bucket.sock")
        self._server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._server.bind(self.socket_path)
        os.chmod(self.socket_path, 0o600)
        self._server.listen(64)
        self._closed = threading.Event()
        self._thread = threading.Thread(
            target=self._serve, name="sglang-weight-sync-fd", daemon=True
        )
        self._thread.start()

    @classmethod
    def from_tensor(cls, tensor: torch.Tensor) -> "SharedFlattenedTensor":
        flat = tensor.contiguous().view(-1).view(torch.uint8)
        shared = cls(flat.numel(), tensor.device.index)
        shared.tensor.copy_(flat)
        torch.cuda.synchronize(tensor.device)
        return shared

    def _serve(self) -> None:
        while not self._closed.is_set():
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            with conn:
                try:
                    if _peer_uid(conn) != os.getuid():
                        logger.warning("Rejected weight sync peer of another user")
                        continue
                    fd = self._allocation.export_fd()
                    try:
                        _send_fd(conn, fd, 0)
                    finally:
                        os.close(fd)
                except OSError as e:
                    logger.warning("Weight sync fd handoff failed: %s", e)

    def __reduce__(self):
        return (
            rebuild_vmm_fd_tensor,
            (self.socket_path, self.size, self.nbytes, self.transfer_id),
        )

    def close(self) -> None:
        """Stop serving and drop the trainer's reference; servers that mapped
        the buffer keep theirs until their tensor is freed."""
        self._closed.set()
        self._server.close()
        self._thread.join(timeout=5)
        try:
            os.unlink(self.socket_path)
            os.rmdir(self._dir)
        except OSError:
            pass
        torch.cuda.synchronize()
        self._allocation.close()


class _ImportedMapping:
    """Owns one imported mapping. Exposed through ``__cuda_array_interface__``
    so a tensor built from it keeps it alive until the last view is freed."""

    def __init__(self, allocation: ShareableAllocation, nbytes: int):
        self.allocation = allocation
        self.__cuda_array_interface__ = {
            "shape": (nbytes,),
            "typestr": "|u1",
            "data": (allocation.base, False),
            "version": 3,
            "strides": None,
            "stream": None,
        }

    def __del__(self):
        self.allocation.close()


def rebuild_vmm_fd_tensor(
    socket_path: str, size: int, nbytes: int, transfer_id: str
) -> torch.Tensor:
    """Server side: map a :class:`SharedFlattenedTensor` and return a uint8
    tensor that owns the mapping (released when its storage is freed)."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(60)
        conn.connect(socket_path)
        if _peer_uid(conn) != os.getuid():
            raise RuntimeError("weight sync socket is owned by another user")
        _, fd = _recv_fd(conn)
    device_id = torch.cuda.current_device()
    try:
        allocation = ShareableAllocation.import_fd(fd, int(size), device_id)
    finally:
        os.close(fd)
    mapping = _ImportedMapping(allocation, int(nbytes))
    return torch.as_tensor(mapping, device=f"cuda:{device_id}")
