"""CPU checks for the vmm_fd weight-cache transport: layout and selection."""

import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.weight_cache.transport import (
    TORCH_IPC_BACKEND,
    TorchIpcTransportBackend,
    VmmFdTransportBackend,
    choose_daemon_transport_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


def test_layout_aligns_storages_and_shares_tied_weights():
    w = torch.zeros(3, 5, dtype=torch.bfloat16)
    b = torch.zeros(7, dtype=torch.float32)
    entries, used = VmmFdTransportBackend._layout(
        {"w": (w, True), "tied": (w, True), "b": (b, False)}
    )
    align = VmmFdTransportBackend.TENSOR_ALIGNMENT
    assert entries["w"]["offset"] == entries["tied"]["offset"] == 0
    assert entries["b"]["offset"] == align
    assert entries["b"]["storage_nbytes"] == 7 * 4
    assert used == align + 7 * 4
    assert entries["w"]["shape"] == [3, 5]
    assert entries["w"]["dtype"] == "bfloat16"
    assert entries["b"]["is_param"] is False


def test_layout_keeps_strides_and_shared_storage_of_views():
    # A transposed weight must keep its strides, and views of one fused
    # storage must keep sharing it, so kernels see the daemon's layout.
    fused = torch.zeros(2, 4, 6)
    wt = torch.zeros(6, 4).t()
    entries, used = VmmFdTransportBackend._layout(
        {"q": (fused[0], True), "k": (fused[1], True), "wt": (wt, True)}
    )
    assert entries["q"]["offset"] == entries["k"]["offset"]
    assert entries["k"]["storage_offset"] == 24
    assert entries["q"]["storage_nbytes"] == fused.numel() * 4
    assert entries["wt"]["shape"] == [4, 6]
    assert entries["wt"]["stride"] == [1, 4]
    assert used == VmmFdTransportBackend.TENSOR_ALIGNMENT + wt.numel() * 4


def test_vmm_fd_is_opt_in():
    state = {"w": (torch.zeros(4), True)}
    with envs.SGLANG_WEIGHT_CACHE_TRANSPORT.override(TORCH_IPC_BACKEND):
        assert not VmmFdTransportBackend.can_export_state(state)
        assert isinstance(
            choose_daemon_transport_backend(state), TorchIpcTransportBackend
        )


def test_vmm_fd_needs_cuda_tensors():
    with envs.SGLANG_WEIGHT_CACHE_TRANSPORT.override("vmm_fd"):
        assert not VmmFdTransportBackend.can_export_state({"w": (torch.zeros(4), True)})


if __name__ == "__main__":
    unittest.main()
