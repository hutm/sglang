"""Pluggable memory saver backends and contents-preserving KV allocation."""

import unittest
from contextlib import contextmanager

import pytest
import torch

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE
from sglang.srt.mem_cache import memory_pool
from sglang.srt.utils import torch_memory_saver_adapter as adapter_mod
from sglang.srt.utils.torch_memory_saver_adapter import (
    TorchMemorySaverAdapter,
    register_memory_saver_backend,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


class _HostBackend(TorchMemorySaverAdapter):
    """CPU-only backend that reports re-attached (preserved) KV memory."""

    def __init__(self, preserves: bool):
        self.preserves = preserves
        self.regions = []

    @contextmanager
    def region(self, tag: str, enable_cpu_backup: bool = False):
        self.regions.append(tag)
        yield

    def allocation_preserves_contents(self, tag: str) -> bool:
        return self.preserves and tag == GPU_MEMORY_TYPE_KV_CACHE

    @property
    def enabled(self):
        return True


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(adapter_mod, "_backends", dict(adapter_mod._backends))
    instances = []

    def select(preserves: bool) -> list:
        name = f"test-host-{preserves}"

        def factory():
            instances.append(_HostBackend(preserves))
            return instances[-1]

        register_memory_saver_backend(name, factory)
        monkeypatch.setattr(adapter_mod, "_configured_backend", lambda: name)
        return instances

    return select


@pytest.fixture
def alloc_calls(monkeypatch):
    calls = {"empty": 0, "zeros": 0}
    real_empty, real_zeros = torch.empty, torch.zeros

    def empty(*args, **kwargs):
        calls["empty"] += 1
        return real_empty(*args, **kwargs)

    def zeros(*args, **kwargs):
        calls["zeros"] += 1
        return real_zeros(*args, **kwargs)

    monkeypatch.setattr(memory_pool.torch, "empty", empty)
    monkeypatch.setattr(memory_pool.torch, "zeros", zeros)
    return calls


def _mha_pool():
    return memory_pool.MHATokenToKVPool(
        size=64,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=2,
        head_dim=8,
        layer_num=3,
        device="cpu",
        enable_memory_saver=True,
    )


def test_registered_backend_is_used_when_selected(backend):
    instances = backend(preserves=False)
    adapter = TorchMemorySaverAdapter.create(enable=True)
    assert adapter is instances[-1]
    # Disabled memory saver never consults the registry.
    assert not TorchMemorySaverAdapter.create(enable=False).enabled


def test_unknown_backend_is_rejected(monkeypatch):
    monkeypatch.setattr(adapter_mod, "_configured_backend", lambda: "missing")
    with pytest.raises(ValueError, match="unknown memory saver backend"):
        TorchMemorySaverAdapter.create(enable=True)


def test_duplicate_registration_is_rejected(monkeypatch):
    monkeypatch.setattr(adapter_mod, "_backends", dict(adapter_mod._backends))
    register_memory_saver_backend("dup", lambda: None)
    with pytest.raises(ValueError, match="already registered"):
        register_memory_saver_backend("dup", lambda: None)


def test_kv_buffers_keep_preserved_contents(backend, alloc_calls):
    instances = backend(preserves=True)
    pool = _mha_pool()
    assert GPU_MEMORY_TYPE_KV_CACHE in instances[-1].regions
    # One K and one V buffer per layer, none zero-filled.
    assert alloc_calls["empty"] >= 2 * 3
    assert len(pool.k_buffer) == len(pool.v_buffer) == 3


def test_kv_buffers_zero_fill_by_default(backend, alloc_calls):
    backend(preserves=False)
    pool = _mha_pool()
    assert alloc_calls["zeros"] >= 2 * 3
    assert all(torch.count_nonzero(b) == 0 for b in pool.k_buffer + pool.v_buffer)


def test_mla_buffers_keep_preserved_contents(backend, alloc_calls):
    backend(preserves=True)
    before = alloc_calls["empty"]
    pool = memory_pool.MLATokenToKVPool(
        size=32,
        page_size=1,
        kv_lora_rank=16,
        qk_rope_head_dim=8,
        dtype=torch.bfloat16,
        layer_num=2,
        device="cpu",
        enable_memory_saver=True,
    )
    assert alloc_calls["empty"] - before >= 2
    assert len(pool.kv_buffer) == 2


if __name__ == "__main__":
    unittest.main()
