"""release_memory_occupation keeps the radix cache when KV contents survive."""

import unittest
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from sglang.srt.constants import GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS
from sglang.srt.managers.scheduler_components.weight_updater import (
    SchedulerWeightUpdaterManager,
)
from sglang.srt.utils import torch_memory_saver_adapter as adapter_mod
from sglang.srt.utils.torch_memory_saver_adapter import TorchMemorySaverAdapter
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def _reset_backup_tags():
    yield
    TorchMemorySaverAdapter.set_cpu_backup_tags(())


def test_backed_up_tags_get_cpu_backup_regions(monkeypatch):
    calls = []

    @contextmanager
    def region(tag, enable_cpu_backup):
        calls.append((tag, enable_cpu_backup))
        yield

    monkeypatch.setattr(
        adapter_mod, "_memory_saver", SimpleNamespace(region=region), raising=False
    )
    saver = adapter_mod._TorchMemorySaverAdapterReal()
    TorchMemorySaverAdapter.set_cpu_backup_tags({GPU_MEMORY_TYPE_KV_CACHE})

    with saver.region(GPU_MEMORY_TYPE_KV_CACHE):
        pass
    with saver.region(GPU_MEMORY_TYPE_WEIGHTS):
        pass
    with saver.region(GPU_MEMORY_TYPE_WEIGHTS, enable_cpu_backup=True):
        pass

    assert calls == [
        (GPU_MEMORY_TYPE_KV_CACHE, True),
        (GPU_MEMORY_TYPE_WEIGHTS, False),
        (GPU_MEMORY_TYPE_WEIGHTS, True),
    ]
    assert saver.preserves_contents(GPU_MEMORY_TYPE_KV_CACHE)
    assert not saver.preserves_contents(GPU_MEMORY_TYPE_WEIGHTS)


def test_disabled_saver_never_preserves_contents():
    TorchMemorySaverAdapter.set_cpu_backup_tags({GPU_MEMORY_TYPE_KV_CACHE})
    saver = TorchMemorySaverAdapter.create(enable=False)
    assert not saver.preserves_contents(GPU_MEMORY_TYPE_KV_CACHE)


def _flush_decision(preserved: bool, flush_cache):
    manager = object.__new__(SchedulerWeightUpdaterManager)
    manager.memory_saver_adapter = SimpleNamespace(
        preserves_contents=lambda tag: preserved and tag == GPU_MEMORY_TYPE_KV_CACHE
    )
    return manager._release_flushes_cache(flush_cache)


def test_release_flushes_unless_kv_contents_are_preserved():
    assert _flush_decision(False, None) is True
    assert _flush_decision(True, None) is False
    assert _flush_decision(True, False) is False
    assert _flush_decision(True, True) is True


def test_release_ignores_unsafe_keep_request():
    # Keeping index entries for KV that was discarded would serve garbage.
    assert _flush_decision(False, False) is True


if __name__ == "__main__":
    unittest.main()
