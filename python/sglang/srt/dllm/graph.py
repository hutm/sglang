from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


def _noop() -> None:
    pass


@dataclass(frozen=True)
class DllmGraphPhaseHooks:
    """Weight-view callbacks invoked immediately before graph capture phases."""

    before_draft: Callable[[], None] = _noop
    before_verify: Callable[[], None] = _noop
    after_capture: Callable[[], None] = _noop


def register_dllm_graph_phase_hooks(
    model_runner: "ModelRunner", hooks: DllmGraphPhaseHooks
) -> None:
    """Attach DLLM graph hooks before the decode graph runner is constructed."""
    if getattr(model_runner, "decode_cuda_graph_runner", None) is not None:
        raise RuntimeError("DLLM graph phase hooks must be registered before capture")
    model_runner.dllm_graph_phase_hooks = hooks
