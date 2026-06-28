import unittest
from types import SimpleNamespace

from sglang.srt.dllm.graph import (
    DllmGraphPhaseHooks,
    register_dllm_graph_phase_hooks,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestDllmGraphPhaseHooks(unittest.TestCase):
    def test_registers_hooks_before_graph_construction(self):
        runner = SimpleNamespace(decode_cuda_graph_runner=None)
        hooks = DllmGraphPhaseHooks()

        register_dllm_graph_phase_hooks(runner, hooks)

        self.assertIs(runner.dllm_graph_phase_hooks, hooks)

    def test_rejects_registration_after_graph_construction(self):
        runner = SimpleNamespace(decode_cuda_graph_runner=object())

        with self.assertRaisesRegex(RuntimeError, "before capture"):
            register_dllm_graph_phase_hooks(runner, DllmGraphPhaseHooks())


if __name__ == "__main__":
    unittest.main()
