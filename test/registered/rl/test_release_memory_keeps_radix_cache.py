"""KV cache CPU backup keeps the radix cache valid across release/resume."""

import unittest

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=120, stage="extra-b", runner_config="1-gpu-h100")

MODEL = "Qwen/Qwen3-0.6B"
PROMPT = "The quick brown fox jumps over the lazy dog. " * 32
SAMPLING = {"temperature": 0, "max_new_tokens": 16}


class TestReleaseMemoryKeepsRadixCache(CustomTestCase):
    def _run(self, kv_cache_cpu_backup: bool):
        engine = sgl.Engine(
            model_path=MODEL,
            enable_memory_saver=True,
            enable_kv_cache_cpu_backup=kv_cache_cpu_backup,
            mem_fraction_static=0.2,
            disable_cuda_graph=True,
            random_seed=0,
        )
        try:
            first = engine.generate(PROMPT, SAMPLING)
            engine.release_memory_occupation(tags=["kv_cache"])
            engine.resume_memory_occupation(tags=["kv_cache"])
            second = engine.generate(PROMPT, SAMPLING)
        finally:
            engine.shutdown()
        self.assertEqual(first["output_ids"], second["output_ids"])
        return second["meta_info"]["cached_tokens"]

    def test_backed_up_kv_cache_keeps_prefix_hits(self):
        self.assertGreater(self._run(kv_cache_cpu_backup=True), 0)

    def test_released_kv_cache_flushes_prefix_hits(self):
        self.assertEqual(self._run(kv_cache_cpu_backup=False), 0)


if __name__ == "__main__":
    unittest.main()
