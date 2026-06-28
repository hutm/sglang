import unittest

from sglang.srt.dllm.algorithm import (
    algo_name_to_cls,
    get_algorithm,
    register_algorithm,
)
from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestDllmAlgorithmRegistry(unittest.TestCase):
    def tearDown(self):
        algo_name_to_cls.pop("ExternalTestAlgorithm", None)

    def test_external_algorithm_registration_and_resolution(self):
        @register_algorithm("ExternalTestAlgorithm")
        class ExternalAlgorithm(DllmAlgorithm):
            pass

        config = DllmConfig(
            algorithm="ExternalTestAlgorithm",
            algorithm_config={},
            block_size=4,
            mask_id=100,
            max_running_requests=1,
            max_steps=4,
        )

        algorithm = get_algorithm(config)

        self.assertIsInstance(algorithm, ExternalAlgorithm)
        self.assertEqual(algorithm.block_size, 4)

    def test_duplicate_name_is_rejected(self):
        @register_algorithm("ExternalTestAlgorithm")
        class FirstAlgorithm(DllmAlgorithm):
            pass

        with self.assertRaisesRegex(ValueError, "already registered"):

            @register_algorithm("ExternalTestAlgorithm")
            class SecondAlgorithm(DllmAlgorithm):
                pass


if __name__ == "__main__":
    unittest.main()
