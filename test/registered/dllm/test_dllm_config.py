"""Unit tests for DLLM configuration and block-tier policy."""

import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.dllm.config import DllmConfig, load_dllm_algorithm_config
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")


def _make_config(tiers=None, max_running_requests=128, static_block_size=32):
    return DllmConfig(
        algorithm="ExternalTestAlgorithm",
        algorithm_config={},
        block_size=static_block_size,
        mask_id=0,
        max_running_requests=max_running_requests,
        max_steps=1,
        causal_context=False,
        block_size_tiers=tiers,
    )


class TestSelectBlockSize(unittest.TestCase):
    def test_no_tiers_uses_static_block_size(self):
        cfg = _make_config(tiers=None, static_block_size=32)
        self.assertEqual(cfg.select_block_size(1), 32)
        self.assertEqual(cfg.select_block_size(64), 32)
        self.assertEqual(cfg.select_block_size(10_000), 32)

    def test_tier_dispatch(self):
        tiers = [
            {"max_running_bs": 4, "block_size": 32},
            {"max_running_bs": 32, "block_size": 16},
            {"max_running_bs": 9999, "block_size": 8},
        ]
        cfg = _make_config(tiers=tiers, max_running_requests=128)
        self.assertEqual(cfg.select_block_size(1), 32)
        self.assertEqual(cfg.select_block_size(4), 32)
        self.assertEqual(cfg.select_block_size(5), 16)
        self.assertEqual(cfg.select_block_size(32), 16)
        self.assertEqual(cfg.select_block_size(33), 8)
        self.assertEqual(cfg.select_block_size(128), 8)

    def test_running_bs_beyond_last_tier_uses_last_block(self):
        tiers = [
            {"max_running_bs": 4, "block_size": 32},
            {"max_running_bs": 9999, "block_size": 8},
        ]
        cfg = _make_config(tiers=tiers, max_running_requests=128)
        self.assertEqual(cfg.select_block_size(99_999), 8)

    def test_unsorted_tiers_normalized(self):
        tiers = [
            {"max_running_bs": 9999, "block_size": 8},
            {"max_running_bs": 4, "block_size": 32},
            {"max_running_bs": 32, "block_size": 16},
        ]
        cfg = _make_config(tiers=tiers, max_running_requests=128)
        self.assertEqual(cfg.select_block_size(2), 32)
        self.assertEqual(cfg.select_block_size(16), 16)
        self.assertEqual(cfg.select_block_size(64), 8)

    def test_duplicate_max_running_bs_rejected(self):
        tiers = [
            {"max_running_bs": 4, "block_size": 32},
            {"max_running_bs": 4, "block_size": 16},
        ]
        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            _make_config(tiers=tiers)

    def test_last_tier_below_max_running_requests_rejected(self):
        tiers = [
            {"max_running_bs": 4, "block_size": 32},
            {"max_running_bs": 32, "block_size": 16},
        ]
        with self.assertRaisesRegex(ValueError, "catch-all tier"):
            _make_config(tiers=tiers, max_running_requests=128)

    def test_non_positive_tier_values_rejected(self):
        for tier in (
            {"max_running_bs": 0, "block_size": 8},
            {"max_running_bs": 128, "block_size": -1},
        ):
            with self.subTest(tier=tier):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    _make_config(tiers=[tier])

    def test_static_block_size_set_to_max_tier(self):
        # When tiers are configured, the canonical block_size is the max of
        # the tier block_sizes (used for KV pool worst-case allocation).
        tiers = [
            {"max_running_bs": 4, "block_size": 32},
            {"max_running_bs": 9999, "block_size": 8},
        ]
        cfg = _make_config(tiers=tiers, max_running_requests=128, static_block_size=64)
        self.assertEqual(cfg.block_size, 32)  # max(32, 8)


class TestFromServerArgs(unittest.TestCase):
    def test_algorithm_config_must_be_mapping(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write("- not\n- a\n- mapping\n")
            f.flush()
            with self.assertRaisesRegex(ValueError, "YAML mapping"):
                load_dllm_algorithm_config(f.name)

    def test_max_steps_defaults_to_yaml_block_size(self):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write("block_size: 8\n")
            f.flush()
            server_args = SimpleNamespace(
                dllm_algorithm="FastDiffuser",
                dllm_algorithm_config=f.name,
                max_running_requests=None,
                model_path="dummy",
                revision=None,
            )
            model_config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["NemotronLabsDiffusionModel"])
            )
            with patch(
                "sglang.srt.configs.model_config.ModelConfig.from_server_args",
                return_value=model_config,
            ):
                cfg = DllmConfig.from_server_args(server_args)

        self.assertEqual(cfg.block_size, 8)
        self.assertEqual(cfg.max_steps, 8)


class TestServerArgsDllmValidation(unittest.TestCase):
    def test_pipeline_parallelism_is_disabled_for_dllm(self):
        from sglang.srt.server_args import ServerArgs

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write("block_size: 32\n")
            f.flush()
            server_args = ServerArgs(
                model_path="dummy",
                dllm_algorithm="ExternalTestAlgorithm",
                dllm_algorithm_config=f.name,
                attention_backend="flashinfer",
                disable_radix_cache=True,
                disable_overlap_schedule=True,
                max_running_requests=128,
                pp_size=2,
            )
            model_config = SimpleNamespace(
                hf_config=SimpleNamespace(architectures=["NemotronLabsDiffusionModel"])
            )
            with patch(
                "sglang.srt.configs.model_config.ModelConfig.from_server_args",
                return_value=model_config,
            ):
                server_args._handle_dllm_inference()

        self.assertEqual(server_args.pp_size, 1)


if __name__ == "__main__":
    unittest.main()
