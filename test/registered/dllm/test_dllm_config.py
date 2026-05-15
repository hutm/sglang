"""Unit tests for DllmConfig.select_block_size + tier validation."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.dllm.config import DllmConfig


def _make_config(tiers=None, max_running_requests=128, static_block_size=32):
    return DllmConfig(
        algorithm="LinearSpec",
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
        # Last tier IS the catch-all (max_running_bs=9999) — anything larger
        # must still resolve to its block_size.
        tiers = [
            {"max_running_bs": 4, "block_size": 32},
            {"max_running_bs": 9999, "block_size": 8},
        ]
        cfg = _make_config(tiers=tiers, max_running_requests=128)
        self.assertEqual(cfg.select_block_size(99_999), 8)

    def test_unsorted_tiers_normalized(self):
        # Caller may pass tiers in any order; DllmConfig sorts them.
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

    def test_static_block_size_set_to_max_tier(self):
        # When tiers are configured, the canonical block_size is the max of
        # the tier block_sizes (used for KV pool worst-case allocation).
        tiers = [
            {"max_running_bs": 4, "block_size": 32},
            {"max_running_bs": 9999, "block_size": 8},
        ]
        cfg = _make_config(tiers=tiers, max_running_requests=128, static_block_size=64)
        self.assertEqual(cfg.block_size, 32)  # max(32, 8)

    def test_tiers_rejected_for_fastdiffuser(self):
        tiers = [{"max_running_bs": 9999, "block_size": 8}]
        with self.assertRaisesRegex(ValueError, "only supported with LinearSpec"):
            DllmConfig(
                algorithm="FastDiffuser",
                algorithm_config={},
                block_size=32,
                mask_id=0,
                max_running_requests=128,
                max_steps=32,
                causal_context=False,
                block_size_tiers=tiers,
            )


class TestFromServerArgs(unittest.TestCase):
    def test_max_steps_defaults_to_yaml_block_size(self):
        model_config_module = types.ModuleType("sglang.srt.configs.model_config")

        class FakeModelConfig:
            @staticmethod
            def from_server_args(*args, **kwargs):
                return SimpleNamespace(
                    hf_config=SimpleNamespace(
                        architectures=["NemotronLabsDiffusionEncoderModel"]
                    )
                )

        model_config_module.ModelConfig = FakeModelConfig

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
            with patch.dict(
                sys.modules,
                {"sglang.srt.configs.model_config": model_config_module},
            ):
                cfg = DllmConfig.from_server_args(server_args)

        self.assertEqual(cfg.block_size, 8)
        self.assertEqual(cfg.max_steps, 8)


class TestServerArgsDllmValidation(unittest.TestCase):
    def test_fa4_rejected_for_dynamic_block_tiers(self):
        from sglang.srt.server_args import ServerArgs

        with tempfile.NamedTemporaryFile("w", suffix=".yaml") as f:
            f.write(
                "block_size: 32\n"
                "block_size_tiers:\n"
                "  - {max_running_bs: 4, block_size: 32}\n"
                "  - {max_running_bs: 9999, block_size: 8}\n"
            )
            f.flush()
            server_args = ServerArgs(
                model_path="dummy",
                dllm_algorithm="LinearSpec",
                dllm_algorithm_config=f.name,
                attention_backend="fa4",
                disable_radix_cache=True,
                disable_overlap_schedule=True,
                max_running_requests=128,
            )
            model_config_module = types.ModuleType("sglang.srt.configs.model_config")

            class FakeModelConfig:
                @staticmethod
                def from_server_args(*args, **kwargs):
                    return SimpleNamespace(
                        hf_config=SimpleNamespace(
                            architectures=["NemotronLabsDiffusionEncoderModel"]
                        )
                    )

            model_config_module.ModelConfig = FakeModelConfig

            with patch.dict(
                sys.modules,
                {"sglang.srt.configs.model_config": model_config_module},
            ):
                with self.assertRaisesRegex(ValueError, "block_size_tiers"):
                    server_args._handle_dllm_inference()


if __name__ == "__main__":
    unittest.main()
