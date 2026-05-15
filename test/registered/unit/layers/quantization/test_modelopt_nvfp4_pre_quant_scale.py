import unittest
from unittest.mock import patch

import torch
from torch.nn.parameter import Parameter

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-test-cpu")

_KNOWN_OPTIONAL_IMPORT_ERRORS = (
    "No module named 'flashinfer'",
    "cannot import name 'mm_mxfp8'",
)

try:
    from sglang.srt.layers.quantization.modelopt_quant import ModelOptFp4LinearMethod
except ImportError as exc:
    if not any(marker in str(exc) for marker in _KNOWN_OPTIONAL_IMPORT_ERRORS):
        raise
    ModelOptFp4LinearMethod = None
    MODEL_OPT_IMPORT_ERROR = exc
else:
    MODEL_OPT_IMPORT_ERROR = None


class _TpGroup:
    world_size = 2
    rank_in_group = 1


@unittest.skipIf(
    ModelOptFp4LinearMethod is None,
    f"modelopt_quant is unavailable: {MODEL_OPT_IMPORT_ERROR}",
)
class TestModelOptNvfp4PreQuantScale(CustomTestCase):
    def test_loader_shards_and_pads_with_identity(self):
        param = Parameter(torch.ones(3, dtype=torch.float32), requires_grad=False)
        loaded_weight = torch.tensor([1, 2, 3, 4, 5], dtype=torch.float32)

        with patch(
            "sglang.srt.layers.quantization.modelopt_quant.get_tp_group",
            return_value=_TpGroup(),
        ):
            ModelOptFp4LinearMethod._load_pre_quant_scale(param, loaded_weight)

        torch.testing.assert_close(
            param.data, torch.tensor([4, 5, 1], dtype=torch.float32)
        )

    def test_loader_rejects_conflicting_fused_shards(self):
        param = Parameter(torch.ones(3, dtype=torch.float32), requires_grad=False)

        ModelOptFp4LinearMethod._load_pre_quant_scale(
            param, torch.tensor([1, 2, 3], dtype=torch.float32), loaded_shard_id="q"
        )
        with self.assertRaisesRegex(ValueError, "identical pre_quant_scale"):
            ModelOptFp4LinearMethod._load_pre_quant_scale(
                param,
                torch.tensor([1, 2, 4], dtype=torch.float32),
                loaded_shard_id="k",
            )


if __name__ == "__main__":
    unittest.main()
