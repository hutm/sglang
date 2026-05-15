from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="stage-a-test-cpu")

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from sglang.srt.dllm.lora_utils import load_peft_lora_deltas


class TestLinearSpecLoRADeltas(unittest.TestCase):
    def test_load_lora_deltas_applies_nonzero_o_proj_delta(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            lora_dir = Path(tmpdir)
            r = 2
            alpha = 4
            lora_a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
            lora_b = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
            prefix = "base_model.model.model.layers.0.self_attn.o_proj"
            save_file(
                {
                    f"{prefix}.lora_A.weight": lora_a,
                    f"{prefix}.lora_B.weight": lora_b,
                },
                str(lora_dir / "adapter_model.safetensors"),
            )
            with open(lora_dir / "adapter_config.json", "w") as f:
                json.dump({"r": r, "lora_alpha": alpha}, f)

            o_proj = torch.nn.Linear(2, 3, bias=False)
            model = SimpleNamespace(
                model=SimpleNamespace(
                    layers=[SimpleNamespace(self_attn=SimpleNamespace(o_proj=o_proj))]
                )
            )
            deltas = load_peft_lora_deltas(str(lora_dir), model)

        self.assertEqual(len(deltas), 1)
        param, delta, module = deltas[0]
        self.assertIs(param, o_proj.weight)
        self.assertIs(module, o_proj)
        expected = (lora_b @ lora_a) * (alpha / r)
        torch.testing.assert_close(delta.cpu(), expected)


if __name__ == "__main__":
    unittest.main()
