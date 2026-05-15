"""AR-mode and LoRA coverage for the public Nemotron-Labs-Diffusion 3B DLLM.

PR1 onboards the Nemotron-Labs-Diffusion architecture and covers the two
diffusion-mode algorithms (FastDiffuser, LinearSpec) on the public 3B
checkpoint. This file adds the two features that PR2 introduces on the
same model:

  * **AR mode** — ``ar_mode=true`` on the HF config flips every attention
    layer to causal, turning the model into a plain autoregressive
    generator. We exercise it through ``--json-model-override-args`` and
    run it via the FastDiffuser scheduler to make sure the bidirectional/
    causal split inside the algorithm doesn't blow up when the model is
    fully causal.

  * **LinearSpec + LoRA** — a synthetic LoRA adapter targeting only
    ``o_proj`` (the same projection the real upstream adapter will
    target) with all-zero ``A``/``B`` matrices is generated in
    ``setUpClass``. Because both matrices are zero, the effective delta
    ``B @ A`` is exactly zero, so the adapter is functionally identity;
    that lets us exercise the LoRA load + dual-weight bake + draft-only
    swap code path — including the ``o_proj``-only loader branch — until
    a real trained adapter is released.

Once a real LoRA is released, swap the synthetic adapter for the real
``lora_path`` and tighten the keyword check into an accuracy assertion.
"""

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=240, suite="stage-b-test-1-gpu-small")

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import requests
import torch
from safetensors.torch import save_file
from transformers import AutoConfig

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

_MODEL = "MMaghoumi/Nemotron-Labs-Diffusion-TinyStories-3b"
_CONFIG_DIR = "test/registered/dllm/configs"

# Gingerbread story prompt, mirroring the diffusion-mode test so the
# coherence assertion is the same shape: with greedy decoding the
# TinyStories model should drop an on-topic keyword early.
_STORY_PROMPT = "Once upon a time, there was a little gingerbread man who jumped out of the oven and"
_STORY_KEYWORDS = ("gingerbread", "ran", "fox")

_COMMON_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "1",
    "--mem-fraction-static",
    "0.85",
    "--max-running-requests",
    "4",
    "--attention-backend",
    "flashinfer",
    "--cuda-graph-bs",
    "1",
    "2",
    "3",
    "4",
    "--context-length",
    "4096",
]


def _post_completion(base_url, model, prompt=_STORY_PROMPT, max_tokens=128):
    response = requests.post(
        base_url + "/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=120,
    )
    if response.status_code != 200:
        raise AssertionError(response.text)
    body = response.json()
    return (
        body.get("usage", {}).get("completion_tokens", 0),
        body["choices"][0]["text"],
    )


def _build_zero_o_proj_lora_adapter(
    out_dir: Path, base_model_id: str, r: int = 8, alpha: int = 16
):
    """Write a PEFT-format LoRA adapter targeting only ``o_proj``, all zeros.

    The planned upstream LoRA only applies to ``o_proj``, so we match that
    shape exactly — including the fact that ``LinearSpec._load_lora_deltas``
    has a separate (and skippable) code path for the q/k/v fused branch.
    Targeting only ``o_proj`` here exercises:

      * the no-qkv branch in the loader (qkv-fused delta application is
        intentionally skipped),
      * the dedicated ``o_proj`` delta path and dual-weight bake,
      * the draft-only weight swap during LinearSpec inference.

    With both ``A`` and ``B`` zero the effective delta ``scale * (B @ A)``
    is zero, so the adapter is functionally identity. That keeps this a
    plumbing test until a real trained adapter is released; swap in the
    real ``lora_path`` and tighten the keyword check into an accuracy
    assertion when it lands.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = AutoConfig.from_pretrained(base_model_id, trust_remote_code=True)
    hidden = cfg.hidden_size
    q_dim = cfg.num_attention_heads * cfg.head_dim

    tensors = {}
    for layer_idx in range(cfg.num_hidden_layers):
        prefix = f"base_model.model.model.layers.{layer_idx}.self_attn"
        # A: [r, q_dim], B: [hidden, r]
        tensors[f"{prefix}.o_proj.lora_A.weight"] = torch.zeros(
            r, q_dim, dtype=torch.float32
        )
        tensors[f"{prefix}.o_proj.lora_B.weight"] = torch.zeros(
            hidden, r, dtype=torch.float32
        )

    save_file(tensors, str(out_dir / "adapter_model.safetensors"))
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(
            {
                "peft_type": "LORA",
                "task_type": "CAUSAL_LM",
                "r": r,
                "lora_alpha": alpha,
                "lora_dropout": 0.0,
                "bias": "none",
                "fan_in_fan_out": False,
                "target_modules": ["o_proj"],
                "base_model_name_or_path": base_model_id,
                "init_lora_weights": True,
            },
            f,
            indent=2,
        )


class TestNemotronLabsDiffusion3BARMode(CustomTestCase):
    """AR mode (``ar_mode=true`` on the model config) on the 3B public DLLM.

    The diffusion model is forced into fully-causal attention via
    ``--json-model-override-args``. We run it through the FastDiffuser
    scheduler — this verifies the scheduler does not deadlock or produce
    garbage when the underlying layers are causal.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_SERVER_ARGS
            + [
                "--dllm-algorithm",
                "FastDiffuser",
                "--dllm-algorithm-config",
                f"{_CONFIG_DIR}/nemotron_labs_diffusion_3b_fastdiffuser.yaml",
                "--json-model-override-args",
                json.dumps({"ar_mode": True}),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generates_story_continuation(self):
        completion_tokens, text = _post_completion(self.base_url, self.model)
        self.assertGreater(completion_tokens, 0, "model produced zero tokens")
        self.assertIsInstance(text, str)
        lowered = text.lower()
        self.assertTrue(
            any(kw in lowered for kw in _STORY_KEYWORDS),
            f"AR-mode output missing all on-topic keywords {_STORY_KEYWORDS!r}: {text!r}",
        )


class TestNemotronLabsDiffusion3BLinearSpecZeroLoRA(CustomTestCase):
    """LinearSpec + synthetic zero LoRA on the 3B public DLLM.

    Until a real LoRA adapter is released, we generate a PEFT-format
    adapter with all-zero ``lora_A`` and ``lora_B`` matrices in
    ``setUpClass``. The effective delta is then zero everywhere, so the
    adapter is functionally identity — the test exercises the LoRA load,
    dual-weight bake, and draft-only swap code paths and asserts that
    output is still coherent (not garbled by a broken delta application).
    """

    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls._lora_dir = tempfile.mkdtemp(prefix="nemotron_labs_diffusion_3b_zero_lora_")
        _build_zero_o_proj_lora_adapter(Path(cls._lora_dir), cls.model)
        # Per-fixture YAML — the LoRA path varies per run, so write it
        # alongside the adapter rather than committing to the repo.
        cls._yaml_path = Path(cls._lora_dir) / "linearspec_zero_lora.yaml"
        cls._yaml_path.write_text(
            "algorithm: LinearSpec\n"
            "causal_context: true\n"
            f"lora_path: {cls._lora_dir}\n"
        )
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=_COMMON_SERVER_ARGS
            + [
                "--dllm-algorithm",
                "LinearSpec",
                "--dllm-algorithm-config",
                str(cls._yaml_path),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        if hasattr(cls, "_lora_dir") and cls._lora_dir:
            shutil.rmtree(cls._lora_dir, ignore_errors=True)

    def test_generates_story_continuation_with_zero_lora(self):
        completion_tokens, text = _post_completion(self.base_url, self.model)
        self.assertGreater(completion_tokens, 0, "model produced zero tokens")
        self.assertIsInstance(text, str)
        lowered = text.lower()
        self.assertTrue(
            any(kw in lowered for kw in _STORY_KEYWORDS),
            f"zero-LoRA output missing all on-topic keywords {_STORY_KEYWORDS!r}: {text!r}",
        )


if __name__ == "__main__":
    unittest.main()
