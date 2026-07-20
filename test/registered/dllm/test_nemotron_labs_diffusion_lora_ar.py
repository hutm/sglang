from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=360, stage="base-b", runner_config="1-gpu-large")

import json
import shutil
import tempfile
import unittest
from pathlib import Path

import requests
from huggingface_hub import snapshot_download

from sglang.srt.utils import kill_process_tree
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

_MODEL = "nvidia/Nemotron-Labs-Diffusion-8B"
_CONFIG_DIR = "test/registered/dllm/configs"
_COMMON_SERVER_ARGS = [
    "--trust-remote-code",
    "--tp-size",
    "1",
    "--mem-fraction-static",
    "0.9",
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
_PROMPT = "In one sentence, describe what an autoregressive language model does."
_SPEED_PROMPT = (
    "Human: Explain why LoRA can improve a draft model without changing the "
    "base verifier.\n\nAssistant:"
)


def _completion(base_url, model, prompt=_PROMPT, max_tokens=96):
    response = requests.post(
        base_url + "/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=180,
    )
    if response.status_code != 200:
        raise AssertionError(response.text)
    body = response.json()
    return (
        body.get("usage", {}).get("completion_tokens", 0),
        body["choices"][0]["text"],
    )


def _assert_completion(test_case, base_url, model):
    completion_tokens, text = _completion(base_url, model)
    test_case.assertGreater(completion_tokens, 0, "model produced zero tokens")
    test_case.assertIsInstance(text, str)
    test_case.assertGreater(len(text.strip()), 0)


class TestNemotronLabsDiffusionARMode(CustomTestCase):
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
                f"{_CONFIG_DIR}/nemotron_labs_fastdiffuser.yaml",
                "--json-model-override-args",
                json.dumps({"ar_mode": True}),
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_generates_completion(self):
        _assert_completion(self, self.base_url, self.model)


class TestNemotronLabsDiffusionLinearSpecLoRA(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = _MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls._tmpdir = tempfile.mkdtemp(prefix="nemotron_labs_diffusion_lora_")
        snapshot = snapshot_download(
            repo_id=cls.model,
            allow_patterns=["linear_spec_lora/*"],
        )
        lora_path = Path(snapshot) / "linear_spec_lora"
        cls._yaml_path = Path(cls._tmpdir) / "linearspec_lora.yaml"
        cls._yaml_path.write_text(
            "algorithm: LinearSpec\n"
            "causal_context: true\n"
            f"lora_path: {lora_path}\n"
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
        if hasattr(cls, "_tmpdir") and cls._tmpdir:
            shutil.rmtree(cls._tmpdir, ignore_errors=True)

    def test_generates_completion(self):
        _assert_completion(self, self.base_url, self.model)

    def test_bs_1_speed(self):
        args = BenchArgs(
            port=int(self.base_url.split(":")[-1]),
            max_new_tokens=512,
            prompt=_SPEED_PROMPT,
        )
        _acc_length, speed = send_one_prompt(args)
        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                "### test_bs_1_speed "
                "(nemotron-labs-diffusion-8b-linearspec-lora) tp=1\n"
                f"{speed=:.2f} token/s\n"
            )
            self.assertGreater(speed, 50)


if __name__ == "__main__":
    unittest.main()
