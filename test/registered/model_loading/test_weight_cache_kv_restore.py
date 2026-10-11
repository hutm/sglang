"""Weight cache KV restore end to end: a server SIGKILLed and restarted against
the same daemon resumes with its radix cache instead of a cold one."""

import os
import subprocess
import sys
import tempfile
import time
import unittest

import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="extra-b", runner_config="1-gpu-h100")

MODEL = "Qwen/Qwen3-0.6B"
PREFIX = (
    "You are a meticulous assistant. Read the following notes carefully. "
    + " ".join(f"Note {i}: the value of item {i} is {i * 7 % 13}." for i in range(60))
)
PROMPTS = [PREFIX + " Question: what is the value of item 5?", PREFIX + " Summarize."]


class TestWeightCacheKvRestore(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path

        cls.base_url = DEFAULT_URL_FOR_TEST
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        cls.paths = [get_ready_path(uuid), get_socket_path(uuid)]
        for path in cls.paths:
            if os.path.exists(path):
                os.unlink(path)
        cls.log_dir = tempfile.TemporaryDirectory()
        cls.daemon_log = open(os.path.join(cls.log_dir.name, "daemon.log"), "w+")
        cls.daemon = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                MODEL,
                "--tp-size",
                "1",
            ],
            stdout=cls.daemon_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        while not os.path.exists(cls.paths[0]):
            if cls.daemon.poll() is not None or time.time() > deadline:
                raise RuntimeError("weight cache daemon did not become ready")
            time.sleep(1)

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.utils import kill_process_tree

        if cls.daemon.poll() is None:
            kill_process_tree(cls.daemon.pid)
        for path in cls.paths:
            if os.path.exists(path):
                os.unlink(path)
        cls.daemon_log.close()
        cls.log_dir.cleanup()

    def _launch(self, restore: bool):
        env = dict(os.environ, SGLANG_WEIGHT_CACHE_KV_RESTORE="1" if restore else "0")
        return popen_launch_server(
            MODEL,
            self.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--weight-cache-mode",
                "client",
                "--mem-fraction-static",
                "0.3",
            ],
            env=env,
        )

    def _generate(self, prompt):
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 16},
            },
            timeout=120,
        )
        self.assertEqual(resp.status_code, 200)
        out = resp.json()
        return out["text"], out["meta_info"]["cached_tokens"]

    def test_restarted_server_resumes_with_its_prefix_cache(self):
        from sglang.srt.utils import kill_process_tree

        server = self._launch(restore=True)
        try:
            first = [self._generate(p) for p in PROMPTS]
            time.sleep(1)  # let the last commits reach the daemon
        finally:
            kill_process_tree(server.pid)  # SIGKILL; the daemon stays up

        server = self._launch(restore=True)
        try:
            restored = [self._generate(p) for p in PROMPTS]
        finally:
            kill_process_tree(server.pid)

        server = self._launch(restore=False)
        try:
            cold = [self._generate(p) for p in PROMPTS]
        finally:
            kill_process_tree(server.pid)

        # Cold engines share only what the first prompt left behind.
        self.assertEqual(first[0][1], 0)
        self.assertEqual(cold[0][1], 0)
        # The restarted engine hits the previous engine's cache from request one.
        self.assertGreater(restored[0][1], len(PREFIX) // 8)
        for (text_a, _), (text_b, _), (text_c, _) in zip(first, restored, cold):
            self.assertEqual(text_a, text_b)
            self.assertEqual(text_a, text_c)
        print(
            f"cached tokens on the first request: cold={first[0][1]} "
            f"restored={restored[0][1]} restore_off={cold[0][1]}"
        )


if __name__ == "__main__":
    unittest.main()
