"""Weight cache client mode with block-FP8 weights whose scales the daemon's
post-load processing re-lays out (SM100: requantized to DeepGEMM's packed
UE8M0 int32)."""

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

register_cuda_ci(est_time=600, stage="extra-b", runner_config="1-gpu-b200")

MODEL = "Qwen/Qwen3-0.6B-FP8"
PREFIX = "Read the notes. " + " ".join(
    f"Note {i}: the value of item {i} is {i * 7 % 13}." for i in range(40)
)
PROMPTS = [PREFIX + " What is the value of item 5?", PREFIX + " Summarize."]
PARAMS = {"temperature": 0, "max_new_tokens": 16}
CLIENT = ("--weight-cache-mode", "client")


def _generate(env, *extra_args):
    """Greedy outputs of a fresh server."""
    from sglang.srt.utils import kill_process_tree

    server = popen_launch_server(
        MODEL,
        DEFAULT_URL_FOR_TEST,
        timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
        other_args=["--mem-fraction-static", "0.3", *extra_args],
        env=dict(os.environ, **env),
    )
    try:
        outs = []
        for prompt in PROMPTS:
            resp = requests.post(
                f"{DEFAULT_URL_FOR_TEST}/generate",
                json={"text": prompt, "sampling_params": PARAMS},
                timeout=120,
            )
            resp.raise_for_status()
            outs.append(resp.json())
    finally:
        kill_process_tree(server.pid)
    return [o["text"] for o in outs]


class TestWeightCacheFp8Requant(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path

        cls.baseline = _generate({})  # disk load, no daemon
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

    def test_client_serves_requantized_weights(self):
        texts = _generate({}, *CLIENT)
        self.assertEqual(texts, self.baseline)
        self.daemon_log.seek(0)
        log = self.daemon_log.read()
        # The case this covers: post-load processing re-laid out tensors.
        self.assertRegex(log, r"[1-9]\d* re-laid out by post-load processing")
        self.assertNotIn("falling back to disk load", log)


if __name__ == "__main__":
    unittest.main()
