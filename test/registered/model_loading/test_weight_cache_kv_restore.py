"""Weight cache KV restore end to end, without an orchestrator: a client-mode
server is SIGKILLed mid-decode and a new server on the same daemon serves from
the daemon's weights and the crashed server's prefix cache. Its greedy outputs
must equal the crashed server's and a disk-loaded server's."""

import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

import psutil
import requests
import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH, CustomTestCase

register_cuda_ci(est_time=400, stage="extra-b", runner_config="1-gpu-large")

MODEL = "Qwen/Qwen3-0.6B"
# Longer than several KV pages, so a restarted server has full pages to reuse.
PREFIX = (
    "You are a meticulous assistant. Read the following notes carefully. "
    + " ".join(f"Note {i}: the value of item {i} is {i * 7 % 13}." for i in range(60))
)
PROMPTS = [
    PREFIX + " Question: what is the value of item 5?",
    PREFIX + " Question: which items have the value 3?",
    PREFIX + " Summarize the notes.",
]
# The request in flight when the server is killed; it shares no prefix.
IN_FLIGHT = "Write a very long story about a lighthouse keeper and the sea. " * 20
PARAMS = {"temperature": 0, "max_new_tokens": 24}
KV_RESTORE_ENV = {"SGLANG_WEIGHT_CACHE_KV_RESTORE": "1"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _log_float(log: str, pattern: str) -> float:
    match = re.search(pattern, log)
    assert match, f"{pattern!r} not in the server log"
    return float(match.group(1))


class _Server:
    """An HTTP server in its own session, so the whole tree can be SIGKILLed."""

    def __init__(self, extra_args, env, log_dir, name):
        self.port = _free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.log_path = os.path.join(log_dir, f"{name}.log")
        self._log = open(self.log_path, "w")
        started = time.monotonic()
        self.proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                MODEL,
                "--port",
                str(self.port),
                "--nccl-port",
                str(_free_port()),
                "--mem-fraction-static",
                "0.3",
                *extra_args,
            ],
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, **env),
            start_new_session=True,
        )
        deadline = started + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        while True:
            if self.proc.poll() is not None or time.monotonic() > deadline:
                self.kill()
                raise RuntimeError(f"{name} did not start:\n{self.log()[-4000:]}")
            try:
                if requests.get(f"{self.url}/health_generate", timeout=5).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(0.2)
        self.ready_s = time.monotonic() - started

    def log(self) -> str:
        with open(self.log_path) as f:
            return f.read()

    def generate(self, prompt, params=PARAMS):
        resp = requests.post(
            f"{self.url}/generate",
            json={"text": prompt, "sampling_params": params},
            timeout=300,
        )
        resp.raise_for_status()
        return resp.json()

    def kill_mid_generation(self, prompt, after_chunks=32) -> str:
        """SIGKILL the whole tree while a long streaming request decodes;
        returns the text it streamed so far."""
        resp = requests.post(
            f"{self.url}/generate",
            json={
                "text": prompt,
                "sampling_params": {"temperature": 0, "max_new_tokens": 4096},
                "stream": True,
            },
            stream=True,
            timeout=300,
        )
        chunks, text = 0, ""
        for line in resp.iter_lines():
            if line.startswith(b"data:") and line != b"data: [DONE]":
                text = json.loads(line[len(b"data:") :])["text"]
                chunks += 1
                if chunks >= after_chunks:
                    break
        self.kill()
        resp.close()
        assert chunks >= after_chunks, f"stream ended after {chunks} chunks"
        return text

    def terminate(self, timeout_s=120):
        """Graceful shutdown: SIGTERM the server process and let it exit."""
        self.proc.send_signal(signal.SIGTERM)
        self.proc.wait(timeout=timeout_s)
        self.kill()  # reap anything left in the session

    def kill(self):
        if self._log.closed:
            return
        if self.proc.poll() is None:
            procs = [psutil.Process(self.proc.pid)]
            procs += procs[0].children(recursive=True)
            os.killpg(self.proc.pid, signal.SIGKILL)
            psutil.wait_procs(procs, timeout=60)
        else:
            try:
                os.killpg(self.proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        self._log.close()


class TestWeightCacheKvRestore(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path

        uuid = str(torch.cuda.get_device_properties(0).uuid)
        cls.socket_path = get_socket_path(uuid)
        cls.paths = [get_ready_path(uuid), cls.socket_path]
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
                "--dist-init-method",
                f"tcp://127.0.0.1:{_free_port()}",
            ],
            stdout=cls.daemon_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        deadline = time.time() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        while not os.path.exists(cls.paths[0]):
            if cls.daemon.poll() is not None or time.time() > deadline:
                cls._stop_daemon()
                raise RuntimeError("weight cache daemon did not become ready")
            time.sleep(1)
        cls.reference = None

    @classmethod
    def tearDownClass(cls):
        cls._stop_daemon()

    @classmethod
    def _stop_daemon(cls):
        if cls.daemon.poll() is None:
            os.killpg(cls.daemon.pid, signal.SIGKILL)
            cls.daemon.wait(timeout=60)
        for path in cls.paths:
            if os.path.exists(path):
                os.unlink(path)
        cls.daemon_log.seek(0)
        if os.environ.get("SGLANG_TEST_PRINT_DAEMON_LOG"):
            print(cls.daemon_log.read())
        cls.daemon_log.close()
        cls.log_dir.cleanup()

    def tearDown(self):
        # Each test starts from an empty pool.
        self._drop()

    def _drop(self) -> dict:
        from sglang.srt.weight_cache.kv_restore import drop_pool

        return drop_pool(self.socket_path)

    def _server(self, name, client=True):
        args = ["--weight-cache-mode", "client"] if client else []
        return _Server(args, KV_RESTORE_ENV, self.log_dir.name, name)

    def _disk_reference(self):
        """Greedy outputs and readiness time of a server loading from disk."""
        if type(self).reference is None:
            server = self._server("disk", client=False)
            try:
                outs = [server.generate(p)["text"] for p in PROMPTS + [IN_FLIGHT]]
            finally:
                server.kill()
            type(self).reference = (outs, server.ready_s)
        return type(self).reference

    def test_restarted_server_resumes_with_the_crashed_servers_cache(self):
        writer = self._server("writer")
        try:
            first = [writer.generate(p) for p in PROMPTS]
            time.sleep(1)  # let the last commits reach the daemon
            partial = writer.kill_mid_generation(IN_FLIGHT)
        finally:
            writer.kill()

        restarted = self._server("restarted")
        try:
            log = restarted.log()
            outs = [restarted.generate(p) for p in PROMPTS]
            # Before anything else generates IN_FLIGHT here: only the killed
            # request's prompt can have been committed, never its decode.
            resumed = restarted.generate(IN_FLIGHT + partial)
            in_flight = restarted.generate(IN_FLIGHT)
        finally:
            restarted.kill()
        reference, disk_ready_s = self._disk_reference()

        self.assertIn("[IpcModelLoader] Loaded model via IPC (mode=client)", log)
        self.assertIn("[KvRestore] attached (restore)", log)
        adopted = int(_log_float(log, r"\[KvRestore\] adopted (\d+) of"))
        adopt_ms = _log_float(log, r"\[KvRestore\] adopted \d+ of .* in ([\d.]+) ms")
        self.assertGreater(adopted, 0)
        # The first request hits the crashed server's prefix cache.
        cached = outs[0]["meta_info"]["cached_tokens"]
        self.assertGreater(cached, len(PREFIX) // 8)
        # Same greedy outputs as before the crash and as a disk load.
        texts = [o["text"] for o in outs]
        self.assertEqual(texts, [o["text"] for o in first])
        self.assertEqual(texts + [in_flight["text"]], reference)
        self.assertLessEqual(
            resumed["meta_info"]["cached_tokens"],
            in_flight["meta_info"]["prompt_tokens"],
        )
        print(
            f"[{MODEL}] restart: ready {restarted.ready_s:.1f} s vs disk ready "
            f"{disk_ready_s:.1f} s; first writer ready {writer.ready_s:.1f} s; "
            f"adopted {adopted} pages in {adopt_ms:.1f} ms; first request cached {cached} of "
            f"{outs[0]['meta_info']['prompt_tokens']} prompt tokens (before the "
            f"crash: {first[0]['meta_info']['cached_tokens']}); killed "
            f"{len(partial)} chars into a stream; resumed prompt cached "
            f"{resumed['meta_info']['cached_tokens']} of "
            f"{resumed['meta_info']['prompt_tokens']} prompt tokens"
        )

    def test_clean_shutdown_keeps_the_pool_until_it_is_dropped(self):
        from sglang.srt.weight_cache.kv_restore import pool_status

        server = self._server("before-drop")
        try:
            server.generate(PROMPTS[0])
            time.sleep(1)  # let the last commits reach the daemon
            self.assertTrue(pool_status(self.socket_path)["pool"]["leased"])
            # Single writer: the pool is not dropped under a live engine.
            self.assertEqual(self._drop(), {"status": "busy"})
            server.terminate()
        finally:
            server.kill()
        kept = pool_status(self.socket_path)["pool"]
        self.assertFalse(kept["leased"])
        self.assertGreater(kept["pages"], 0)

        free_before = torch.cuda.mem_get_info()[0]
        reply = self._drop()
        free_after = torch.cuda.mem_get_info()[0]
        self.assertEqual(reply["status"], "dropped")
        self.assertEqual(reply["pages"], kept["pages"])
        # The daemon held the only reference to the KV memory.
        self.assertGreater(free_after - free_before, 0.9 * reply["bytes"])
        self.assertIsNone(pool_status(self.socket_path)["pool"])

        server = self._server("after-drop")
        try:
            log = server.log()
            out = server.generate(PROMPTS[0])
        finally:
            server.kill()
        self.assertIn("[KvRestore] attached (fresh)", log)
        self.assertEqual(out["meta_info"]["cached_tokens"], 0)
        print(
            f"[{MODEL}] clean shutdown kept {kept['pages']} pages "
            f"({kept['bytes'] / 2**30:.2f} GiB); drop freed them and GPU free "
            f"memory rose by {(free_after - free_before) / 2**30:.2f} GiB"
        )


if __name__ == "__main__":
    unittest.main()
