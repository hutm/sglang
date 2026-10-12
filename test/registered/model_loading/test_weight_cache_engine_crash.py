"""Weight cache end to end without an orchestrator: the server process tree is
SIGKILLed mid-generation, and a new one (a restarted writer, or a warm standby
promoted through the Engine API) serves from the daemon's weights and KV cache.
Restarted outputs must equal the crashed server's and a disk-loaded server's,
and the crashed server's in-flight pages must not be adopted."""

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

register_cuda_ci(est_time=1500, stage="extra-b", runner_config="1-gpu-b200")

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
KV_RESTORE_ENV = {
    "SGLANG_WEIGHT_CACHE_TRANSPORT": "vmm_fd",
    "SGLANG_WEIGHT_CACHE_KV_RESTORE": "1",
}


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

    def __init__(self, model, extra_args, env, log_dir, name):
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
                model,
                "--port",
                str(self.port),
                "--nccl-port",
                str(_free_port()),
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
        log = self.log()
        self.load_s = _log_float(log, r"Load weight end\. elapsed=([\d.]+) s")
        timings = re.search(r"Engine startup timings \(s\): (.*)", log)
        self.timings = timings.group(1) if timings else "n/a"

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

    def kill(self):
        if self.proc.poll() is None:
            procs = [psutil.Process(self.proc.pid)]
            procs += procs[0].children(recursive=True)
            os.killpg(self.proc.pid, signal.SIGKILL)
            psutil.wait_procs(procs, timeout=60)
        self._log.close()


class _EngineCrashMixin:
    MODEL: str
    MEM_FRACTION: str

    @classmethod
    def setUpClass(cls):
        from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path

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
                cls.MODEL,
                "--tp-size",
                "1",
                "--dist-init-method",
                f"tcp://127.0.0.1:{_free_port()}",
            ],
            stdout=cls.daemon_log,
            stderr=subprocess.STDOUT,
            env=dict(os.environ, **KV_RESTORE_ENV),
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

    def _server(self, name, client=True, env=None):
        args = ["--mem-fraction-static", self.MEM_FRACTION]
        if client:
            args += ["--weight-cache-mode", "client"]
        return _Server(
            self.MODEL,
            args,
            dict(KV_RESTORE_ENV, **(env or {})),
            self.log_dir.name,
            name,
        )

    def _disk_reference(self):
        """Greedy outputs and startup times of a server loading from disk;
        started after the daemon runs, so its JIT and page caches are warm."""
        if type(self).reference is None:
            server = self._server("disk", client=False)
            try:
                outs = [server.generate(p)["text"] for p in PROMPTS + [IN_FLIGHT]]
            finally:
                server.kill()
            type(self).reference = (outs, server.ready_s, server.load_s)
        return type(self).reference

    def test_restarted_server_reuses_weights_and_kv_cache(self):
        writer = self._server("writer")
        try:
            first = [writer.generate(p)["text"] for p in PROMPTS]
            time.sleep(1)  # let the last commits reach the daemon
            partial = writer.kill_mid_generation(IN_FLIGHT)
        finally:
            writer.kill()

        restarted = self._server("restarted")
        try:
            log = restarted.log()
            outs = [restarted.generate(p) for p in PROMPTS]
            # Before anything else generates IN_FLIGHT here: only the killed
            # request's prompt was committed, so nothing it decoded can hit.
            resumed = restarted.generate(IN_FLIGHT + partial)
            in_flight = restarted.generate(IN_FLIGHT)
        finally:
            restarted.kill()
        reference, disk_ready_s, disk_load_s = self._disk_reference()

        # (a) Weights come from the daemon. Only the weight load is skipped;
        # imports and CUDA graph capture still run in the new process.
        self.assertIn("[IpcModelLoader] Loaded model via IPC (mode=client)", log)
        self.assertLess(restarted.load_s * 5, disk_load_s)
        # (b) The first request hits the crashed server's prefix cache.
        cached = outs[0]["meta_info"]["cached_tokens"]
        self.assertGreater(cached, len(PREFIX) // 8)
        # (c) Same greedy outputs as before the crash and as a disk load.
        self.assertEqual([o["text"] for o in outs], first)
        self.assertEqual([o["text"] for o in outs] + [in_flight["text"]], reference)
        # (d) The killed writer's uncommitted pages (its decode tail) were
        # quarantined, not imported into the radix tree.
        quarantined = int(
            _log_float(log, r"\[KvRestore\] adopted \d+ pages; quarantined (\d+)")
        )
        self.assertGreater(quarantined, 0)
        self.assertLessEqual(
            resumed["meta_info"]["cached_tokens"],
            in_flight["meta_info"]["prompt_tokens"],
        )
        print(
            f"[{self.MODEL}] restart: ready {restarted.ready_s:.1f} s (weights "
            f"{restarted.load_s:.2f} s; {restarted.timings}) vs disk ready "
            f"{disk_ready_s:.1f} s (weights {disk_load_s:.2f} s); first writer ready "
            f"{writer.ready_s:.1f} s; first request cached {cached} tokens; "
            f"killed {len(partial)} chars into a stream, {quarantined} pages "
            f"quarantined, resumed prompt cached "
            f"{resumed['meta_info']['cached_tokens']} of "
            f"{in_flight['meta_info']['prompt_tokens']} prompt tokens"
        )

    def test_standby_promoted_after_the_writer_is_killed(self):
        import sglang as sgl

        writer = self._server("writer-for-standby")
        standby = None
        try:
            first = [writer.generate(p)["text"] for p in PROMPTS]
            time.sleep(1)  # let the last commits reach the daemon

            env = dict(KV_RESTORE_ENV, SGLANG_WEIGHT_CACHE_KV_RESTORE_ROLE="standby")
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            free_before = torch.cuda.mem_get_info()[0]
            started = time.monotonic()
            try:
                standby = sgl.Engine(
                    model_path=self.MODEL,
                    weight_cache_mode="client",
                    mem_fraction_static=float(self.MEM_FRACTION),
                )
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k)
                    else:
                        os.environ[k] = v
            standby_ready_s = time.monotonic() - started
            standby_bytes = free_before - torch.cuda.mem_get_info()[0]

            partial = writer.kill_mid_generation(IN_FLIGHT)
            started = time.monotonic()
            adopted = standby.kv_restore_promote(10, quarantine_grace_s=0.5)
            promote_s = time.monotonic() - started
            outs = [standby.generate(p, PARAMS) for p in PROMPTS]
            resumed = standby.generate(IN_FLIGHT + partial, PARAMS)
            in_flight = standby.generate(IN_FLIGHT, PARAMS)
        finally:
            writer.kill()
            if standby is not None:
                standby.shutdown()
        reference, disk_ready_s, _ = self._disk_reference()

        # Weights and KV pool are the daemon's: the standby adds far less
        # than its static budget (context, activations and graphs only).
        self.assertLess(standby_bytes, 0.1 * torch.cuda.mem_get_info()[1])
        self.assertGreater(adopted, 0)
        self.assertLess(promote_s, 5.0)
        cached = outs[0]["meta_info"]["cached_tokens"]
        self.assertGreater(cached, len(PREFIX) // 8)
        self.assertEqual([o["text"] for o in outs], first)
        self.assertEqual([o["text"] for o in outs] + [in_flight["text"]], reference)
        # Nothing the killed writer decoded is reused.
        self.assertLessEqual(
            resumed["meta_info"]["cached_tokens"],
            in_flight["meta_info"]["prompt_tokens"],
        )
        print(
            f"[{self.MODEL}] standby: added {standby_bytes / 2**30:.2f} GiB, "
            f"ready {standby_ready_s:.1f} s vs disk "
            f"ready {disk_ready_s:.1f} s; promote after SIGKILL "
            f"{promote_s * 1e3:.0f} ms adopted {adopted} pages; first request "
            f"cached {cached} tokens; resumed prompt cached "
            f"{resumed['meta_info']['cached_tokens']} of "
            f"{in_flight['meta_info']['prompt_tokens']} prompt tokens"
        )


class TestEngineCrashDense(_EngineCrashMixin, CustomTestCase):
    MODEL = "Qwen/Qwen3-0.6B"
    MEM_FRACTION = "0.3"


class TestEngineCrashFp8Moe(_EngineCrashMixin, CustomTestCase):
    # Block-FP8 MoE: on SM100 post-load processing re-lays out scales and sets
    # module attributes the client must get from the daemon.
    MODEL = "Qwen/Qwen3-30B-A3B-FP8"
    MEM_FRACTION = "0.4"


if __name__ == "__main__":
    unittest.main()
