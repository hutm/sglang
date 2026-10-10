"""vmm_fd weight-cache transport across processes, and mappings outliving
the exporting daemon."""

import multiprocessing as mp
import os
import socket
import tempfile
import threading
import unittest

import torch

from sglang.srt.environ import envs
from sglang.srt.weight_cache.protocol import recv_msg
from sglang.srt.weight_cache.transport import (
    VMM_FD_BACKEND,
    VmmFdTransportBackend,
    choose_daemon_transport_backend,
    get_client_transport_backend,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")


def _client(socket_path, expected, exporter_closed, result_queue):
    torch.cuda.set_device(0)
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.connect(socket_path)
        result = recv_msg(sock)
        backend = get_client_transport_backend(result["transport_backend"])
        result = backend.recv_fetch_state_response(sock, result)
        sock.close()

        def check():
            for name, entry in result["entries"].items():
                tensor = backend.import_tensor(entry)
                assert tensor.device.type == "cuda", name
                torch.testing.assert_close(tensor.cpu(), expected[name])

        check()
        result_queue.put("mapped")
        # The exporter unmaps and releases its handle; ours keeps the pages.
        exporter_closed.wait(timeout=60)
        check()
        backend.close()
        result_queue.put("ok")
    except BaseException as error:  # surface the failure to the parent
        result_queue.put(f"error: {error!r}")


class TestWeightCacheVmmFdTransport(CustomTestCase):
    def test_client_maps_weights_and_outlives_the_exporter(self):
        torch.cuda.set_device(0)
        weight = torch.randn(257, 129, device="cuda", dtype=torch.bfloat16)
        state = {
            "model.embed.weight": (weight, True),
            "lm_head.weight": (weight, True),  # tied to the embedding
            "model.norm.weight": (torch.rand(129, device="cuda"), True),
            "rotary.cos_sin_cache": (torch.arange(1000, device="cuda"), False),
        }
        expected = {name: t.cpu() for name, (t, _) in state.items()}

        with envs.SGLANG_WEIGHT_CACHE_TRANSPORT.override(VMM_FD_BACKEND):
            backend = choose_daemon_transport_backend(state)
        self.assertIsInstance(backend, VmmFdTransportBackend)
        entries = backend.prepare_export(state)
        self.assertEqual(
            entries["model.embed.weight"]["offset"],
            entries["lm_head.weight"]["offset"],
        )
        for name, entry in entries.items():
            torch.testing.assert_close(
                backend.tensor_for_entry(entry).cpu(), expected[name]
            )

        ctx = mp.get_context("spawn")
        exporter_closed = ctx.Event()
        results = ctx.Queue()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "wc.sock")
            server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            server.bind(path)
            server.listen(1)

            def serve():
                conn, _ = server.accept()
                backend.send_fetch_state_response(
                    conn, config={}, entries=entries, pid=os.getpid()
                )
                conn.close()

            thread = threading.Thread(target=serve)
            thread.start()
            client = ctx.Process(
                target=_client, args=(path, expected, exporter_closed, results)
            )
            client.start()
            try:
                self.assertEqual(results.get(timeout=120), "mapped")
                thread.join(timeout=30)
                backend.close()
                del weight, state
                torch.cuda.empty_cache()
                exporter_closed.set()
                self.assertEqual(results.get(timeout=120), "ok")
            finally:
                client.join(timeout=30)
                if client.is_alive():
                    client.kill()
                server.close()
        self.assertEqual(client.exitcode, 0)


if __name__ == "__main__":
    unittest.main()


class TestWeightCacheVmmFdServer(CustomTestCase):
    """Daemon + client server over vmm_fd; the server keeps serving after the
    daemon that exported its weights exits."""

    model = "Qwen/Qwen3-0.6B"

    @classmethod
    def setUpClass(cls):
        import subprocess
        import sys
        import time

        from sglang.srt.weight_cache.protocol import get_ready_path, get_socket_path
        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            DEFAULT_URL_FOR_TEST,
            popen_launch_server,
        )

        cls.base_url = DEFAULT_URL_FOR_TEST
        uuid = str(torch.cuda.get_device_properties(0).uuid)
        cls.paths = [get_ready_path(uuid), get_socket_path(uuid)]
        for path in cls.paths:
            if os.path.exists(path):
                os.unlink(path)
        cls.log_dir = tempfile.TemporaryDirectory()
        cls.daemon_log = open(os.path.join(cls.log_dir.name, "daemon.log"), "w+")
        env = dict(os.environ, SGLANG_WEIGHT_CACHE_TRANSPORT=VMM_FD_BACKEND)
        cls.daemon = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.srt.weight_cache.daemon",
                "--model-path",
                cls.model,
                "--tp-size",
                "1",
            ],
            env=env,
            stdout=cls.daemon_log,
            stderr=subprocess.STDOUT,
        )
        deadline = time.time() + DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH
        while not os.path.exists(cls.paths[0]):
            if cls.daemon.poll() is not None or time.time() > deadline:
                raise RuntimeError("weight cache daemon did not become ready")
            time.sleep(1)
        cls.server = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--weight-cache-mode",
                "client",
                "--mem-fraction-static",
                "0.3",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        from sglang.srt.utils import kill_process_tree

        for proc in (getattr(cls, "server", None), getattr(cls, "daemon", None)):
            if proc is not None and proc.poll() is None:
                kill_process_tree(proc.pid)
        for path in cls.paths:
            if os.path.exists(path):
                os.unlink(path)
        cls.daemon_log.close()
        cls.log_dir.cleanup()

    def _complete(self):
        import requests

        resp = requests.post(
            f"{self.base_url}/v1/completions",
            json={
                "model": self.model,
                "prompt": "The capital of France is",
                "max_tokens": 16,
                "temperature": 0,
            },
            timeout=60,
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["choices"][0]["text"]

    def test_server_keeps_serving_after_daemon_exit(self):
        from sglang.srt.utils import kill_process_tree

        self.daemon_log.flush()
        self.daemon_log.seek(0)
        self.assertIn("transport=vmm_fd", self.daemon_log.read())
        before = self._complete()
        self.assertTrue(before)

        kill_process_tree(self.daemon.pid)
        self.daemon.wait(timeout=60)

        self.assertEqual(self._complete(), before)
        self.assertIsNone(self.server.poll())
