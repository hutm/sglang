"""Flattened-bucket weight sync through a CUDA VMM buffer passed by fd."""

import multiprocessing
import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="extra-b", runner_config="1-gpu-h100")


def _named_tensors():
    g = torch.Generator(device="cuda").manual_seed(0)
    return [
        ("a.weight", torch.randn(64, 48, device="cuda", generator=g)),
        ("b.weight", torch.randn(34, device="cuda", generator=g).to(torch.bfloat16)),
        ("c.bias", torch.arange(7, device="cuda", dtype=torch.int32)),
    ]


def _trainer(payload_q, done_q):
    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket
    from sglang.srt.weight_sync.vmm_fd import SharedFlattenedTensor

    torch.cuda.set_device(0)
    bucket = FlattenedTensorBucket(named_tensors=_named_tensors())
    shared = SharedFlattenedTensor.from_tensor(bucket.get_flattened_tensor())
    payload_q.put(
        MultiprocessingSerializer.serialize(
            {"flattened_tensor": shared, "metadata": bucket.get_metadata()},
            output_str=True,
        )
    )
    assert done_q.get(timeout=120) == "mapped"
    shared.close()  # the server keeps its own reference
    done_q.put("closed")


def _server(payload_q, done_q, results_q):
    import gc

    from sglang.srt.utils import MultiprocessingSerializer
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorBucket

    torch.cuda.set_device(0)
    bucket_dict = MultiprocessingSerializer.deserialize(payload_q.get(timeout=120))
    tensors = dict(
        FlattenedTensorBucket(
            flattened_tensor=bucket_dict["flattened_tensor"],
            metadata=bucket_dict["metadata"],
        ).reconstruct_tensors()
    )
    # Views must keep the mapping alive after the base tensor is dropped.
    del bucket_dict
    gc.collect()
    done_q.put("mapped")
    assert done_q.get(timeout=120) == "closed"
    expected = dict(_named_tensors())
    results_q.put(all(torch.equal(tensors[k], expected[k]) for k in expected))


class TestWeightSyncVmmFd(CustomTestCase):
    def test_bucket_between_sibling_processes(self):
        # Siblings: PyTorch's own VMM sharing (pidfd_getfd) is refused here
        # under the default ptrace policy; the fd handoff is not.
        ctx = multiprocessing.get_context("spawn")
        payload_q, done_q, results_q = ctx.Queue(), ctx.Queue(), ctx.Queue()
        trainer = ctx.Process(target=_trainer, args=(payload_q, done_q))
        server = ctx.Process(target=_server, args=(payload_q, done_q, results_q))
        trainer.start()
        server.start()
        self.assertTrue(results_q.get(timeout=300))
        trainer.join(timeout=60)
        server.join(timeout=60)
        self.assertEqual(trainer.exitcode, 0)
        self.assertEqual(server.exitcode, 0)


if __name__ == "__main__":
    unittest.main()
