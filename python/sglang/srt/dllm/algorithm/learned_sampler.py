"""Learned acceptance sampler for diffusion LMs."""

import logging
import os
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.algorithm.learned_sampler_model import (
    _build_features_144,
    _load_sampler,
)
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

try:
    from sglang.srt.dllm.algorithm.learned_sampler_kernels import (
        fused_build_features_and_normalize,
    )

    _HAS_TRITON_FEATURES = True
except ImportError:
    _HAS_TRITON_FEATURES = False

logger = logging.getLogger(__name__)


class LearnedSampler(DllmAlgorithm):
    def __init__(self, config: DllmConfig):
        super().__init__(config)
        cfg = config.algorithm_config

        self.threshold = cfg.get("threshold", 0.97)
        self.causal_context = config.causal_context
        self._eos_token_id: Optional[int] = None

        self._stats_total_fp: int = 0
        self._stats_cuda_graph: int = 0
        self._stats_eager: int = 0
        self._stats_tokens_accepted: int = 0

        sampler_dir = cfg.get("sampler_dir")
        if sampler_dir is None:
            raise ValueError(
                "LearnedSampler requires 'sampler_dir' in algorithm config "
                "pointing to directory with checkpoint.pt"
            )
        if not os.path.isdir(sampler_dir):
            raise FileNotFoundError(f"Sampler directory not found: {sampler_dir}")

        checkpoint_path = os.path.join(sampler_dir, "checkpoint.pt")
        sem_path = os.path.join(sampler_dir, "token_semantics_32d.pt")

        if not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(f"Sampler checkpoint not found: {checkpoint_path}")

        self._sampler: Optional[nn.Module] = None
        self._feat_mean: Optional[torch.Tensor] = None
        self._feat_std: Optional[torch.Tensor] = None
        self._sem_table: Optional[torch.Tensor] = None
        self._checkpoint_path = checkpoint_path
        self._sem_path = sem_path
        self._f_dim: Optional[int] = None
        self._is_v2: bool = False
        self._v2_meta: dict = {}
        self._sampler_graphs: dict = {}

    def _ensure_loaded(self, device: torch.device):
        if self._sampler is not None:
            return

        self._sampler, self._feat_mean, self._feat_std, self._v2_meta = _load_sampler(
            self._checkpoint_path, device
        )
        self._is_v2 = self._v2_meta.get("is_v2", False)

        if self._is_v2:
            self._f_dim = None

            BL = self.block_size
            dummy_h = torch.zeros(1, BL, 0, device=device, dtype=torch.float32)
            dummy_mask = torch.ones(1, BL, device=device, dtype=torch.float32)
            dummy_ids = torch.zeros(1, BL, dtype=torch.long, device=device)
            dummy_oids = torch.zeros(1, BL, 3, dtype=torch.long, device=device)
            dummy_probs = torch.ones(1, BL, 5, device=device) / 5
            with torch.no_grad():
                _ = self._sampler(
                    dummy_h,
                    mask=dummy_mask,
                    token_ids=dummy_ids,
                    output_ids=dummy_oids,
                    top_probs=dummy_probs,
                )
        else:
            self._sem_table = torch.load(
                self._sem_path, map_location=device, weights_only=True
            ).half()
            self._f_dim = self._feat_mean.shape[0]

            dummy = torch.zeros(
                1, self.block_size, self._f_dim, device=device, dtype=torch.half
            )
            with torch.no_grad():
                _ = self._sampler(dummy)

        for bs in range(1, 5):
            try:
                self._capture_sampler_graph(bs, device)
            except Exception as e:
                logger.warning(f"CUDA graph capture failed for bs={bs}: {e}")
                break

        if self._sampler_graphs:
            logger.info(
                "Sampler CUDA graphs captured for bs=%s",
                sorted(self._sampler_graphs.keys()),
            )
        else:
            logger.warning("No sampler CUDA graphs captured, using eager mode")

    def _capture_sampler_graph(self, batch_size: int, device: torch.device):
        BL = self.block_size

        if self._is_v2:
            static_hidden = torch.zeros(
                batch_size, BL, 0, device=device, dtype=torch.float32
            )
            static_mask = torch.ones(batch_size, BL, device=device, dtype=torch.float32)
            static_token_ids = torch.zeros(
                batch_size, BL, dtype=torch.long, device=device
            )
            static_output_ids = torch.zeros(
                batch_size, BL, 3, dtype=torch.long, device=device
            )
            static_top_probs = (
                torch.ones(batch_size, BL, 5, device=device, dtype=torch.float32) / 5
            )

            def _run():
                raw = self._sampler(
                    static_hidden,
                    mask=static_mask,
                    token_ids=static_token_ids,
                    output_ids=static_output_ids,
                    top_probs=static_top_probs,
                )
                logits = raw[0] if isinstance(raw, tuple) else raw
                return torch.sigmoid(logits.float())

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    with torch.no_grad():
                        static_output = _run()
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    static_output = _run()

            static_inputs = {
                "hidden": static_hidden,
                "mask": static_mask,
                "token_ids": static_token_ids,
                "output_ids": static_output_ids,
                "top_probs": static_top_probs,
            }
            self._sampler_graphs[batch_size] = (g, static_inputs, static_output)
        else:
            static_input = torch.zeros(
                batch_size,
                BL,
                self._f_dim,
                device=device,
                dtype=torch.float16,
            )

            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    with torch.no_grad():
                        static_output = torch.sigmoid(
                            self._sampler(static_input).float()
                        )
            torch.cuda.current_stream().wait_stream(s)

            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                with torch.no_grad():
                    static_output = torch.sigmoid(self._sampler(static_input).float())

            self._sampler_graphs[batch_size] = (g, static_input, static_output)

    def _sampler_forward(self, feat: torch.Tensor) -> torch.Tensor:
        B = feat.shape[0]
        if B not in self._sampler_graphs:
            try:
                self._capture_sampler_graph(B, feat.device)
                logger.info("Sampler CUDA graph captured for bs=%d", B)
            except Exception:
                with torch.no_grad():
                    return torch.sigmoid(self._sampler(feat).float())

        graph, static_input, static_output = self._sampler_graphs[B]
        static_input.copy_(feat)
        graph.replay()
        return static_output

    def _sampler_forward_v2(
        self,
        mask: torch.Tensor,
        token_ids: torch.Tensor,
        output_ids: torch.Tensor,
        top_probs: torch.Tensor,
    ) -> torch.Tensor:
        B = token_ids.shape[0]
        device = token_ids.device

        if B not in self._sampler_graphs:
            try:
                self._capture_sampler_graph(B, device)
                logger.info("v2 sampler CUDA graph captured for bs=%d", B)
            except Exception:
                with torch.no_grad():
                    h = torch.zeros(
                        B, self.block_size, 0, device=device, dtype=torch.float32
                    )
                    raw = self._sampler(
                        h,
                        mask=mask,
                        token_ids=token_ids,
                        output_ids=output_ids,
                        top_probs=top_probs,
                    )
                    logits = raw[0] if isinstance(raw, tuple) else raw
                    return torch.sigmoid(logits.float())

        graph, static_inputs, static_output = self._sampler_graphs[B]
        static_inputs["mask"].copy_(mask)
        static_inputs["token_ids"].copy_(token_ids)
        static_inputs["output_ids"].copy_(output_ids)
        static_inputs["top_probs"].copy_(top_probs)
        graph.replay()
        return static_output

    def _get_eos_id(self, model_runner: ModelRunner) -> Optional[int]:
        if self._eos_token_id is None:
            try:
                hf_cfg = model_runner.model_config.hf_config
                eos = getattr(hf_cfg, "eos_token_id", None)
                if isinstance(eos, list):
                    eos = eos[0]
                self._eos_token_id = int(eos) if eos is not None else None
            except Exception:
                self._eos_token_id = None
        return self._eos_token_id

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        device = forward_batch.input_ids.device
        self._ensure_loaded(device)
        eos_id = self._get_eos_id(model_runner)

        mask_index = forward_batch.input_ids == self.mask_id

        if not mask_index.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        start_list = []
        for block_id in range(batch_size):
            block_start = block_id * self.block_size
            block_end = block_start + self.block_size
            block_input_ids = forward_batch.input_ids[block_start:block_end]
            block_mask = block_input_ids == self.mask_id
            start_list.append(self.block_size - block_mask.sum().item())

        eos_freeze = torch.zeros(
            batch_size, self.block_size, dtype=torch.bool, device=device
        )

        for _ in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if not mask_index.any():
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            self._stats_total_fp += 1
            if can_run_cuda_graph:
                self._stats_cuda_graph += 1
            else:
                self._stats_eager += 1

            all_logits = logits_output.full_logits
            logits_2d = all_logits.reshape(batch_size, self.block_size, -1)
            probs_2d = F.softmax(logits_2d.float(), dim=-1)
            ids_2d = forward_batch.input_ids.reshape(batch_size, self.block_size)
            mask_2d = ids_2d == self.mask_id

            active_2d = mask_2d & ~eos_freeze

            if not active_2d.any():
                break

            if self._is_v2:
                top5_vals, top5_ids = probs_2d.topk(5, dim=-1)
                scores = self._sampler_forward_v2(
                    mask=active_2d.float(),
                    token_ids=ids_2d.long(),
                    output_ids=top5_ids[..., :3].long(),
                    top_probs=top5_vals.float(),
                )

                if scores.ndim == 3:
                    cls_probs = torch.softmax(scores, dim=-1)
                    scores = 1.0 - cls_probs[..., 0]
                top1_ids = top5_ids[..., 0]
            elif _HAS_TRITON_FEATURES:
                feat, top1_ids_fused = fused_build_features_and_normalize(
                    probs_2d,
                    active_2d,
                    ids_2d,
                    self._sem_table,
                    self._feat_mean,
                    self._feat_std,
                    self.block_size,
                    self._f_dim,
                )
                scores = self._sampler_forward(feat)
                top1_ids = top1_ids_fused
            else:
                feat, _ = _build_features_144(
                    probs_2d,
                    active_2d,
                    self.block_size,
                    self._sem_table,
                    device,
                    ids_2d,
                    self._f_dim,
                )
                feat = (feat.half() - self._feat_mean) / self._feat_std.clamp(min=1e-6)
                scores = self._sampler_forward(feat)
                top1_ids = probs_2d.argmax(dim=-1)

            accept = active_2d & (scores >= self.threshold)

            for j in range(batch_size):
                if not accept[j].any() and active_2d[j].any():
                    masked_scores = torch.where(
                        active_2d[j], scores[j], torch.full_like(scores[j], -1.0)
                    )
                    accept[j, masked_scores.argmax()] = True

            self._stats_tokens_accepted += int(accept.sum().item())

            for j in range(batch_size):
                if not accept[j].any():
                    continue

                block_start = j * self.block_size
                gen_start = start_list[j]
                accepted_pos = accept[j].nonzero(as_tuple=True)[0]

                for pos in accepted_pos:
                    p = pos.item()
                    forward_batch.input_ids[block_start + p] = top1_ids[j, p]

                if eos_id is not None:
                    placed_in_gen = ~mask_2d[j].clone()
                    placed_in_gen[:gen_start] = False

                    placed_in_gen |= accept[j]
                    block_ids_now = forward_batch.input_ids[
                        block_start : block_start + self.block_size
                    ]
                    eos_placed = placed_in_gen & (block_ids_now == eos_id)
                    if eos_placed.any():
                        first_eos = int(eos_placed.nonzero(as_tuple=True)[0][0])
                        eos_freeze[j, first_eos:] = True

                        for k in range(first_eos, self.block_size):
                            if forward_batch.input_ids[block_start + k] == self.mask_id:
                                forward_batch.input_ids[block_start + k] = eos_id

        ids_2d = forward_batch.input_ids.reshape(batch_size, self.block_size)
        remaining = (ids_2d == self.mask_id) & ~eos_freeze
        if remaining.any():
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            self._stats_total_fp += 1
            self._stats_tokens_accepted += int(remaining.sum().item())
            if can_run_cuda_graph:
                self._stats_cuda_graph += 1
            else:
                self._stats_eager += 1
            all_logits = logits_output.full_logits.reshape(
                batch_size, self.block_size, -1
            )

            all_logits[:, :, self.mask_id] = -np.inf
            x0 = torch.argmax(all_logits, dim=-1)
            for j in range(batch_size):
                if remaining[j].any():
                    block_start = j * self.block_size
                    for p in remaining[j].nonzero(as_tuple=True)[0]:
                        forward_batch.input_ids[block_start + p] = x0[j, p]

        if self.causal_context:
            forward_batch.dllm_causal_kv_update = True
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        if self.causal_context:
            forward_batch.dllm_causal_kv_update = False
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

        self._stats_total_fp += 1
        if can_run_cuda_graph:
            self._stats_cuda_graph += 1
        else:
            self._stats_eager += 1

        next_token_ids = forward_batch.input_ids.reshape(batch_size, -1)
        next_token_ids_list = [
            next_token_ids[i, start_list[i] :] for i in range(batch_size)
        ]

        tok_per_fp = self._stats_tokens_accepted / max(1, self._stats_total_fp)
        logger.info(
            "LearnedSampler block done: CG=%d  eager=%d  total_fp=%d  "
            "tokens_accepted=%d  tok/fp=%.2f",
            self._stats_cuda_graph,
            self._stats_eager,
            self._stats_total_fp,
            self._stats_tokens_accepted,
            tok_per_fp,
        )

        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LearnedSampler
