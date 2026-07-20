"""LinearSpec draft/verify decoding for diffusion LMs.

Implements the linear self-speculation path described in the
Nemotron-Labs-Diffusion technical report, Section 3.3.
"""

import json as _json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.lora_utils import load_peft_lora_deltas
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class LinearSpec(DllmAlgorithm):
    """Draft with bidirectional attention and verify causally."""

    def __init__(self, config: DllmConfig) -> None:
        super().__init__(config)
        cfg = config.algorithm_config
        self.causal_context: bool = config.causal_context
        self._seed_tokens: Dict[str, int] = {}
        self._eos_token_id: Optional[int] = None

        self._stats_file: Optional[str] = cfg.get("stats_file", None)
        self._stats_forward_passes: int = 0

        self._lora_path: Optional[str] = cfg.get("lora_path", None)
        self._lora_mode: str = cfg.get("lora_mode", "draft_only")
        if self._lora_mode not in ("draft_only", "both"):
            raise ValueError(
                "LinearSpec lora_mode must be 'draft_only' or 'both', "
                f"got {self._lora_mode!r}."
            )
        self._lora_deltas: Optional[
            List[Tuple[torch.nn.Parameter, torch.Tensor, torch.nn.Module]]
        ] = None
        self._dual_weights: Optional[
            List[Tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]]
        ] = None
        self._graphs_baked: bool = False

        logger.info(
            "LinearSpec: block_size=%d  causal_context=%s  lora=%s  lora_mode=%s",
            self.block_size,
            self.causal_context,
            self._lora_path or "none",
            self._lora_mode,
        )

    def _get_decode_graph_runner(self, model_runner: ModelRunner):
        return getattr(model_runner, "decode_cuda_graph_runner", None) or getattr(
            model_runner, "graph_runner", None
        )

    def setup(self, model_runner: ModelRunner) -> None:
        """Load optional LoRA weights before graph capture."""
        if self._lora_path is None:
            return

        self._load_lora_deltas(model_runner)
        graph_runner = self._get_decode_graph_runner(model_runner)
        if graph_runner is None:
            return

        if not self._lora_deltas:
            logger.warning(
                "LinearSpec LoRA path %s did not produce any supported deltas; "
                "continuing with base weights.",
                self._lora_path,
            )
            self._graphs_baked = True
            self._lora_deltas = None
            return

        if self._lora_mode == "both":
            for param, delta, _mod in self._lora_deltas:
                param.data.add_(delta)
            lora_mb = (
                sum(d.numel() * d.element_size() for _, d, _ in self._lora_deltas) / 1e6
            )
            logger.info(
                "LinearSpec: applied %.1f MB LoRA in-place to base weights "
                "(permanent, both-mode)",
                lora_mb,
            )
            if model_runner.server_args.defer_cuda_graph_capture:
                logger.info(
                    "LinearSpec: capturing CUDA graphs with permanent in-place LoRA..."
                )
                graph_runner.init_capture()
                logger.info("LinearSpec: CUDA graph capture done (in-place LoRA)")
            self._graphs_baked = True
            self._lora_deltas = None
            return

        self._bake_dual_weights_into_graphs(model_runner)

    def _bake_dual_weights_into_graphs(self, model_runner: ModelRunner) -> None:
        """Prepare base and draft weight views for LoRA-backed CUDA graphs."""
        dual: List[Tuple[torch.nn.Parameter, torch.Tensor, torch.Tensor]] = []

        first_param = self._lora_deltas[0][0]
        is_fp8 = first_param.data.dtype in (
            torch.float8_e4m3fn,
            getattr(torch, "float8_e4m3fnuz", torch.float8_e4m3fn),
        )

        if is_fp8:
            from sglang.kernels.ops.quantization.fp8_kernel import (
                per_token_group_quant_fp8,
            )

            logger.info("LinearSpec: FP8 mode - dequant+requant for draft weights")

        for param, delta, module in self._lora_deltas:
            base_ref = param.data
            if is_fp8:
                scale_param = module.weight_scale
                qw_orig = base_ref.t().float()
                sc_orig = scale_param.data.t().float()
                base_bf16 = (qw_orig * sc_orig).to(torch.bfloat16)
                draft_bf16 = base_bf16 + delta.to(base_bf16.device, torch.bfloat16)
                K = draft_bf16.shape[-1]
                draft_fp8, draft_sc = per_token_group_quant_fp8(draft_bf16, K)
                draft_fp8_stored = draft_fp8.t()
                draft_sc_stored = draft_sc.t().contiguous()
                dual.append((param, base_ref, draft_fp8_stored))
                dual.append((scale_param, scale_param.data, draft_sc_stored))
            else:
                draft_copy = base_ref + delta
                dual.append((param, base_ref, draft_copy))

        self._dual_weights = dual

        total_mb = sum(d.numel() * d.element_size() for _, _, d in dual) / 1e6
        logger.info(
            "LinearSpec dual weights built from LoRA: %d params, %.1f MB extra",
            len(dual),
            total_mb,
        )

        def set_lora():
            for p, _, d in self._dual_weights:
                p.data = d

        def set_base():
            for p, b, _ in self._dual_weights:
                p.data = b

        gr = self._get_decode_graph_runner(model_runner)
        if gr is None:
            return
        gr._dllm_pre_draft_hook = set_lora
        gr._dllm_pre_verify_hook = set_lora if self._lora_mode == "both" else set_base

        if model_runner.server_args.defer_cuda_graph_capture:
            logger.info("LinearSpec: capturing CUDA graphs with baked LoRA weights")
            gr.init_capture()
            self._graphs_baked = True
            self._lora_deltas = None

    def _load_lora_deltas(self, model_runner: ModelRunner) -> None:
        """Load PEFT LoRA deltas once."""
        if (
            self._lora_deltas is not None
            or self._lora_path is None
            or self._graphs_baked
        ):
            return
        deltas = load_peft_lora_deltas(self._lora_path, model_runner.model)
        self._lora_deltas = deltas
        total_mb = sum(d.numel() * d.element_size() for _, d, _ in deltas) / 1e6
        logger.info(
            "LinearSpec LoRA loaded: %d delta tensors, %.1f MB", len(deltas), total_mb
        )

    def _get_eos_id(self, model_runner: ModelRunner) -> Optional[int]:
        if self._eos_token_id is None:
            try:
                hf_cfg = model_runner.model_config.hf_config
                eos = getattr(hf_cfg, "eos_token_id", None)
                if isinstance(eos, list):
                    eos = eos[0]
                self._eos_token_id = int(eos) if eos is not None else None
            except (AttributeError, TypeError, ValueError, IndexError):
                self._eos_token_id = None
        return self._eos_token_id

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
        algo_states=None,
    ):
        batch_size = forward_batch.batch_size
        bs = forward_batch.dllm_block_size
        if bs is None:
            bs = self.block_size
        eos_id = self._get_eos_id(model_runner)

        mask_index_all = forward_batch.input_ids == self.mask_id
        if not mask_index_all.any():
            if forward_batch.input_ids.numel() == 0:
                empty_logits = LogitsProcessorOutput(
                    next_token_logits=None, full_logits=None
                )
                return empty_logits, [], None, None, False
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output = out.logits_output
            if logits_output.next_token_logits is not None:
                rids = forward_batch.rids
                seed_logits_no_mask = logits_output.next_token_logits.clone()
                seed_logits_no_mask[:, self.mask_id] = -np.inf
                seeds = torch.argmax(seed_logits_no_mask, dim=-1)
                for b_idx in range(len(rids)):
                    self._seed_tokens[rids[b_idx]] = int(seeds[b_idx].item())
            return logits_output, [], None, None, out.can_run_graph

        start_list: List[int] = []
        for b in range(batch_size):
            b_start = b * bs
            b_end = b_start + bs
            n_masks = int(
                (forward_batch.input_ids[b_start:b_end] == self.mask_id).sum()
            )
            start_list.append(bs - n_masks)

        rids = forward_batch.rids[:batch_size]
        has_seed = [False] * batch_size
        for b in range(batch_size):
            rid = rids[b]
            if rid in self._seed_tokens:
                seed = self._seed_tokens[rid]
                block_start = b * bs + start_list[b]
                if block_start < forward_batch.input_ids.numel():
                    forward_batch.input_ids[block_start] = seed
                    has_seed[b] = True

        self._load_lora_deltas(model_runner)

        need_swap = self._lora_deltas is not None and not self._graphs_baked
        if need_swap:
            for param, delta, _mod in self._lora_deltas:
                param.data.add_(delta)

        graph_runner = getattr(model_runner, "graph_runner", None)
        can_use_graph = (
            graph_runner is not None
            and graph_runner.is_dllm
            and graph_runner.can_run_graph(forward_batch)
        )
        if can_use_graph:
            draft_output = graph_runner.execute(forward_batch)
            draft_logits = draft_output.full_logits
            can_run_graph = True
        else:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            can_run_graph = out.can_run_graph
            draft_logits = out.logits_output.full_logits

        if need_swap and self._lora_mode == "draft_only":
            for param, delta, _mod in self._lora_deltas:
                param.data.sub_(delta)
        self._stats_forward_passes += 1

        draft_logits[:, self.mask_id] = -1e9
        draft_all = torch.argmax(draft_logits, dim=-1)

        forward_batch.input_ids.copy_(draft_all)
        for b in range(batch_size):
            if has_seed[b]:
                forward_batch.input_ids[b * bs + start_list[b]] = self._seed_tokens[
                    rids[b]
                ]

        forward_batch.dllm_causal_kv_update = self.causal_context
        can_verify_with_graph = (
            can_run_graph
            and graph_runner is not None
            and graph_runner.can_run_graph(forward_batch)
        )
        if can_verify_with_graph:
            try:
                logits_output = graph_runner.execute(forward_batch)
            finally:
                forward_batch.dllm_causal_kv_update = False
            verify_logits = logits_output.full_logits
            can_run_graph = True
            if need_swap and self._lora_mode == "both":
                for param, delta, _mod in self._lora_deltas:
                    param.data.sub_(delta)
        else:
            try:
                out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            finally:
                forward_batch.dllm_causal_kv_update = False
            if need_swap and self._lora_mode == "both":
                for param, delta, _mod in self._lora_deltas:
                    param.data.sub_(delta)
            verify_logits = out.logits_output.full_logits
            logits_output = out.logits_output
            can_run_graph = out.can_run_graph
        self._stats_forward_passes += 1

        verify_logits[:, self.mask_id] = -1e9
        ar_all = torch.argmax(verify_logits, dim=-1)

        next_token_ids_list: List[torch.Tensor] = []
        accepted_counts: List[int] = []

        for b in range(batch_size):
            b_start = b * bs
            gen_start = start_list[b]
            gen_len = bs - gen_start

            if gen_len > 1:
                offset = b_start + gen_start
                draft_block = forward_batch.input_ids[offset + 1 : offset + gen_len]
                ar_prev = ar_all[offset : offset + gen_len - 1]
                matches = draft_block == ar_prev
                c = int(matches.cumprod(0).sum().item())
            else:
                c = 0

            accepted = c + 1

            seed_pos = b_start + gen_start
            seed_tensor = forward_batch.input_ids[seed_pos : seed_pos + 1]
            if c > 0:
                ar_slice = ar_all[seed_pos : seed_pos + c]
                output_tokens = torch.cat([seed_tensor, ar_slice])
            else:
                output_tokens = seed_tensor

            if eos_id is not None:
                eos_mask = output_tokens == eos_id
                if eos_mask.any():
                    eos_pos = int(eos_mask.to(torch.int32).argmax().item()) + 1
                    output_tokens = output_tokens[:eos_pos]
                    accepted = eos_pos

            next_token_ids_list.append(output_tokens)
            accepted_counts.append(accepted)

            rid = rids[b]
            ar_c_pos = b_start + gen_start + c
            if ar_c_pos < b_start + bs:
                self._seed_tokens[rid] = int(ar_all[ar_c_pos].item())
            else:
                self._seed_tokens[rid] = int(ar_all[b_start + bs - 1].item())

        for b in range(batch_size):
            tokens = next_token_ids_list[b]
            if (
                eos_id is not None
                and len(tokens) > 0
                and int(tokens[-1].item()) == eos_id
            ):
                self._seed_tokens.pop(rids[b], None)

        total_accepted = sum(accepted_counts)
        avg_accepted = total_accepted / max(batch_size, 1)

        if self._stats_file:
            with open(self._stats_file, "a") as _sf:
                for b in range(batch_size):
                    tokens = int(next_token_ids_list[b].shape[0])
                    gen_len = bs - start_list[b]
                    _sf.write(
                        _json.dumps(
                            {
                                "forward_passes": 2,
                                "tokens": tokens,
                                "block_gen_positions": gen_len,
                                "acceptance_rate": (
                                    tokens / gen_len if gen_len > 0 else 0
                                ),
                            }
                        )
                        + "\n"
                    )

        logger.debug(
            "LinearSpec block done: fp=%d  avg_accepted=%.1f/%d  total_tokens=%d",
            self._stats_forward_passes,
            avg_accepted,
            bs,
            total_accepted,
        )

        return logits_output, next_token_ids_list, None, None, can_run_graph


Algorithm = LinearSpec
