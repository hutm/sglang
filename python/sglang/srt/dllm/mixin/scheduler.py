from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Set, Union

import torch

from sglang.srt.dllm.config import DllmConfig
from sglang.srt.dllm.mixin.req import DllmReqPhase
from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder
from sglang.srt.mem_cache.common import release_kv_cache
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.req_time_stats import set_time_batch
from sglang.srt.observability.scheduler_metrics_mixin import PrefillStats
from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.scheduler import GenerationBatchResult, Scheduler


class SchedulerDllmMixin:
    @staticmethod
    def _truncate_dllm_tokens_for_finish(req: Req, token_ids: List[int]) -> List[int]:
        remaining = req.sampling_params.max_new_tokens - len(req.output_ids)
        if remaining <= 0:
            return []

        token_ids = token_ids[:remaining]
        if req.sampling_params.ignore_eos:
            return token_ids

        stop_ids: Set[int] = set(req.sampling_params.stop_token_ids or [])
        stop_ids.update(req.eos_token_ids or [])

        tokenizer = req.tokenizer
        if tokenizer is not None:
            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            if eos_token_id is not None:
                stop_ids.add(eos_token_id)
            stop_ids.update(getattr(tokenizer, "additional_stop_token_ids", []) or [])

        for i, token_id in enumerate(token_ids):
            if token_id in stop_ids:
                return token_ids[: i + 1]

        return token_ids

    def init_diffusion_llm(self: Scheduler):
        self.dllm_config = (
            DllmConfig.from_server_args(self.server_args)
            if self.server_args.dllm_algorithm is not None
            else None
        )
        self.dllm_manager = DllmManager(dllm_config=self.dllm_config)
        # Pinned-host + device tensor buffers reused across DLLM_EXTEND
        # blocks (and SamplingBatchInfo cached for stable batch composition).
        # The ScheduleBatch object itself is NOT reused: its reqs list is
        # mutated by the scheduler's merge logic between iterations
        # (DLLM_EXTEND.is_extend() == True triggers
        # last_batch.filter_batch(...)). Buffers live here so they survive.
        self._dllm_block_buffers: Optional[dict] = None
        self._dllm_cached_sampling_info = None
        self._dllm_cached_sampling_info_reqs_id: Optional[int] = None
        # Optional optimization: run K extra DLLM_EXTEND blocks per
        # event-loop iteration, reusing the same batch object and bypassing
        # the scheduler's recv / new-batch-build / merge code per inner
        # block. Falls back to K=1 behavior on any state change (new
        # request arrives, a req finishes, batch_size != 1, etc.) so
        # multi-request workloads are unaffected.
        #
        # Default K=1 (no inner loop). Latency-bound single-request
        # workloads can set inner_k_blocks=4 in the YAML to amortize the
        # outside-run_batch scheduler overhead across K inner blocks.
        # Empirically K=4 saturates the amortization win; higher K shows
        # diminishing or slightly negative returns from EOS overshoot on
        # short responses.
        self._dllm_inner_k_blocks = (
            self.dllm_config.algorithm_config.get("inner_k_blocks", 1)
            if self.dllm_config is not None
            else 1
        )
        # Counter for throttled per-iter prefill stats reporting in
        # process_batch_result_dllm. Initialized to 0 here so the first
        # iteration starts a fresh window of 20 blocks.
        self._dllm_stats_skip_count = 0
        # Tier-policy histogram (block_size_tiers); populated only when tiers
        # are configured. Pre-initialized so the hot path doesn't branch on
        # hasattr each block.
        self._dllm_tier_hist: dict[int, int] = {}
        self._dllm_tier_blocks: int = 0
        self._dllm_pending_tier_bs: Optional[int] = None
        # Fine-grained DLLM scheduler profiling
        self._dllm_sched_prof = (
            self.dllm_config is not None
            and self.dllm_config.algorithm_config.get("profile", False)
        )
        if self._dllm_sched_prof:
            import time

            self._dllm_time = time
            self._dsp_n = 0
            # process_batch_result_dllm sub-timings
            self._dsp_sync = 0.0
            self._dsp_tolist = 0.0
            self._dsp_kv_free = 0.0
            self._dsp_stream = 0.0
            self._dsp_stats = 0.0
            # get_new_batch_dllm sub-timings
            self._dsp_prepare = 0.0
            self._dsp_process = 0.0
            self._dsp_create = 0.0
            # _create_dllm_batch sub-timings
            self._dsp_init_new = 0.0
            self._dsp_prep_extend = 0.0
            self._dsp_stats_build = 0.0
            # SamplingBatchInfo cache reuse counters
            self._dsp_a2_sampling_hits = 0
            self._dsp_a2_sampling_misses = 0

    def get_new_batch_dllm(self: Scheduler) -> Optional[ScheduleBatch]:
        """Build the next DLLM batch (EXTEND for prompt caching or DLLM_EXTEND for denoising)."""
        _prof = self._dllm_sched_prof
        if _prof:
            _gt0 = self._dllm_time.perf_counter()

        if self.enable_priority_scheduling:
            self.running_batch.batch_is_full = False

        if self._should_skip_prefill():
            return None

        running_bs = len(self.running_batch.reqs)
        self.policy.calc_priority(self.waiting_queue)

        adder = self._create_dllm_prefill_adder(running_bs)

        self._prepare_staging_reqs()
        self._fetch_waiting_reqs()

        if _prof:
            _gt1 = self._dllm_time.perf_counter()

        forward_mode = self._process_dllm_batches(adder)

        can_run_list = adder.can_run_list
        if not can_run_list:
            return None

        set_time_batch(can_run_list, "set_forward_entry_time")
        self._update_state_for_batch(can_run_list, adder, running_bs)

        if _prof:
            _gt2 = self._dllm_time.perf_counter()

        batch = self._create_dllm_batch(can_run_list, forward_mode)

        # Propagate the tier-selected block_size to the schedule batch so
        # ForwardBatch.init_new picks the same tier_bs for positions /
        # dllm_block_offsets. Per-req dllm_active_block_size is set earlier
        # (in _prepare_staging_reqs / process_dllm_incoming_reqs) so that
        # init_next_round_input builds fill_ids with the same tier_bs.
        # Use the SAME stamped value to keep input_ids and positions length
        # consistent — recomputing from len(batch.reqs) here can pick a
        # different tier than reqs were initialized with if running_bs
        # estimate diverged from final can_run_list size.
        pending_tier_bs = self._dllm_pending_tier_bs
        if (
            getattr(self.dllm_config, "block_size_tiers", None)
            and batch is not None
            and batch.reqs
            and pending_tier_bs is not None
        ):
            batch.dllm_block_size = pending_tier_bs

        # Tier-policy histogram: log the distribution of block_size choices
        # across blocks so we can validate the policy against actual workload.
        if getattr(self.dllm_config, "block_size_tiers", None):
            picked = self.dllm_config.select_block_size(running_bs)
            self._dllm_tier_hist[picked] = self._dllm_tier_hist.get(picked, 0) + 1
            self._dllm_tier_blocks += 1
            if self._dllm_tier_blocks % 200 == 0:
                tot = self._dllm_tier_blocks
                breakdown = ", ".join(
                    f"bs={k}: {v} ({v / tot * 100:.1f}%)"
                    for k, v in sorted(self._dllm_tier_hist.items())
                )
                logger.info(
                    "DLLM TIER POLICY (%d blocks): %s [last running_bs=%d -> bs=%d]",
                    tot,
                    breakdown,
                    running_bs,
                    picked,
                )

        if _prof:
            _gt3 = self._dllm_time.perf_counter()
            self._dsp_prepare += _gt1 - _gt0
            self._dsp_process += _gt2 - _gt1
            self._dsp_create += _gt3 - _gt2
            self._dsp_n += 1
            if self._dsp_n % 500 == 0:
                n = self._dsp_n
                a2_total = self._dsp_a2_sampling_hits + self._dsp_a2_sampling_misses
                a2_hit_rate = (
                    self._dsp_a2_sampling_hits / a2_total * 100 if a2_total > 0 else 0
                )
                logger.info(
                    "DLLM SCHED PROFILE (%d iters): "
                    "process_result[sync=%.2f tolist+kv=%.2f stream=%.2f stats=%.2f]ms  "
                    "get_batch[prepare=%.2f process=%.2f create=%.2f"
                    " (init_new=%.3f prep_extend=%.3f stats_build=%.3f)]ms  "
                    "A2_sampling_cache[hits=%d misses=%d rate=%.1f%%]",
                    n,
                    self._dsp_sync / n * 1000,
                    self._dsp_tolist / n * 1000,
                    self._dsp_stream / n * 1000,
                    self._dsp_stats / n * 1000,
                    self._dsp_prepare / n * 1000,
                    self._dsp_process / n * 1000,
                    self._dsp_create / n * 1000,
                    self._dsp_init_new / n * 1000,
                    self._dsp_prep_extend / n * 1000,
                    self._dsp_stats_build / n * 1000,
                    self._dsp_a2_sampling_hits,
                    self._dsp_a2_sampling_misses,
                    a2_hit_rate,
                )

        return batch

    def process_batch_result_dllm(
        self: Scheduler,
        batch: ScheduleBatch,
        result: GenerationBatchResult,
    ):
        _prof = self._dllm_sched_prof
        if _prof:
            _pt0 = self._dllm_time.perf_counter()

        if result.copy_done is not None:
            result.copy_done.synchronize()

        if _prof:
            _pt1 = self._dllm_time.perf_counter()

        if not batch.forward_mode.is_dllm_extend():
            for req in batch.reqs:
                req.fill_ids = []
            self.report_prefill_stats(
                batch=batch,
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=False,
                dp_cooperation_info=batch.dp_cooperation_info,
            )
            return

        if result.next_token_ids:
            self.token_to_kv_pool_allocator.free_group_begin()

            for idx in range(batch.batch_size()):
                req = batch.reqs[idx]

                next_token_ids = result.next_token_ids[idx].tolist()
                next_token_ids = self._truncate_dllm_tokens_for_finish(
                    req, next_token_ids
                )
                new_tokens = len(next_token_ids)
                # Block size used for this block (dynamic-tier or static).
                block_bs = (
                    req.dllm_active_block_size
                    if req.dllm_active_block_size is not None
                    else self.dllm_config.block_size
                )
                if new_tokens == 0:
                    # Free the entire allocated block to prevent kv_committed_len
                    # inflation. Without this, cache_finished_req frees only
                    # len(origin_input_ids + output_ids) positions which is less
                    # than kv_committed_len, permanently leaking block_size tokens.
                    rejected = block_bs
                    free_start = req.kv_committed_len - rejected
                    free_end = req.kv_committed_len
                    free_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, free_start:free_end
                    ]
                    self.token_to_kv_pool_allocator.free(free_indices)
                    req.kv_committed_len = free_start
                    req.kv_allocated_len = free_start
                    continue

                req.fill_ids[-new_tokens:] = next_token_ids[:]
                self.num_generated_tokens += new_tokens

                req.output_ids.extend(next_token_ids)

                if new_tokens < block_bs:
                    rejected = block_bs - new_tokens
                    free_start = req.kv_committed_len - rejected
                    free_end = req.kv_committed_len
                    free_indices = self.req_to_token_pool.req_to_token[
                        req.req_pool_idx, free_start:free_end
                    ]
                    self.token_to_kv_pool_allocator.free(free_indices)
                    req.kv_committed_len = free_start
                    req.kv_allocated_len = free_start

                req.check_finished_stop_before_length(new_accepted_len=new_tokens)

                if req.finished():
                    release_kv_cache(req, self.tree_cache, is_insert=False)
                    req.time_stats.set_completion_time()

            if _prof:
                _pt2 = self._dllm_time.perf_counter()

            # Skip stream_output entirely when no req is finished and no
            # req has stream=True. Non-streaming completions only need their
            # output sent on completion; streaming clients still trigger it
            # every block (gated by stream_interval inside stream_output).
            need_stream = any(
                r.finished() or getattr(r, "stream", False) for r in batch.reqs
            )
            if need_stream:
                self.stream_output(batch.reqs, batch.return_logprob)

            if _prof:
                _pt3 = self._dllm_time.perf_counter()

            self.token_to_kv_pool_allocator.free_group_end()

        can_run_cuda_graph = getattr(result, "can_run_cuda_graph", False)
        should_report_stats = True
        # Throttle logging-only prefill stats to every N=20 DLLM_EXTEND blocks.
        # report_prefill_stats also updates metrics, so keep the every-block
        # path when metrics collection is enabled.
        if (
            batch.forward_mode.is_dllm_extend()
            and not self.current_scheduler_metrics_enabled
        ):
            self._dllm_stats_skip_count += 1
            should_report_stats = self._dllm_stats_skip_count >= 20
            if should_report_stats:
                self._dllm_stats_skip_count = 0

        if should_report_stats:
            self.report_prefill_stats(
                batch=batch,
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=can_run_cuda_graph,
                dp_cooperation_info=batch.dp_cooperation_info,
            )

        if _prof and batch.forward_mode.is_dllm_extend():
            _pt4 = self._dllm_time.perf_counter()
            self._dsp_sync += _pt1 - _pt0
            self._dsp_tolist += _pt2 - _pt1
            self._dsp_stream += _pt3 - _pt2
            self._dsp_stats += _pt4 - _pt3

    def maybe_run_dllm_inner_loop(self: Scheduler, batch: ScheduleBatch) -> None:
        inner_k = getattr(self, "_dllm_inner_k_blocks", 1)
        if (
            inner_k > 1
            and self.dllm_config is not None
            and batch.forward_mode == ForwardMode.DLLM_EXTEND
            and batch.batch_size() == 1
            and not batch.reqs[0].finished()
            and not self.waiting_queue
            and len(self.dllm_manager.waiting_queue) == 1
        ):
            self._dllm_run_inner_k_blocks(batch, inner_k - 1)

    def _dllm_run_inner_k_blocks(self: Scheduler, batch: ScheduleBatch, k: int) -> None:
        """Run up to k additional DLLM_EXTEND blocks reusing the same batch
        object, bypassing the scheduler's recv / new-batch-build /
        last_batch-merge code for each inner block.

        Caller must guarantee (validated below):
        - batch.forward_mode == DLLM_EXTEND
        - batch.batch_size() == 1
        - the single req is not finished
        - no new requests are pending
        - no other reqs are in the dllm waiting queue

        Falls out early if the req finishes during the inner loop.
        """
        if not batch.forward_mode.is_dllm_extend() or batch.batch_size() != 1:
            raise RuntimeError(
                "_dllm_run_inner_k_blocks called with invalid batch state: "
                f"forward_mode={batch.forward_mode}, "
                f"batch_size={batch.batch_size()}"
            )
        req = batch.reqs[0]
        for _ in range(k):
            # Re-prepare the running req for the next block (mirror of the
            # work _prepare_staging_reqs does, but for a single in-flight
            # req — no queue dance).
            self._set_active_block_size_for(batch)
            req.init_next_round_input()
            if req.req_pool_idx is not None and req.kv_committed_len > 0:
                kv_len = req.kv_committed_len
                req.prefix_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :kv_len
                ].to(torch.int64)
                req.determine_dllm_phase()
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))

            # Refresh the per-block tensor state on the existing batch
            # using the fast path (reuses pinned host + device buffers).
            batch.prepare_for_dllm_block_extend(buffers=self._dllm_block_buffers)
            self._dllm_block_buffers = getattr(batch, "_dllm_block_buffers", None)
            batch.forward_mode = ForwardMode.DLLM_EXTEND
            batch.decoding_reqs = None

            result = self.run_batch(batch)
            self.process_batch_result(batch, result)

            if req.finished():
                break

    def _set_active_block_size_for(self: Scheduler, batch: ScheduleBatch) -> None:
        """Compute the tier-policy block_size for the current running batch and
        propagate it to every req that's about to build a new block.

        No-op when block_size_tiers are not configured.
        """
        if not getattr(self.dllm_config, "block_size_tiers", None):
            return
        running_bs = max(1, batch.batch_size())
        bs = self.dllm_config.select_block_size(running_bs)
        batch.dllm_block_size = bs
        for req in batch.reqs:
            req.dllm_active_block_size = bs

    def _prepare_staging_reqs(self: Scheduler) -> None:
        """Rebuild fill_ids and set prefix_indices for the next scheduling round.

        For each staged request, append a fresh block of mask tokens to fill_ids
        and set prefix_indices from the committed KV in req_to_token_pool so the
        next DLLM_EXTEND forward can attend to the full previously-denoised prefix.

        Note: the model writes correct KV for every position during DLLM_EXTEND
        (save_kv_cache=True in nemotron_labs_dllm.py), so no separate KV-update EXTEND
        pass is needed between blocks.
        """
        # Apply tier-policy block_size to ALL decode-phase DLLM reqs that
        # will participate in the next batch. The next batch combines:
        #   - STAGING_DECODE reqs in dllm_manager.waiting_queue
        #     (in-flight reqs from prior DLLM blocks)
        #   - INCOMING_DECODE reqs in dllm_manager.waiting_queue
        #     (newly-prefilled reqs ready for first DLLM block)
        # These two collectively form get_decode_requests(), and tier_bs
        # MUST match across all of them so prepare_for_dllm_block_extend
        # (called in _create_dllm_batch) builds input_ids with a uniform
        # mask-tail length matching the positions/dllm_block_offsets that
        # ForwardBatch.init_new produces from batch.dllm_block_size.
        # Updating only staging_queue (the prior version of this fix)
        # missed STAGING_DECODE reqs that didn't run a block in the
        # immediately-preceding iteration (e.g. a prefill ran instead).
        if getattr(self.dllm_config, "block_size_tiers", None):
            # Reqs that will participate in this iter's DLLM batch:
            #   - waiting_queue reqs already in a DECODE phase
            #   - staging_queue reqs (these include just-prefilled reqs that
            #     will transition to STAGING_DECODE via init_next_round_input
            #     below).
            staging_set = set(id(r) for r in self.dllm_manager.staging_queue)
            decode_reqs = [
                r
                for r in self.dllm_manager.waiting_queue
                if getattr(r, "dllm_phase", None)
                in (DllmReqPhase.STAGING_DECODE, DllmReqPhase.INCOMING_DECODE)
                or id(r) in staging_set
            ]
            running_bs = max(1, len(decode_reqs))
            tier_bs = self.dllm_config.select_block_size(running_bs)
            self._dllm_pending_tier_bs = tier_bs
            # Stamp tier_bs on every participating req BEFORE
            # init_next_round_input (in the loop below or in
            # process_dllm_incoming_reqs) so fill_ids get a uniform mask
            # tail length matching batch.dllm_block_size.
            for req in decode_reqs:
                old_bs = req.dllm_active_block_size
                req.dllm_active_block_size = tier_bs
                # Rebuild fill_ids inline for STAGING_DECODE reqs that are
                # NOT in staging_queue (they wouldn't otherwise be re-init'd
                # this iter — staging_queue reqs are rebuilt below).
                if (
                    id(req) not in staging_set
                    and req.dllm_phase == DllmReqPhase.STAGING_DECODE
                    and old_bs != tier_bs
                ):
                    req.init_next_round_input()
                    if req.req_pool_idx is not None and req.kv_committed_len > 0:
                        kv_len = req.kv_committed_len
                        req.prefix_indices = self.req_to_token_pool.req_to_token[
                            req.req_pool_idx, :kv_len
                        ].to(torch.int64)
                        req.determine_dllm_phase()
                        req.set_extend_input_len(
                            len(req.fill_ids) - len(req.prefix_indices)
                        )
        for req in self.dllm_manager.staging_queue:
            req.init_next_round_input()
            if req.req_pool_idx is not None and req.kv_committed_len > 0:
                kv_len = req.kv_committed_len
                # Convert to int64: req_to_token_pool uses int32, but the Triton
                # write_req_to_token_pool_triton kernel casts prefix_tensor to
                # int64* — reading int32 data as int64 corrupts prefix positions.
                req.prefix_indices = self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, :kv_len
                ].to(torch.int64)
                req.determine_dllm_phase()
                req.set_extend_input_len(len(req.fill_ids) - len(req.prefix_indices))
        self.dllm_manager.staging_queue = []

    def _fetch_waiting_reqs(self: Scheduler):
        max_dllm_capacity = self.dllm_config.max_running_requests - len(
            self.dllm_manager.waiting_queue
        )
        num_requests_to_add = min(max_dllm_capacity, len(self.waiting_queue))
        if num_requests_to_add > 0:
            requests_to_add = self.waiting_queue[:num_requests_to_add]
            self.dllm_manager.add_waiting_reqs(requests_to_add)
            self.waiting_queue = self.waiting_queue[num_requests_to_add:]

    def _should_skip_prefill(self: Scheduler) -> bool:
        if (
            self.running_batch.batch_is_full or not self.waiting_queue
        ) and self.dllm_manager.is_empty():
            return True
        running_bs = len(self.running_batch.reqs)
        if (
            self.get_num_allocatable_reqs(running_bs) <= 0
            and self.dllm_manager.is_empty()
            and not self.enable_priority_scheduling
        ):
            self.running_batch.batch_is_full = True
            return True
        return False

    def _create_dllm_prefill_adder(self: Scheduler, running_bs: int) -> PrefillAdder:
        return PrefillAdder(
            self.page_size,
            self.tree_cache,
            self.token_to_kv_pool_allocator,
            self.running_batch,
            self.new_token_ratio,
            self.max_prefill_tokens,
            self.chunked_prefill_size,
            running_bs if self.is_mixed_chunk else 0,
            self.priority_scheduling_preemption_threshold,
            prefill_max_requests=self.server_args.prefill_max_requests,
            dllm_config=self.dllm_config,
        )

    def _process_dllm_batches(self: Scheduler, adder: PrefillAdder) -> ForwardMode:
        """Decide batch type and populate adder.

        Priority:
          1. INCOMING_PREFILL requests → causal EXTEND to cache prompt KV.
          2. STAGING_DECODE / INCOMING_DECODE requests → DLLM_EXTEND denoising.
        """
        incoming_prefill = [
            req
            for req in self.dllm_manager.waiting_queue
            if req.dllm_phase == DllmReqPhase.INCOMING_PREFILL
        ]
        if incoming_prefill:
            self._process_incoming_prefill_reqs(adder, incoming_prefill)
            return ForwardMode.EXTEND

        # Try prefill batch (STAGING_PREFILL only)
        prefill_reqs = self.dllm_manager.get_prefill_requests()
        if prefill_reqs:
            self._process_batch_by_phase(
                adder,
                prefill_reqs,
                DllmReqPhase.STAGING_PREFILL,
                DllmReqPhase.INCOMING_PREFILL,
            )
        else:
            decode_reqs = self.dllm_manager.get_decode_requests()
            self._process_batch_by_phase(
                adder,
                decode_reqs,
                DllmReqPhase.STAGING_DECODE,
                DllmReqPhase.INCOMING_DECODE,
            )

        return ForwardMode.DLLM_EXTEND

    def _process_incoming_prefill_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> None:
        """Schedule INCOMING_PREFILL requests as a causal EXTEND (prompt caching)."""
        for req in reqs:
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_scheduling
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            req.init_prompt_cache_input()
            # Ensure last_node is set so dec_lock_ref is safe.
            if req.last_node is None and hasattr(self.tree_cache, "root_node"):
                req.last_node = self.tree_cache.root_node
            res = adder.add_dllm_prompt_cache_req(req)

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

    def _process_batch_by_phase(
        self,
        adder: PrefillAdder,
        batch: List[Req],
        staging_phase: DllmReqPhase,
        incoming_phase: DllmReqPhase,
    ) -> None:
        staging_reqs = [req for req in batch if req.dllm_phase == staging_phase]
        if staging_reqs:
            result = self.process_dllm_staging_reqs(adder, staging_reqs)
            if result != AddReqResult.CONTINUE:
                return

        incoming_reqs = [req for req in batch if req.dllm_phase == incoming_phase]
        if incoming_reqs:
            self.process_dllm_incoming_reqs(adder, incoming_reqs)

    def _update_state_for_batch(
        self: Scheduler, can_run_list: List[Req], adder: PrefillAdder, running_bs: int
    ) -> None:
        if adder.preempt_list:
            for req in adder.preempt_list:
                self._add_request_to_queue(req)

        if can_run_list:
            self.dllm_manager.add_staging_reqs(can_run_list)
            self.dllm_manager.increment_chunked_count()

        self.adder = adder
        self.can_run_list = can_run_list
        self.running_bs = len(self.running_batch.reqs)

    def _create_dllm_batch(
        self: Scheduler, can_run_list: List[Req], forward_mode: ForwardMode
    ) -> ScheduleBatch:
        _prof = self._dllm_sched_prof
        if _prof:
            _ct0 = self._dllm_time.perf_counter()

        new_batch = ScheduleBatch.init_new(
            can_run_list,
            self.req_to_token_pool,
            self.token_to_kv_pool_allocator,
            self.tree_cache,
            self.model_config,
            self.enable_overlap,
            self.spec_algorithm,
            dllm_config=self.dllm_config,
        )

        if _prof:
            _ct1 = self._dllm_time.perf_counter()

        # Fast path (A2): for DLLM_EXTEND blocks, use the trimmed
        # prepare_for_dllm_block_extend that skips multimodal / embeds /
        # mamba / logprob / MIS branches (unused in the DLLM_EXTEND fast path)
        # and reuses pinned host + device buffers across blocks.
        # SamplingBatchInfo is also cached: it depends only on req sampling
        # params which don't change between blocks of the same request.
        if forward_mode == ForwardMode.DLLM_EXTEND:
            new_batch.prepare_for_dllm_block_extend(
                buffers=self._dllm_block_buffers,
            )
            # Save the (possibly newly allocated) buffer dict back so the
            # next block can reuse it.
            self._dllm_block_buffers = getattr(new_batch, "_dllm_block_buffers", None)

            # Reuse SamplingBatchInfo when reqs are identity-stable; rebuild
            # otherwise. id() of the reqs list is unique per call but list
            # contents are identity-stable for stable batches.
            reqs_signature = tuple(id(r) for r in can_run_list)
            if (
                self._dllm_cached_sampling_info is not None
                and self._dllm_cached_sampling_info_reqs_id == reqs_signature
            ):
                new_batch.sampling_info = self._dllm_cached_sampling_info
                penalizer_orchestrator = new_batch.sampling_info.penalizer_orchestrator
                if penalizer_orchestrator is not None:
                    penalizer_orchestrator.batch = new_batch
                if _prof:
                    self._dsp_a2_sampling_hits += 1
            else:
                new_batch.sampling_info = SamplingBatchInfo.from_schedule_batch(
                    new_batch,
                    self.model_config.vocab_size,
                )
                self._dllm_cached_sampling_info = new_batch.sampling_info
                self._dllm_cached_sampling_info_reqs_id = reqs_signature
                if _prof:
                    self._dsp_a2_sampling_misses += 1
        else:
            new_batch.prepare_for_extend()

        new_batch.forward_mode = forward_mode
        new_batch.decoding_reqs = None

        if _prof:
            _ct2 = self._dllm_time.perf_counter()

        new_batch.prefill_stats = PrefillStats.from_adder(
            self.adder,
            self.running_batch.reqs,
            getattr(self, "enable_priority_scheduling", False),
        )

        if _prof:
            _ct3 = self._dllm_time.perf_counter()
            self._dsp_init_new += _ct1 - _ct0
            self._dsp_prep_extend += _ct2 - _ct1
            self._dsp_stats_build += _ct3 - _ct2

        return new_batch

    def process_dllm_incoming_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        res = AddReqResult.CONTINUE
        # Tier policy: incoming-decode reqs must use the same tier_bs
        # computed in _prepare_staging_reqs so the entire batch shares one
        # block_size. Without this, init_next_round_input below builds
        # fill_ids with the static block_size, causing positions/input_ids
        # length mismatch in ForwardBatch.init_new.
        pending_tier_bs = self._dllm_pending_tier_bs
        for req in reqs:
            running_bs = len(self.running_batch.reqs)
            if len(adder.can_run_list) >= self.get_num_allocatable_reqs(running_bs):
                self.running_batch.batch_is_full = True

            if self.running_batch.batch_is_full:
                if (
                    not self.enable_priority_scheduling
                    or not adder.preempt_to_schedule(req, self.server_args)
                ):
                    break

            if pending_tier_bs is not None:
                req.dllm_active_block_size = pending_tier_bs
            req.init_next_round_input(self.tree_cache)
            res = adder.add_one_req(
                req,
                has_chunked_req=True,
                truncation_align_size=self.truncation_align_size,
            )

            if res != AddReqResult.CONTINUE:
                if res == AddReqResult.NO_TOKEN:
                    self.running_batch.batch_is_full = True
                break

        return res

    def process_dllm_staging_reqs(
        self: Scheduler, adder: PrefillAdder, reqs: List[Req]
    ) -> AddReqResult:
        for req in reqs:
            res = adder.add_dllm_staging_req(req)
            if res == AddReqResult.NO_TOKEN:
                return res
        return AddReqResult.CONTINUE


class DllmManager:
    """Manages DLLM request queues.

    waiting_queue: all active DLLM requests (persists across rounds).
    staging_queue: requests scheduled this round (cleared after each round).
    """

    def __init__(self, dllm_config: Optional[DllmConfig] = None):
        self.dllm_config = dllm_config
        self.max_running_reqs = (
            dllm_config.max_running_requests if dllm_config is not None else 1
        )
        self.waiting_queue: List[Req] = []
        self.staging_queue: List[Req] = []

    def get_prefill_requests(self) -> List[Req]:
        return [req for req in self.waiting_queue if req.is_dllm_prefill()]

    def get_decode_requests(self) -> List[Req]:
        return [req for req in self.waiting_queue if not req.is_dllm_prefill()]

    def add_waiting_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        assert self.dllm_config is not None
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        if self._has_duplicate_reqs(reqs_to_add):
            raise RuntimeError("Redundant requests detected in dLLM requests.")
        self.waiting_queue.extend(reqs_to_add)

    def add_staging_reqs(self, reqs: Union[Req, List[Req]]) -> None:
        reqs_to_add = reqs if isinstance(reqs, list) else [reqs]
        self.staging_queue.extend(reqs_to_add)

    def _has_duplicate_reqs(self, reqs: List[Req]) -> bool:
        existing_rids: Set[str] = {r.rid for r in self.waiting_queue}
        return any(req.rid in existing_rids for req in reqs)

    def any_staging_reqs(self) -> bool:
        return self.dllm_config is not None and len(self.staging_queue) > 0

    def is_empty(self) -> bool:
        if self.dllm_config is None:
            return True
        return len(self.waiting_queue) == 0

    def increment_chunked_count(self) -> None:
        for req in self.staging_queue:
            req.is_chunked += 1

    def filter_finished_reqs(self) -> None:
        self.waiting_queue = [req for req in self.waiting_queue if not req.finished()]
        self.staging_queue = [req for req in self.staging_queue if not req.finished()]
