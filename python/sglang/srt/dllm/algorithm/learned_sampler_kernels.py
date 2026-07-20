"""Triton feature extraction kernels for LearnedSampler."""

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_features_kernel(
    probs_ptr,
    mask_ptr,
    input_ids_ptr,
    sem_table_ptr,
    feat_mean_ptr,
    feat_std_ptr,
    n_masked_ptr,
    feat_out_ptr,
    top1_ids_out_ptr,
    V: tl.constexpr,
    V_sem: tl.constexpr,
    BL: tl.constexpr,
    F_DIM: tl.constexpr,
    SEM_D: tl.constexpr,
    BLOCK_V: tl.constexpr,
):

    pid = tl.program_id(0)
    b = pid // BL
    pos = pid % BL

    n_total = BL
    n_total_f = n_total.to(tl.float32)

    n_masked_f = tl.load(n_masked_ptr + b)
    masked_ratio = n_masked_f / n_total_f

    is_masked_val = tl.load(mask_ptr + b * BL + pos).to(tl.float32)

    probs_base = b * BL * V + pos * V

    best_v0: tl.float32 = -1.0e30
    best_i0 = tl.cast(0, tl.int64)
    for tile_start in range(0, V, BLOCK_V):
        offs = tile_start + tl.arange(0, BLOCK_V)
        m = offs < V
        v = tl.load(probs_ptr + probs_base + offs, mask=m, other=-1.0e30)
        tv = tl.max(v)
        if tv > best_v0:
            best_v0 = tv
            best_i0 = (tile_start + tl.argmax(v, axis=0)).to(tl.int64)

    best_v1: tl.float32 = -1.0e30
    best_i1 = tl.cast(0, tl.int64)
    for tile_start in range(0, V, BLOCK_V):
        offs = tile_start + tl.arange(0, BLOCK_V)
        m = offs < V
        v = tl.load(probs_ptr + probs_base + offs, mask=m, other=-1.0e30)
        v = tl.where(offs.to(tl.int64) == best_i0, -1.0e30, v)
        tv = tl.max(v)
        if tv > best_v1:
            best_v1 = tv
            best_i1 = (tile_start + tl.argmax(v, axis=0)).to(tl.int64)

    best_v2: tl.float32 = -1.0e30
    best_i2 = tl.cast(0, tl.int64)
    for tile_start in range(0, V, BLOCK_V):
        offs = tile_start + tl.arange(0, BLOCK_V)
        m = offs < V
        v = tl.load(probs_ptr + probs_base + offs, mask=m, other=-1.0e30)
        offs64 = offs.to(tl.int64)
        v = tl.where(offs64 == best_i0, -1.0e30, v)
        v = tl.where(offs64 == best_i1, -1.0e30, v)
        tv = tl.max(v)
        if tv > best_v2:
            best_v2 = tv
            best_i2 = (tile_start + tl.argmax(v, axis=0)).to(tl.int64)

    best_v3: tl.float32 = -1.0e30
    best_i3 = tl.cast(0, tl.int64)
    for tile_start in range(0, V, BLOCK_V):
        offs = tile_start + tl.arange(0, BLOCK_V)
        m = offs < V
        v = tl.load(probs_ptr + probs_base + offs, mask=m, other=-1.0e30)
        offs64 = offs.to(tl.int64)
        v = tl.where(offs64 == best_i0, -1.0e30, v)
        v = tl.where(offs64 == best_i1, -1.0e30, v)
        v = tl.where(offs64 == best_i2, -1.0e30, v)
        tv = tl.max(v)
        if tv > best_v3:
            best_v3 = tv
            best_i3 = (tile_start + tl.argmax(v, axis=0)).to(tl.int64)

    best_v4: tl.float32 = -1.0e30
    for tile_start in range(0, V, BLOCK_V):
        offs = tile_start + tl.arange(0, BLOCK_V)
        m = offs < V
        v = tl.load(probs_ptr + probs_base + offs, mask=m, other=-1.0e30)
        offs64 = offs.to(tl.int64)
        v = tl.where(offs64 == best_i0, -1.0e30, v)
        v = tl.where(offs64 == best_i1, -1.0e30, v)
        v = tl.where(offs64 == best_i2, -1.0e30, v)
        v = tl.where(offs64 == best_i3, -1.0e30, v)
        tv = tl.max(v)
        if tv > best_v4:
            best_v4 = tv

    tl.store(top1_ids_out_ptr + b * BL + pos, best_i0)

    eps = 1e-10

    top1 = tl.maximum(best_v0, 0.0)
    top2 = tl.maximum(best_v1, 0.0)
    top3 = tl.maximum(best_v2, 0.0)
    top4 = tl.maximum(best_v3, 0.0)
    top5 = tl.maximum(best_v4, 0.0)

    margin = top1 - top2
    top3_mass = top1 + top2 + top3

    log_t1 = tl.log(top1 + eps)
    log_t2 = tl.log(top2 + eps)
    log_t3 = tl.log(top3 + eps)
    log_t4 = tl.log(top4 + eps)
    log_t5 = tl.log(top5 + eps)
    entropy = -(
        top1 * log_t1 + top2 * log_t2 + top3 * log_t3 + top4 * log_t4 + top5 * log_t5
    )

    logprob = log_t1
    gini = 1.0 - (top1 * top1 + top2 * top2 + top3 * top3 + top4 * top4 + top5 * top5)
    pos_feat = pos.to(tl.float32) / tl.maximum((n_total - 1).to(tl.float32), 1.0)
    log_masked = tl.log(n_masked_f + 1.0)

    feat_base = (b * BL + pos) * F_DIM

    scalars_01 = top1
    scalars_02 = top2
    scalars_03 = top3
    scalars_04 = top4
    scalars_05 = top5
    scalars_06 = top1
    scalars_07 = margin
    scalars_08 = top3_mass
    scalars_09 = entropy
    scalars_10 = is_masked_val
    scalars_11 = 0.0
    scalars_12 = masked_ratio

    mean_0 = tl.load(feat_mean_ptr + 0).to(tl.float32)
    std_0 = tl.maximum(tl.load(feat_std_ptr + 0).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 0, ((scalars_01 - mean_0) / std_0).to(tl.float16)
    )

    mean_1 = tl.load(feat_mean_ptr + 1).to(tl.float32)
    std_1 = tl.maximum(tl.load(feat_std_ptr + 1).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 1, ((scalars_02 - mean_1) / std_1).to(tl.float16)
    )

    mean_2 = tl.load(feat_mean_ptr + 2).to(tl.float32)
    std_2 = tl.maximum(tl.load(feat_std_ptr + 2).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 2, ((scalars_03 - mean_2) / std_2).to(tl.float16)
    )

    mean_3 = tl.load(feat_mean_ptr + 3).to(tl.float32)
    std_3 = tl.maximum(tl.load(feat_std_ptr + 3).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 3, ((scalars_04 - mean_3) / std_3).to(tl.float16)
    )

    mean_4 = tl.load(feat_mean_ptr + 4).to(tl.float32)
    std_4 = tl.maximum(tl.load(feat_std_ptr + 4).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 4, ((scalars_05 - mean_4) / std_4).to(tl.float16)
    )

    mean_5 = tl.load(feat_mean_ptr + 5).to(tl.float32)
    std_5 = tl.maximum(tl.load(feat_std_ptr + 5).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 5, ((scalars_06 - mean_5) / std_5).to(tl.float16)
    )

    mean_6 = tl.load(feat_mean_ptr + 6).to(tl.float32)
    std_6 = tl.maximum(tl.load(feat_std_ptr + 6).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 6, ((scalars_07 - mean_6) / std_6).to(tl.float16)
    )

    mean_7 = tl.load(feat_mean_ptr + 7).to(tl.float32)
    std_7 = tl.maximum(tl.load(feat_std_ptr + 7).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 7, ((scalars_08 - mean_7) / std_7).to(tl.float16)
    )

    mean_8 = tl.load(feat_mean_ptr + 8).to(tl.float32)
    std_8 = tl.maximum(tl.load(feat_std_ptr + 8).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 8, ((scalars_09 - mean_8) / std_8).to(tl.float16)
    )

    mean_9 = tl.load(feat_mean_ptr + 9).to(tl.float32)
    std_9 = tl.maximum(tl.load(feat_std_ptr + 9).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 9, ((scalars_10 - mean_9) / std_9).to(tl.float16)
    )

    mean_10 = tl.load(feat_mean_ptr + 10).to(tl.float32)
    std_10 = tl.maximum(tl.load(feat_std_ptr + 10).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 10, ((scalars_11 - mean_10) / std_10).to(tl.float16)
    )

    mean_11 = tl.load(feat_mean_ptr + 11).to(tl.float32)
    std_11 = tl.maximum(tl.load(feat_std_ptr + 11).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + 11, ((scalars_12 - mean_11) / std_11).to(tl.float16)
    )

    input_tok = tl.load(input_ids_ptr + b * BL + pos).to(tl.int64)
    input_tok = tl.minimum(tl.maximum(input_tok, 0), V_sem - 1)
    top1_tok = tl.minimum(tl.maximum(best_i0, 0), V_sem - 1)
    top2_tok = tl.minimum(tl.maximum(best_i1, 0), V_sem - 1)
    top3_tok = tl.minimum(tl.maximum(best_i2, 0), V_sem - 1)

    sem_offsets = tl.arange(0, SEM_D)

    sem_vals_0 = tl.load(
        sem_table_ptr + input_tok * SEM_D + sem_offsets, mask=sem_offsets < SEM_D
    ).to(tl.float32)
    mean_sem_0 = tl.load(feat_mean_ptr + 12 + sem_offsets, mask=sem_offsets < SEM_D).to(
        tl.float32
    )
    std_sem_0 = tl.maximum(
        tl.load(feat_std_ptr + 12 + sem_offsets, mask=sem_offsets < SEM_D).to(
            tl.float32
        ),
        1e-6,
    )
    tl.store(
        feat_out_ptr + feat_base + 12 + sem_offsets,
        ((sem_vals_0 - mean_sem_0) / std_sem_0).to(tl.float16),
        mask=sem_offsets < SEM_D,
    )

    sem_vals_1 = tl.load(
        sem_table_ptr + top1_tok * SEM_D + sem_offsets, mask=sem_offsets < SEM_D
    ).to(tl.float32)
    mean_sem_1 = tl.load(
        feat_mean_ptr + 12 + SEM_D + sem_offsets, mask=sem_offsets < SEM_D
    ).to(tl.float32)
    std_sem_1 = tl.maximum(
        tl.load(feat_std_ptr + 12 + SEM_D + sem_offsets, mask=sem_offsets < SEM_D).to(
            tl.float32
        ),
        1e-6,
    )
    tl.store(
        feat_out_ptr + feat_base + 12 + SEM_D + sem_offsets,
        ((sem_vals_1 - mean_sem_1) / std_sem_1).to(tl.float16),
        mask=sem_offsets < SEM_D,
    )

    sem_vals_2 = tl.load(
        sem_table_ptr + top2_tok * SEM_D + sem_offsets, mask=sem_offsets < SEM_D
    ).to(tl.float32)
    mean_sem_2 = tl.load(
        feat_mean_ptr + 12 + 2 * SEM_D + sem_offsets, mask=sem_offsets < SEM_D
    ).to(tl.float32)
    std_sem_2 = tl.maximum(
        tl.load(
            feat_std_ptr + 12 + 2 * SEM_D + sem_offsets, mask=sem_offsets < SEM_D
        ).to(tl.float32),
        1e-6,
    )
    tl.store(
        feat_out_ptr + feat_base + 12 + 2 * SEM_D + sem_offsets,
        ((sem_vals_2 - mean_sem_2) / std_sem_2).to(tl.float16),
        mask=sem_offsets < SEM_D,
    )

    sem_vals_3 = tl.load(
        sem_table_ptr + top3_tok * SEM_D + sem_offsets, mask=sem_offsets < SEM_D
    ).to(tl.float32)
    mean_sem_3 = tl.load(
        feat_mean_ptr + 12 + 3 * SEM_D + sem_offsets, mask=sem_offsets < SEM_D
    ).to(tl.float32)
    std_sem_3 = tl.maximum(
        tl.load(
            feat_std_ptr + 12 + 3 * SEM_D + sem_offsets, mask=sem_offsets < SEM_D
        ).to(tl.float32),
        1e-6,
    )
    tl.store(
        feat_out_ptr + feat_base + 12 + 3 * SEM_D + sem_offsets,
        ((sem_vals_3 - mean_sem_3) / std_sem_3).to(tl.float16),
        mask=sem_offsets < SEM_D,
    )

    mean_e0 = tl.load(feat_mean_ptr + F_DIM - 4).to(tl.float32)
    std_e0 = tl.maximum(tl.load(feat_std_ptr + F_DIM - 4).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + F_DIM - 4,
        ((logprob - mean_e0) / std_e0).to(tl.float16),
    )

    mean_e1 = tl.load(feat_mean_ptr + F_DIM - 3).to(tl.float32)
    std_e1 = tl.maximum(tl.load(feat_std_ptr + F_DIM - 3).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + F_DIM - 3, ((gini - mean_e1) / std_e1).to(tl.float16)
    )

    mean_e2 = tl.load(feat_mean_ptr + F_DIM - 2).to(tl.float32)
    std_e2 = tl.maximum(tl.load(feat_std_ptr + F_DIM - 2).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + F_DIM - 2,
        ((pos_feat - mean_e2) / std_e2).to(tl.float16),
    )

    mean_e3 = tl.load(feat_mean_ptr + F_DIM - 1).to(tl.float32)
    std_e3 = tl.maximum(tl.load(feat_std_ptr + F_DIM - 1).to(tl.float32), 1e-6)
    tl.store(
        feat_out_ptr + feat_base + F_DIM - 1,
        ((log_masked - mean_e3) / std_e3).to(tl.float16),
    )


def fused_build_features_and_normalize(
    probs: torch.Tensor,
    mask: torch.Tensor,
    input_ids: torch.Tensor,
    sem_table: torch.Tensor,
    feat_mean: torch.Tensor,
    feat_std: torch.Tensor,
    block_length: int,
    f_dim: int = 144,
) -> tuple:
    """Build normalized sampler features."""
    B = probs.shape[0]
    BL = block_length
    V = probs.shape[2]
    V_sem = sem_table.shape[0]
    SEM_D = sem_table.shape[1]

    n_masked = mask.sum(dim=1).float()

    feat_out = torch.empty(B, BL, f_dim, device=probs.device, dtype=torch.float16)
    top1_ids = torch.empty(B, BL, device=probs.device, dtype=torch.int64)

    BLOCK_V = min(4096, triton.next_power_of_2(V))

    grid = (B * BL,)

    _fused_features_kernel[grid](
        probs,
        mask,
        input_ids,
        sem_table,
        feat_mean,
        feat_std,
        n_masked,
        feat_out,
        top1_ids,
        V=V,
        V_sem=V_sem,
        BL=BL,
        F_DIM=f_dim,
        SEM_D=SEM_D,
        BLOCK_V=BLOCK_V,
    )

    return feat_out, top1_ids
