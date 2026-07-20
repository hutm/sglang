"""Checkpoint helpers for the learned diffusion sampler."""

import importlib.util
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _build_rope_cache(
    seq_len: int,
    head_dim: int,
    device: torch.device,
    theta: float = 10000.0,
):
    pos = torch.arange(seq_len, device=device, dtype=torch.float32)
    dim = torch.arange(0, head_dim, 2, device=device, dtype=torch.float32)
    freqs = pos.unsqueeze(1) / (theta ** (dim / head_dim))
    return freqs.cos(), freqs.sin()


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    L = x.shape[2]
    cos = cos[:L].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
    sin = sin[:L].unsqueeze(0).unsqueeze(0).to(dtype=x.dtype)
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
    return out.flatten(-2)


class _RoPEAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale = self.head_dim**-0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        if qk_norm:
            self.q_norm = nn.LayerNorm(self.head_dim)
            self.k_norm = nn.LayerNorm(self.head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        self._rope_cache_len = 0
        self._rope_cos: Optional[torch.Tensor] = None
        self._rope_sin: Optional[torch.Tensor] = None

    def _get_rope(self, seq_len: int, device: torch.device):
        if seq_len > self._rope_cache_len:
            self._rope_cos, self._rope_sin = _build_rope_cache(
                max(seq_len, 128), self.head_dim, device
            )
            self._rope_cache_len = max(seq_len, 128)
        return self._rope_cos, self._rope_sin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        H, hd = self.nhead, self.head_dim
        qkv = self.qkv(x).reshape(B, L, 3, H, hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        if self.q_norm is not None:
            q = self.q_norm(q)
            k = self.k_norm(k)
        cos, sin = self._get_rope(L, x.device)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))
        return self.proj((attn @ v).transpose(1, 2).reshape(B, L, D))


class _TransformerLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_ff: int,
        dropout: float = 0.0,
        qk_norm: bool = True,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = _RoPEAttention(d_model, nhead, dropout=dropout, qk_norm=qk_norm)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ff(self.norm2(x))


class _BlockAcceptanceTransformerRoPE(nn.Module):
    def __init__(
        self,
        f_dim: int = 144,
        d_model: int = 192,
        nhead: int = 8,
        dim_ff: int = 384,
        n_layers: int = 4,
        dropout: float = 0.0,
        qk_norm: bool = True,
        mlp_pre_encoder: bool = False,
    ):
        super().__init__()
        if mlp_pre_encoder:
            mid = max(d_model, f_dim)
            self.input_proj = nn.Sequential(
                nn.Linear(f_dim, mid),
                nn.GELU(),
                nn.LayerNorm(mid),
                nn.Dropout(dropout),
                nn.Linear(mid, d_model),
            )
        else:
            self.input_proj = nn.Linear(f_dim, d_model)
        self.drop_in = nn.Dropout(dropout)
        self.layers = nn.ModuleList(
            [
                _TransformerLayer(d_model, nhead, dim_ff, dropout, qk_norm=qk_norm)
                for _ in range(n_layers)
            ]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.drop_in(self.input_proj(x))
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm(x)).squeeze(-1)


class _PointwiseMLP(nn.Module):
    def __init__(self, f_dim: int = 144, hidden: int = 384, n_layers: int = 4):
        super().__init__()
        layers = []
        in_dim = f_dim
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(in_dim, hidden), nn.GELU()])
            in_dim = hidden
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, BL, F = x.shape
        return self.net(x.reshape(B * BL, F)).reshape(B, BL)


def _load_sampler(checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    is_v1_ckpt = "feat_mean" in ckpt

    ckpt_dir = os.path.dirname(os.path.abspath(checkpoint_path))
    sampler_model_py = os.path.join(ckpt_dir, "sampler_model.py")
    if not is_v1_ckpt and os.path.isfile(sampler_model_py):
        return _load_sampler_v2(checkpoint_path, sampler_model_py, device)
    sd = ckpt["state_dict"]
    f_dim = ckpt.get("f_dim", 144)

    arch = ckpt.get("arch", "transformer")

    if arch == "pointwise_mlp":
        hidden = ckpt["hidden_dim"]
        n_layers = ckpt["n_layers"]
        model = (
            _PointwiseMLP(
                f_dim=f_dim,
                hidden=hidden,
                n_layers=n_layers,
            )
            .to(device)
            .half()
        )
        model.load_state_dict(sd)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Loaded MLP sampler from {checkpoint_path} "
            f"(hidden={hidden}, layers={n_layers}, f_dim={f_dim}, "
            f"params={n_params:,})"
        )
    else:
        d_model = ckpt["d_model"]
        dim_ff = sd["layers.0.ff.0.weight"].shape[0]
        n_layers = sum(1 for k in sd if k.endswith(".attn.qkv.weight"))
        has_qk = "layers.0.attn.q_norm.weight" in sd
        has_mlp_pre = "input_proj.0.weight" in sd

        if has_qk:
            head_dim = sd["layers.0.attn.q_norm.weight"].shape[0]
            nhead = d_model // head_dim
        else:
            nhead = 8

        model = (
            _BlockAcceptanceTransformerRoPE(
                f_dim=f_dim,
                d_model=d_model,
                nhead=nhead,
                dim_ff=dim_ff,
                n_layers=n_layers,
                dropout=0.0,
                qk_norm=has_qk,
                mlp_pre_encoder=has_mlp_pre,
            )
            .to(device)
            .half()
        )
        model.load_state_dict(sd)
        model.eval()

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(
            f"Loaded sampler from {checkpoint_path} "
            f"(d={d_model}, layers={n_layers}, f_dim={f_dim}, "
            f"mlp_pre={has_mlp_pre}, params={n_params:,})"
        )

    return (
        model,
        ckpt["feat_mean"].to(device).half(),
        ckpt["feat_std"].to(device).half(),
        {},
    )


def _load_sampler_v2(checkpoint_path: str, sampler_model_py: str, device: torch.device):
    spec = importlib.util.spec_from_file_location("_sampler_model", sampler_model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    loaded = mod.load_model(checkpoint_path, device=str(device))
    if isinstance(loaded, tuple) and len(loaded) == 4:
        model, feat_mean, feat_std, meta = loaded
    else:
        model, feat_mean, feat_std = loaded
        meta = {}

    n_params = sum(p.numel() for p in model.parameters())

    if feat_mean is not None and isinstance(feat_mean, torch.Tensor):
        model = model.to(device).half()
        model.eval()
        logger.info(
            f"Loaded v1 sampler via sampler_model.py from {checkpoint_path} "
            f"(params={n_params:,}, f_dim={feat_mean.shape[0]})"
        )
        return (model, feat_mean.to(device).half(), feat_std.to(device).half(), {})

    model = model.to(device).float()
    model.eval()
    num_classes = meta.get("num_classes", 1)
    logger.info(
        f"Loaded v2 sampler from {checkpoint_path} "
        f"(kind={meta.get('sampler_kind', 'unknown')}, "
        f"classes={num_classes}, params={n_params:,})"
    )

    meta["is_v2"] = True
    return model, None, None, meta


def _build_features_144(
    probs: torch.Tensor,
    mask_idx: torch.Tensor,
    block_length: int,
    sem_table: torch.Tensor,
    device: torch.device,
    input_ids: torch.Tensor,
    f_dim: int = 144,
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = probs.shape[0]
    n_total = block_length

    top5_vals, top5_ids = probs.topk(5, dim=-1)
    top5_f = top5_vals.float()

    n_masked_per_b = mask_idx.sum(dim=1).float()
    masked_ratio = n_masked_per_b / n_total

    feat_t = torch.zeros(B, n_total, f_dim, device=device, dtype=torch.float32)

    feat_t[:, :, :5] = top5_f
    feat_t[:, :, 5] = top5_f[:, :, 0]
    feat_t[:, :, 6] = top5_f[:, :, 0] - top5_f[:, :, 1]
    feat_t[:, :, 7] = top5_f[:, :, :3].sum(dim=-1)
    feat_t[:, :, 8] = -(top5_f * torch.log(top5_f + 1e-10)).sum(dim=-1)
    feat_t[:, :, 9] = mask_idx.float()
    feat_t[:, :, 10] = 0.0
    feat_t[:, :, 11] = masked_ratio.unsqueeze(1).expand(-1, n_total)

    sem_d = (f_dim - 16) // 4
    for ki, off in enumerate([12, 12 + sem_d, 12 + 2 * sem_d, 12 + 3 * sem_d]):
        if ki == 0:
            tok_ids = input_ids.reshape(-1)
        elif ki == 1:
            tok_ids = top5_ids[:, :, 0].reshape(-1)
        else:
            tok_ids = top5_ids[:, :, ki - 1].reshape(-1)
        tok_ids = tok_ids.clamp(0, sem_table.shape[0] - 1)
        feat_t[:, :, off : off + sem_d] = sem_table[tok_ids, :sem_d].reshape(
            B, n_total, sem_d
        )

    feat_t[:, :, f_dim - 4] = torch.log(top5_f[:, :, 0] + 1e-10)
    feat_t[:, :, f_dim - 3] = 1.0 - (top5_f**2).sum(dim=-1)
    pos = torch.arange(n_total, device=device, dtype=torch.float32) / max(
        n_total - 1, 1
    )
    feat_t[:, :, f_dim - 2] = pos.unsqueeze(0).expand(B, -1)
    n_m = n_masked_per_b.float()
    feat_t[:, :, f_dim - 1] = torch.log(n_m + 1).unsqueeze(1).expand(-1, n_total)

    return feat_t, top5_ids
