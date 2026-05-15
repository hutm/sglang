import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch

LoraDelta = Tuple[torch.nn.Parameter, torch.Tensor, torch.nn.Module]


def load_peft_lora_deltas(lora_path: str, model: torch.nn.Module) -> List[LoraDelta]:
    """Load PEFT LoRA weights and compute fused deltas for DLLM attention layers."""
    lora_dir = Path(lora_path)
    with open(lora_dir / "adapter_config.json") as f:
        acfg = json.load(f)
    r, alpha = acfg["r"], acfg["lora_alpha"]
    scale = alpha / r

    from safetensors.torch import load_file

    lora_weights = load_file(str(lora_dir / "adapter_model.safetensors"))

    layer_projs: Dict[int, Dict[str, Dict[str, torch.Tensor]]] = {}
    for key, tensor in lora_weights.items():
        parts = key.split(".")
        layer_idx = int(parts[parts.index("layers") + 1])
        if "self_attn" in parts:
            proj = parts[parts.index("self_attn") + 1]
        elif "mlp" in parts:
            proj = parts[parts.index("mlp") + 1]
        else:
            continue
        ab = parts[-2]
        layer_projs.setdefault(layer_idx, {}).setdefault(proj, {})[ab] = tensor

    deltas: List[LoraDelta] = []
    for layer_idx, projs in sorted(layer_projs.items()):
        layer = model.model.layers[layer_idx]
        if any(p in projs for p in ("q_proj", "k_proj", "v_proj")):
            qkv_parts = []
            for proj in ("q_proj", "k_proj", "v_proj"):
                if proj in projs:
                    a = projs[proj]["lora_A"]
                    b = projs[proj]["lora_B"]
                    qkv_parts.append((b @ a) * scale)
                else:
                    raise ValueError(f"Missing {proj} in LoRA layer {layer_idx}")
            qkv_delta = torch.cat(qkv_parts, dim=0)
            param = layer.self_attn.qkv_proj.weight
            deltas.append(
                (
                    param,
                    qkv_delta.to(param.dtype).to(param.device),
                    layer.self_attn.qkv_proj,
                )
            )
        if "o_proj" in projs:
            a = projs["o_proj"]["lora_A"]
            b = projs["o_proj"]["lora_B"]
            o_delta = (b @ a) * scale
            param = layer.self_attn.o_proj.weight
            deltas.append(
                (
                    param,
                    o_delta.to(param.dtype).to(param.device),
                    layer.self_attn.o_proj,
                )
            )

    return deltas
