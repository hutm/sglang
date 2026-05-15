import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


def load_dllm_algorithm_config(algorithm_config_path: str | None) -> dict[str, Any]:
    if algorithm_config_path is None:
        return {}

    try:
        import yaml
    except ImportError:
        raise ImportError(
            "Please install PyYAML to use YAML config files. " "`pip install pyyaml`"
        )
    with open(algorithm_config_path, "r") as f:
        return yaml.safe_load(f) or {}


def should_defer_cuda_graph_capture(server_args: "ServerArgs") -> bool:
    if server_args.dllm_algorithm is None:
        return False

    algorithm_config = load_dllm_algorithm_config(server_args.dllm_algorithm_config)
    return bool(algorithm_config.get("lora_path"))


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        max_steps: int,
        causal_context: bool = False,
        block_size_tiers: list[dict[str, int]] | None = None,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.max_steps = max_steps
        # If True, the prefix KV cache is built with causal attention (matches
        # HF's causal_context=True in generate_with_prefix_cache_block_diff).
        # Required for Nemotron-Labs-Diffusion-Exp-Ministral-8B-Instruct; leave False
        # for bidirectional-prefix models like NVRDiff / LLaDA2.
        self.causal_context = causal_context
        # Concurrency-tiered block_size schedule. Each tier specifies the
        # max running batch size that should use the given block_size.
        # Empty / None → static block_size (current default).
        # YAML form:
        #   block_size_tiers:
        #     - {max_running_bs: 4,    block_size: 32}
        #     - {max_running_bs: 32,   block_size: 16}
        #     - {max_running_bs: 9999, block_size: 8}
        # Must be sorted by max_running_bs ascending.
        self.block_size_tiers: list[dict[str, int]] | None = None
        if block_size_tiers:
            if algorithm != "LinearSpec":
                raise ValueError(
                    "block_size_tiers are only supported with LinearSpec; "
                    f"got dllm_algorithm={algorithm}."
                )
            tiers = sorted(
                (
                    {
                        "max_running_bs": int(t["max_running_bs"]),
                        "block_size": int(t["block_size"]),
                    }
                    for t in block_size_tiers
                ),
                key=lambda t: t["max_running_bs"],
            )
            # Validate strictly ascending max_running_bs
            for i in range(1, len(tiers)):
                if tiers[i]["max_running_bs"] <= tiers[i - 1]["max_running_bs"]:
                    raise ValueError(
                        "block_size_tiers max_running_bs must be strictly "
                        f"ascending; got {[t['max_running_bs'] for t in tiers]}"
                    )
            # Verify the last tier covers the full max_running_requests range
            # so no actual workload size falls past the configured policy.
            if tiers[-1]["max_running_bs"] < max_running_requests:
                raise ValueError(
                    f"block_size_tiers last tier max_running_bs="
                    f"{tiers[-1]['max_running_bs']} is below "
                    f"max_running_requests={max_running_requests}; add a "
                    f"catch-all tier (e.g. max_running_bs: 9999)."
                )
            self.block_size_tiers = tiers
            # `block_size` (the static field) is set to the largest tier's
            # block_size — used for KV pool sizing (worst-case allocation).
            # Warn if the caller passed an explicit block_size that diverges,
            # so configs that set both don't silently get overridden.
            max_tier_block = max(t["block_size"] for t in tiers)
            if self.block_size != max_tier_block:
                logger.warning(
                    "DllmConfig: overriding static block_size=%d with "
                    "max(block_size_tiers.block_size)=%d for KV pool sizing.",
                    self.block_size,
                    max_tier_block,
                )
                self.block_size = max_tier_block

    def select_block_size(self, running_bs: int) -> int:
        """Pick block_size for a block dispatched at the given running batch size.

        Falls back to the static block_size when no tiers are configured.
        """
        if not self.block_size_tiers:
            return self.block_size
        for tier in self.block_size_tiers:
            if running_bs <= tier["max_running_bs"]:
                return tier["block_size"]
        # running_bs exceeds all tier bounds; use the last (largest) tier
        return self.block_size_tiers[-1]["block_size"]

    @staticmethod
    def from_server_args(
        server_args: "ServerArgs",
    ):
        if server_args.dllm_algorithm is None:
            return None

        from sglang.srt.configs.model_config import ModelConfig

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )
        DLLM_PARAMS = {
            "LLaDA2MoeModelLM": {"block_size": 32, "mask_id": 156895},
            "SDARForCausalLM": {"block_size": 4, "mask_id": 151669},
            "SDARMoeForCausalLM": {"block_size": 4, "mask_id": 151669},
            "DiffEncoderModel": {"block_size": 32, "mask_id": 151662},
            "NemotronLabsDiffusionEncoderModel": {"block_size": 32, "mask_id": 100},
        }

        arch = model_config.hf_config.architectures[0]
        if arch in DLLM_PARAMS:
            params = DLLM_PARAMS[arch]
            block_size = params["block_size"]
            mask_id = params["mask_id"]
        else:
            raise RuntimeError(f"Unknown diffusion LLM: {arch}")

        algorithm_config = load_dllm_algorithm_config(server_args.dllm_algorithm_config)

        # Parse common algorithm configurations
        block_size = algorithm_config.get("block_size", block_size)

        max_steps = algorithm_config.get("max_steps", block_size)

        causal_context = algorithm_config.get("causal_context", False)
        block_size_tiers = algorithm_config.get("block_size_tiers", None)

        # Compute max_running_requests after YAML config so block_size is final
        if server_args.max_running_requests is not None:
            max_running_requests = server_args.max_running_requests
        else:
            max_running_requests = 1

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            max_steps=max_steps,
            causal_context=causal_context,
            block_size_tiers=block_size_tiers,
        )
