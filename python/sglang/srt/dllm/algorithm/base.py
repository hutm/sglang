from typing import TYPE_CHECKING

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig

if TYPE_CHECKING:
    from sglang.srt.server_args import ServerArgs


class DllmAlgorithm:

    def __init__(
        self,
        config: DllmConfig,
    ):
        self.config = config
        self.block_size = config.block_size
        self.mask_id = config.mask_id

    def select_block_size(self, running_bs: int) -> int:
        return self.config.select_block_size(running_bs)

    @staticmethod
    def from_server_args(server_args: "ServerArgs"):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)
