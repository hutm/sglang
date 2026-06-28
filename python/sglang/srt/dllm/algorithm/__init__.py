import importlib
import logging
import pkgutil
from collections.abc import Callable
from typing import TypeVar

from sglang.srt.dllm.config import DllmConfig

logger = logging.getLogger(__name__)

AlgorithmT = TypeVar("AlgorithmT")
algo_name_to_cls: dict[str, type] = {}


def register_algorithm(
    name: str | None = None, *, overwrite: bool = False
) -> Callable[[type[AlgorithmT]], type[AlgorithmT]]:
    """Register a diffusion-LM decoding algorithm.

    External packages can call this decorator from an ``sglang.srt.plugins``
    entry point before workers initialize their DLLM algorithm.
    """

    def decorator(algorithm_cls: type[AlgorithmT]) -> type[AlgorithmT]:
        algorithm_name = name or algorithm_cls.__name__
        if not algorithm_name:
            raise ValueError("DLLM algorithm name must be non-empty")
        existing = algo_name_to_cls.get(algorithm_name)
        if existing is not None and existing is not algorithm_cls and not overwrite:
            raise ValueError(f"DLLM algorithm {algorithm_name!r} is already registered")
        algo_name_to_cls[algorithm_name] = algorithm_cls
        return algorithm_cls

    return decorator


def import_algorithms() -> None:
    package_name = "sglang.srt.dllm.algorithm"
    package = importlib.import_module(package_name)
    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if ispkg:
            continue
        try:
            module = importlib.import_module(name)
        except Exception as e:
            logger.warning(f"Ignore import error when loading {name}: {e}")
            continue
        if not hasattr(module, "Algorithm"):
            continue

        register_algorithm()(module.Algorithm)


def get_algorithm(config: DllmConfig):
    name = config.algorithm
    try:
        algorithm_cls = algo_name_to_cls[name]
    except KeyError as exc:
        raise RuntimeError(f"Unknown diffusion LLM algorithm: {name}") from exc
    return algorithm_cls(config)


import_algorithms()
