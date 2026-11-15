"""The models uLLM is built and measured with, pinned to exact revisions (F4).

A revision is a commit of the model's Hugging Face repository. Pinning it means
the weights, tokenizer and chat template cannot change underneath the goldens,
evals or benchmarks. Changing one is a reviewed edit that regenerates goldens.
"""

from dataclasses import dataclass
from typing import Final, Literal


@dataclass(frozen=True, slots=True)
class ModelSpec:
    key: str
    repo: str
    revision: str
    dtype: Literal["bf16", "8bit", "4bit"]
    approx_gb: float  # download size, for planning memory and CI caches
    role: str


MODELS: Final[dict[str, ModelSpec]] = {
    spec.key: spec
    for spec in (
        ModelSpec(
            "qwen3-0.6b-bf16",
            "mlx-community/Qwen3-0.6B-bf16",
            "42096995f6402fde107068cf530136fe64b604f8",
            "bf16",
            1.21,
            "the default: goldens, equivalence, CI, multi-replica runs",
        ),
        ModelSpec(
            "qwen3-0.6b-8bit",
            "mlx-community/Qwen3-0.6B-8bit",
            "11de96878523501bcaa86104e3c186de07ff9068",
            "8bit",
            0.65,
            "quantization evals (Q5)",
        ),
        ModelSpec(
            "qwen3-0.6b-4bit",
            "mlx-community/Qwen3-0.6B-4bit",
            "73e3e38d981303bc594367cd910ea6eb48349da8",
            "4bit",
            0.35,
            "quantization evals (Q5)",
        ),
        ModelSpec(
            "qwen3-1.7b-bf16",
            "mlx-community/Qwen3-1.7B-bf16",
            "9cd6692855d3e06772228e9a962b2606359b2d24",
            "bf16",
            3.46,
            "headline benchmark tables (M7-T1)",
        ),
    )
}

DEFAULT_MODEL: Final = MODELS["qwen3-0.6b-bf16"]
