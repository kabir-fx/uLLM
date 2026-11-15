"""C3: the runner protocol, v0 (decision 0003).

The boundary between the core, which makes every decision, and the runner,
which owns every tensor. The core says which sequences run and which tokens
they process; the runner runs exactly that, samples, calls ``mx.eval`` and
reports timings taken after it.

This module is MLX-free so ``ullm.core`` can import it. It is a sketch: the
exact signatures are settled in M1-T1 and recorded in a decision then.
"""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

SeqId = int  # engine-internal; the core maps request ids to these


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature: float  # 0 means greedy
    top_p: float
    top_k: int  # 0 means off
    seed: int | None


@dataclass(frozen=True, slots=True)
class RunnerInfo:
    model_id: str
    vocab_size: int
    eos_token_ids: frozenset[int]
    max_model_len: int


@dataclass(frozen=True, slots=True)
class PrefillPlan:
    """Prefill one sequence: input ``[1, L]``, no padding (DEC-7)."""

    seq: SeqId
    token_ids: tuple[int, ...]  # the tokens to run this step
    start: int  # position of token_ids[0]: 0, or more after a prefix hit or an earlier chunk
    sample: bool = True  # false for a chunk that is not the last (E13)

    def __post_init__(self) -> None:
        if not self.token_ids:
            raise ValueError("a prefill step needs at least one token")
        if self.start < 0:
            raise ValueError("start must be non-negative")


@dataclass(frozen=True, slots=True)
class DecodePlan:
    """Decode one token for each sequence: input ``[B, 1]``."""

    seqs: tuple[SeqId, ...]  # batch order
    token_ids: tuple[int, ...]  # each sequence's last token, fed back

    def __post_init__(self) -> None:
        if not self.seqs:
            raise ValueError("a decode step needs at least one sequence")
        if len(self.seqs) != len(self.token_ids):
            raise ValueError("one input token per sequence")
        if len(set(self.seqs)) != len(self.seqs):
            raise ValueError("a sequence appears twice in one decode step")


StepPlan = PrefillPlan | DecodePlan


@dataclass(frozen=True, slots=True)
class StepResult:
    tokens: tuple[int, ...]  # one per sampled row, in plan order; empty if nothing was sampled
    forward_ms: float  # taken after mx.eval (rule 4)
    total_ms: float


@runtime_checkable
class Runner(Protocol):
    def info(self) -> RunnerInfo: ...

    def add(self, seq: SeqId, sampling: SamplingParams) -> None:
        """Create the sequence's state: its KV and its own random state."""
        ...

    def execute(self, plan: StepPlan) -> StepResult: ...

    def release(self, seq: SeqId) -> None:
        """Free everything the runner holds for the sequence."""
        ...
