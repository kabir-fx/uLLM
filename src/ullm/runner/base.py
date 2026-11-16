"""C3: the runner protocol, v1 (decisions 0003 and 0008).

The boundary between the core, which makes every decision, and the runner,
which owns every tensor. The core says which sequences run and which tokens
they process; the runner runs exactly that, samples, calls ``mx.eval`` and
reports timings taken after it.

Invariants every runner keeps, and every caller may rely on:

1. Lifecycle per ``SeqId``: ``add`` once, then ``execute`` any number of times,
   then ``release`` once. ``add`` of a live id, or ``execute`` or ``release`` of
   an unknown id, raises ``ValueError``.
2. Positions are contiguous. A ``PrefillPlan`` starts at the number of tokens
   the runner holds for the sequence; a ``DecodePlan`` row appends one token.
   No sequence grows past ``max_model_len``, and every token ID is in
   ``[0, vocab_size)``. A violation raises ``ValueError`` before any forward
   pass and changes nothing. Plans, ``SamplingParams``, ``add`` and ``release``
   refuse sequence ids, positions, token ids, flags, sampling settings and
   containers of the wrong type or range.
3. ``StepResult.tokens`` has one entry per sampled row, in plan order.
4. An ``execute`` that fails after validation, in the forward pass or the
   eval, marks every sequence in the plan as failed. The only legal call for a
   failed sequence is ``release``.
5. Not thread-safe: only the engine thread calls a runner (rule 2).

This module is MLX-free so ``ullm.core`` can import it.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict, runtime_checkable


def require_ints(name: str, values: tuple[object, ...]) -> None:
    # Annotations are not checked at runtime, and bool is a subclass of int.
    wrong = [v for v in values if type(v) is not int]
    if wrong:
        raise ValueError(f"{name} must be int, got {wrong!r}")


def _require_tuple(name: str, value: object) -> None:
    # A list stays mutable inside a frozen plan; a generator is used up by the check.
    if type(value) is not tuple:
        raise ValueError(f"{name} must be a tuple, got {type(value).__name__}")


SeqId = int  # engine-internal; the core maps request ids to these


@dataclass(frozen=True, slots=True)
class SamplingParams:
    temperature: float  # 0 means greedy
    top_p: float  # 1 means off
    top_k: int  # 0 means off
    seed: int | None  # None promises no reproducibility; each runner chooses its own

    def __post_init__(self) -> None:
        # The sampler divides float32 log-probabilities by the temperature, so it must be a
        # number float32 holds exactly enough: 0 (greedy) or a normal float32 (decision 0009).
        if not (
            _is_number(self.temperature)
            and (self.temperature == 0 or _FLOAT32_TINY <= self.temperature <= _FLOAT32_MAX)
        ):
            raise ValueError(
                f"temperature must be 0 or a normal float32 > 0, got {self.temperature!r}"
            )
        if not (_is_number(self.top_p) and 0 < self.top_p <= 1):
            raise ValueError(f"top_p must be in (0, 1], got {self.top_p!r}")
        if type(self.top_k) is not int or self.top_k < 0:
            raise ValueError(f"top_k must be an int >= 0, got {self.top_k!r}")
        if self.seed is not None and type(self.seed) is not int:
            raise ValueError(f"seed must be an int or None, got {self.seed!r}")
        # An int would reach MLX as an int, which it cannot convert past int64.
        object.__setattr__(self, "temperature", float(self.temperature))
        object.__setattr__(self, "top_p", float(self.top_p))


_FLOAT32_TINY = 1.1754943508222875e-38  # the smallest normal float32
_FLOAT32_MAX = 3.4028234663852886e38


def _is_number(value: object) -> bool:
    return type(value) in (int, float)


@dataclass(frozen=True, slots=True)
class RunnerInfo:
    model_id: str
    vocab_size: int
    eos_token_ids: frozenset[int]
    max_model_len: int


@dataclass(frozen=True, slots=True)
class PrefillPlan:
    """Prefill one sequence, no padding (DEC-7). The runner may split the forward (0008)."""

    seq: SeqId
    token_ids: tuple[int, ...]  # the tokens to run this step
    start: int  # position of token_ids[0]: 0, or more after a prefix hit or an earlier chunk
    sample: bool = True  # false for a chunk that is not the last (E13)

    def __post_init__(self) -> None:
        _require_tuple("token_ids", self.token_ids)
        require_ints("seq, start and token ids", (self.seq, self.start, *self.token_ids))
        if type(self.sample) is not bool:
            raise ValueError(f"sample must be a bool, got {self.sample!r}")
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
        _require_tuple("seqs", self.seqs)
        _require_tuple("token_ids", self.token_ids)
        require_ints("seqs and token ids", (*self.seqs, *self.token_ids))
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

    def execute(self, plan: StepPlan) -> StepResult:
        """Run one step. With ``sample`` set, a multi-token prefill samples once, at its end."""
        ...

    def release(self, seq: SeqId) -> None:
        """Free everything the runner holds for the sequence."""
        ...


class TemplateMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


class Detokenizer(Protocol):
    """One sequence's output tokens to text, streamed. Concatenated, its chunks are the
    tokenizer's decoding of every token added."""

    def add(self, token: int) -> str:
        """The text this token completes; empty while a character is still incomplete."""
        ...

    def flush(self) -> str:
        """The text held back, with any unfinished character as U+FFFD, at the end."""
        ...


class Tokenizer(Protocol):
    def render(
        self, messages: Sequence[TemplateMessage], template_kwargs: Mapping[str, object]
    ) -> tuple[int, ...]:
        """Apply the model's chat template with a generation prompt, then tokenize."""
        ...

    def detokenizer(self) -> Detokenizer:
        """A new streaming detokenizer for one sequence."""
        ...
