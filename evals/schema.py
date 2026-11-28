"""The shape of an eval run's results (Q5, decision 0035): ``result.json`` holds one
``EvalResult``, and ``items.jsonl`` holds one ``ItemResult`` per GSM8K item."""

from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Perplexity(Frozen):
    window: int  # tokens per window; the first of each is context only
    windows: int
    tokens: int  # tokens scored: windows * (window - 1)
    mean_nll: float  # nats per token
    perplexity: float
    standard_error: float  # of the perplexity, by the delta method


FinishReason = Literal["stop", "length"]


class ItemResult(Frozen):
    index: int
    gold: str
    answer: str | None  # None: no number in a closed \boxed{}
    correct: bool
    finish_reason: FinishReason
    completion_tokens: int
    reply: str


class ExactMatch(Frozen):
    items: int
    correct: int
    score: float
    no_answer: int
    truncated: int  # stopped by max_tokens
    output_tokens: int
    wall_s: float  # first send to last reply, one request at a time
    output_tok_s: float
    server_peak_footprint_bytes: int | None  # None: footprint could not read the server

    @classmethod
    def of(
        cls, items: Sequence[ItemResult], wall_s: float, server_peak_footprint_bytes: int | None
    ) -> "ExactMatch":
        correct = sum(i.correct for i in items)
        tokens = sum(i.completion_tokens for i in items)
        return cls(
            items=len(items),
            correct=correct,
            score=correct / len(items),
            no_answer=sum(i.answer is None for i in items),
            truncated=sum(i.finish_reason == "length" for i in items),
            output_tokens=tokens,
            wall_s=wall_s,
            output_tok_s=tokens / wall_s,
            server_peak_footprint_bytes=server_peak_footprint_bytes,
        )


class EvalResult(Frozen):
    model: str  # a key of ullm.models.MODELS
    repo: str
    revision: str
    dtype: Literal["bf16", "8bit", "4bit"]
    datasets: dict[str, str]  # file under evals/datasets/ -> sha256
    prompt_suffix: str
    max_tokens: int
    chat_template_kwargs: dict[str, bool]
    perplexity: Perplexity
    exact_match: ExactMatch
    tool_calls: None = None  # tool-call validity (Q6) arrives with M6-T2
