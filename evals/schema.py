"""The shape of an eval run's results (Q5, decision 0035): ``result.json`` holds one
``EvalResult``, and ``items.jsonl`` holds one ``ItemResult`` per GSM8K item."""

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
