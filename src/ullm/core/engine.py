"""The engine thread's loop, and the types that cross the bridge (E5, decision 0010)."""

from dataclasses import dataclass

from ullm.runner.base import SamplingParams, require_ints


@dataclass(frozen=True, slots=True)
class Submit:
    """API thread to engine thread: one new request. Plain data, checked when built."""

    rid: str
    prompt: tuple[int, ...]
    sampling: SamplingParams
    max_tokens: int
    stop: tuple[str, ...]
    trace: str | None = None  # W3C traceparent, carried across the thread boundary (C4)
    tenant: str | None = None  # DEC-9 hook; set from M8

    def __post_init__(self) -> None:
        if type(self.rid) is not str or not self.rid:
            raise ValueError(f"rid must be a non-empty str, got {self.rid!r}")
        if type(self.prompt) is not tuple or not self.prompt:
            raise ValueError(f"prompt must be a non-empty tuple, got {self.prompt!r}")
        require_ints("prompt token ids", self.prompt)
        if min(self.prompt) < 0:
            raise ValueError(f"prompt token ids must be >= 0, got {self.prompt!r}")
        if not isinstance(self.sampling, SamplingParams):
            raise ValueError(f"sampling must be SamplingParams, got {self.sampling!r}")
        if type(self.max_tokens) is not int or self.max_tokens < 1:
            raise ValueError(f"max_tokens must be an int >= 1, got {self.max_tokens!r}")
        if type(self.stop) is not tuple or not all(type(s) is str and s for s in self.stop):
            raise ValueError(f"stop must be a tuple of non-empty strings, got {self.stop!r}")
        for name in ("trace", "tenant"):
            value = getattr(self, name)
            if value is not None and type(value) is not str:
                raise ValueError(f"{name} must be a str or None, got {value!r}")
