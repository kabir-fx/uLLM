"""The stop check (E3, decision 0009): one per request, fed each sampled token in turn.

A reply ends on an end token (``"stop"``), on ``max_tokens`` (``"length"``), or on
a stop string (``"stop"``). The stop string is never emitted, so text that could
be its start is held back until it cannot. When stop strings overlap, the one
that ends first wins, so the result does not depend on where tokens split the text.
"""

from dataclasses import dataclass
from typing import Literal

from ullm.runner.base import Detokenizer

FinishReason = Literal["stop", "length"]


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str  # may be empty while text is held back
    finish_reason: FinishReason | None


class StopCheck:
    def __init__(
        self,
        detokenizer: Detokenizer,
        eos_token_ids: frozenset[int],
        max_tokens: int,
        stop: tuple[str, ...],
    ) -> None:
        if type(max_tokens) is not int or max_tokens < 1:
            raise ValueError(f"max_tokens must be an int >= 1, got {max_tokens!r}")
        if type(stop) is not tuple or not all(type(s) is str and s for s in stop):
            raise ValueError(f"stop must be a tuple of non-empty strings, got {stop!r}")
        self._detokenizer = detokenizer
        self._eos = eos_token_ids
        self._max_tokens = max_tokens
        self._stop = stop
        self._generated = 0
        self._pending = ""  # decoded, not emitted: it could be the start of a stop string
        self._finished = False

    def add(self, token: int) -> Chunk:
        if self._finished:
            raise ValueError("the reply has finished; no more tokens")
        self._generated += 1
        finish: FinishReason | None = None
        if token in self._eos:
            self._pending += self._detokenizer.flush()
            finish = "stop"
        else:
            self._pending += self._detokenizer.add(token)
            if self._generated == self._max_tokens:
                self._pending += self._detokenizer.flush()
                finish = "length"
        cut = self._first_stop()
        if cut is not None:
            return self._finish(self._pending[:cut], "stop")
        if finish is not None:
            return self._finish(self._pending, finish)
        safe = len(self._pending) - self._held()
        text, self._pending = self._pending[:safe], self._pending[safe:]
        return Chunk(text, None)

    def _finish(self, text: str, reason: FinishReason) -> Chunk:
        self._finished = True
        self._pending = ""
        return Chunk(text, reason)

    def _first_stop(self) -> int | None:
        """Where the stop string that ends first starts; the longer one on a tie."""
        hits = []
        for s in self._stop:
            start = self._pending.find(s)
            if start >= 0:
                hits.append((start + len(s), start))
        return min(hits)[1] if hits else None

    def _held(self) -> int:
        """The length of the longest end of the pending text that starts a stop string."""
        longest = min(len(self._pending), max((len(s) for s in self._stop), default=1) - 1)
        for n in range(longest, 0, -1):
            if any(s.startswith(self._pending[-n:]) for s in self._stop):
                return n
        return 0
