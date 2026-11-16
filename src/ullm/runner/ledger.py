"""The protocol's invariants 1, 2 and 4 (``runner.base``), kept once for every runner."""

from collections.abc import Iterator
from contextlib import contextmanager

from ullm.runner.base import DecodePlan, PrefillPlan, RunnerInfo, SeqId, StepPlan, require_ints


class SeqLedger:
    def __init__(self, info: RunnerInfo) -> None:
        self._info = info
        # Tokens in the sequence's KV, or None once a step on it raised: its KV is then
        # in an unknown state, and only release is legal.
        self._held: dict[SeqId, int | None] = {}

    def add(self, seq: SeqId) -> None:
        require_ints("seq", (seq,))
        if seq in self._held:
            raise ValueError(f"sequence {seq} is already added")
        self._held[seq] = 0

    def release(self, seq: SeqId) -> None:
        require_ints("seq", (seq,))
        self._live(seq)
        del self._held[seq]

    def held(self, seq: SeqId) -> int:
        held = self._held[self._live(seq)]
        if held is None:
            raise ValueError(f"sequence {seq} failed in an earlier step; only release is legal")
        return held

    @contextmanager
    def step(self, plan: StepPlan) -> Iterator[None]:
        """Check the plan, then record its tokens if the body succeeds or fail its sequences."""
        growth = self._check(plan)
        try:
            yield
        except BaseException:
            for seq in growth:
                self._held[seq] = None
            raise
        for seq, held in growth.items():
            self._held[seq] = held

    def _check(self, plan: StepPlan) -> dict[SeqId, int]:
        match plan:
            case PrefillPlan(seq=seq, token_ids=token_ids, start=start):
                held = self.held(seq)
                if start != held:
                    raise ValueError(f"sequence {seq} holds {held} tokens; prefill start {start}")
                growth = {seq: held + len(token_ids)}
            case DecodePlan(seqs=seqs, token_ids=token_ids):
                growth = {seq: self.held(seq) + 1 for seq in seqs}
        for seq, held in growth.items():
            if held > self._info.max_model_len:
                raise ValueError(
                    f"sequence {seq} would hold {held} tokens, over max_model_len "
                    f"{self._info.max_model_len}"
                )
        bad = [t for t in token_ids if not 0 <= t < self._info.vocab_size]
        if bad:
            raise ValueError(
                f"token ids {bad} are outside the vocabulary [0, {self._info.vocab_size})"
            )
        return growth

    def _live(self, seq: SeqId) -> SeqId:
        if seq not in self._held:
            raise ValueError(f"unknown sequence {seq}")
        return seq
