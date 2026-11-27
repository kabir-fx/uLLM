"""The protocol's invariants 1, 2 and 4 (``runner.base``), kept once for every runner."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from ullm.runner.base import (
    BlockId,
    DecodePlan,
    PrefillPlan,
    RunnerInfo,
    SeqId,
    StepPlan,
    require_ints,
)


@dataclass(frozen=True, slots=True)
class _Live:
    held: int  # tokens in the sequence's KV
    decoding: bool  # false from add to the first decode (decision 0019)


class SeqLedger:
    def __init__(self, info: RunnerInfo) -> None:
        self._info = info
        # None once a step on the sequence raised: its KV is then in an unknown state, and
        # only release is legal.
        self._seqs: dict[SeqId, _Live | None] = {}

    def add(self, seq: SeqId) -> None:
        require_ints("seq", (seq,))
        if seq in self._seqs:
            raise ValueError(f"sequence {seq} is already added")
        self._seqs[seq] = _Live(held=0, decoding=False)

    def release(self, seq: SeqId, freed: tuple[BlockId, ...]) -> None:
        require_ints("seq", (seq,))
        if type(freed) is not tuple:
            raise ValueError(f"freed must be a tuple, got {type(freed).__name__}")
        require_ints("freed block ids", freed)
        self._live(seq)
        del self._seqs[seq]

    def held(self, seq: SeqId) -> int:
        return self._entry(seq).held

    def _entry(self, seq: SeqId) -> _Live:
        live = self._seqs[self._live(seq)]
        if live is None:
            raise ValueError(f"sequence {seq} failed in an earlier step; only release is legal")
        return live

    @contextmanager
    def step(self, plan: StepPlan) -> Iterator[None]:
        """Check the plan, then record its tokens if the body succeeds or fail its sequences."""
        growth = self._check(plan)
        try:
            yield
        except BaseException:
            for seq in growth:
                self._seqs[seq] = None
            raise
        decoding = isinstance(plan, DecodePlan)
        for seq, held in growth.items():
            self._seqs[seq] = _Live(held=held, decoding=decoding)

    def _check(self, plan: StepPlan) -> dict[SeqId, int]:
        match plan:
            case PrefillPlan(seq=seq, token_ids=token_ids, start=start, blocks=blocks):
                live = self._entry(seq)
                if live.decoding:
                    raise ValueError(
                        f"sequence {seq} is decoding; a prefill comes before its first decode "
                        "(0019)"
                    )
                if start != live.held:
                    raise ValueError(
                        f"sequence {seq} holds {live.held} tokens; prefill start {start}"
                    )
                growth = {seq: live.held + len(token_ids)}
                tables = {seq: blocks}
            case DecodePlan(seqs=seqs, token_ids=token_ids, blocks=blocks):
                growth = {seq: self.held(seq) + 1 for seq in seqs}
                tables = dict(zip(seqs, blocks, strict=True))
                omitted = sorted(
                    s
                    for s, live in self._seqs.items()
                    if live is not None and live.decoding and s not in growth
                )
                if omitted:
                    raise ValueError(
                        f"decode plan omits decoding sequences {omitted}; a batch decodes "
                        "together (0019)"
                    )
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
        size = self._info.block_size
        for seq, held in growth.items():
            if len(tables[seq]) != -(-held // size):
                raise ValueError(
                    f"sequence {seq} holds {held} tokens after this step, which take "
                    f"{-(-held // size)} blocks of {size}; the plan names {len(tables[seq])}"
                )
        return growth

    def _live(self, seq: SeqId) -> SeqId:
        if seq not in self._seqs:
            raise ValueError(f"unknown sequence {seq}")
        return seq
