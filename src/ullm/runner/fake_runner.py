"""A runner with no model and no MLX, for core tests and ``ullm sim`` (E1, decision 0008).

Each token is a hash of the sequence's seed, its id and its position, so output
never depends on what else is in the batch. Each step sleeps for a modelled
time and reports that time, so telemetry shows the latency model rather than
the host's noise.
"""

import hashlib
import time
from collections.abc import Callable
from dataclasses import dataclass

from ullm.runner.base import (
    DecodePlan,
    PrefillPlan,
    RunnerInfo,
    SamplingParams,
    SeqId,
    StepPlan,
    StepResult,
)
from ullm.runner.ledger import SeqLedger


@dataclass(frozen=True, slots=True)
class Latency:
    prefill_base_ms: float
    prefill_per_token_ms: float
    decode_base_ms: float
    decode_per_row_ms: float

    def of(self, plan: StepPlan) -> float:
        match plan:
            case PrefillPlan(token_ids=token_ids):
                return self.prefill_base_ms + self.prefill_per_token_ms * len(token_ids)
            case DecodePlan(seqs=seqs):
                return self.decode_base_ms + self.decode_per_row_ms * len(seqs)


class FakeRunner:
    def __init__(
        self,
        info: RunnerInfo,
        latency: Latency,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if info.vocab_size <= len(info.eos_token_ids):
            raise ValueError("FakeRunner needs a token that is not EOS")
        self._info = info
        self._latency = latency
        self._sleep = sleep
        self._ledger = SeqLedger(info)
        self._seeds: dict[SeqId, int] = {}

    def info(self) -> RunnerInfo:
        return self._info

    def add(self, seq: SeqId, sampling: SamplingParams) -> None:
        self._ledger.add(seq)
        self._seeds[seq] = sampling.seed or 0

    def release(self, seq: SeqId) -> None:
        self._ledger.release(seq)
        del self._seeds[seq]

    def execute(self, plan: StepPlan) -> StepResult:
        with self._ledger.step(plan):
            match plan:
                case PrefillPlan(seq=seq, token_ids=token_ids, start=start, sample=sample):
                    rows = [(seq, start + len(token_ids))] if sample else []
                case DecodePlan(seqs=seqs):
                    rows = [(seq, self._ledger.held(seq) + 1) for seq in seqs]
            ms = self._latency.of(plan)
            self._sleep(ms / 1000)
            tokens = tuple(self._token(seq, position) for seq, position in rows)
        return StepResult(tokens=tokens, forward_ms=ms, total_ms=ms)

    def _token(self, seq: SeqId, position: int) -> int:
        """The token sampled at ``position``, the index it takes in the sequence. Never EOS."""
        key = f"{self._seeds[seq]}:{seq}:{position}".encode()
        digest = hashlib.blake2b(key, digest_size=8).digest()
        eos = sorted(self._info.eos_token_ids)
        token = int.from_bytes(digest) % (self._info.vocab_size - len(eos))
        for skipped in eos:  # maps [0, vocab - |eos|) onto the ids that are not EOS
            if token >= skipped:
                token += 1
        return token
