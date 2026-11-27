"""A runner wrapper that records what the engine asked of it, for the engine and bridge tests."""

from ullm.runner.base import (
    BlockId,
    DecodePlan,
    PrefillPlan,
    Runner,
    RunnerInfo,
    SamplingParams,
    SeqId,
    StepPlan,
    StepResult,
)
from ullm.telemetry import events as ev


class RecordingRunner:
    def __init__(self, inner: Runner) -> None:
        self.inner = inner
        self.adds = 0
        self.releases = 0
        self.live: set[SeqId] = set()
        self.steps: list[tuple[tuple[SeqId, ...], tuple[int, ...]]] = []  # successful only

    def info(self) -> RunnerInfo:
        return self.inner.info()

    def add(self, seq: SeqId, sampling: SamplingParams) -> None:
        self.inner.add(seq, sampling)
        self.adds += 1
        self.live.add(seq)

    def release(self, seq: SeqId, freed: tuple[BlockId, ...]) -> None:
        self.inner.release(seq, freed)
        self.releases += 1
        self.live.discard(seq)

    def execute(self, plan: StepPlan) -> StepResult:
        result = self.inner.execute(plan)
        match plan:
            case PrefillPlan(seq=seq):
                self.steps.append(((seq,), result.tokens))
            case DecodePlan(seqs=seqs):
                self.steps.append((seqs, result.tokens))
        return result


def tokens_by_rid(
    events: list[ev.Event], steps: list[tuple[tuple[SeqId, ...], tuple[int, ...]]]
) -> dict[str, list[int]]:
    """Each request's sampled tokens: the n-th step event names the rows of the n-th step."""
    step_events = [e for e in events if isinstance(e, ev.Step)]
    assert len(step_events) == len(steps)
    out: dict[str, list[int]] = {}
    for event, (_, tokens) in zip(step_events, steps, strict=True):
        for rid, token in zip(event.rids, tokens, strict=True):
            out.setdefault(rid, []).append(token)
    return out
