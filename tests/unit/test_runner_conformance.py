"""C3 v1: every runner keeps the protocol's invariants (decision 0008).

The same checks run on ``FakeRunner`` everywhere and on ``MLXRunner`` where
MLX and the model are present, so the core can be tested on the fake and trust
the real one. Each runner is built with ``max_model_len = 8`` so the length
limit is reachable without a long prompt.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest

from ullm.runner.base import DecodePlan, PrefillPlan, Runner, RunnerInfo, SamplingParams
from ullm.runner.fake_runner import FakeRunner, Latency

MAX_LEN = 8
GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)
INFO = RunnerInfo(
    model_id="fake", vocab_size=1000, eos_token_ids=frozenset({999}), max_model_len=MAX_LEN
)
NO_LATENCY = Latency(
    prefill_base_ms=0.0, prefill_per_token_ms=0.0, decode_base_ms=0.0, decode_per_row_ms=0.0
)


class Breakable:
    """Wraps the callable a runner steps through; raises once when armed."""

    def __init__(self, inner: Callable[..., Any]) -> None:
        self.inner = inner
        self.armed = False

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        if self.armed:
            self.armed = False
            raise RuntimeError("injected failure")
        return self.inner(*args, **kwargs)


@pytest.fixture(params=["fake", pytest.param("mlx", marks=pytest.mark.mlx)])
def built(request: pytest.FixtureRequest) -> tuple[Runner, Breakable]:
    if request.param == "fake":
        sleep = Breakable(lambda seconds: None)
        return FakeRunner(INFO, NO_LATENCY, sleep=sleep), sleep
    from ullm.runner.mlx_runner import MLXRunner

    loaded, _ = request.getfixturevalue("mlx_loaded")
    model = Breakable(loaded.model)
    return MLXRunner(model, dataclasses.replace(loaded.info(), max_model_len=MAX_LEN)), model


@pytest.fixture
def runner(built: tuple[Runner, Breakable]) -> Runner:
    return built[0]


def _vocab(runner: Runner) -> int:
    return runner.info().vocab_size


# Invariant 1: lifecycle


def test_a_sequence_is_added_once(runner: Runner) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="already"):
        runner.add(1, GREEDY)


@pytest.mark.parametrize("bad", [True, 1.0, "1", None])
def test_sequence_ids_must_be_plain_ints(runner: Runner, bad: Any) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="int"):
        runner.add(bad, GREEDY)
    with pytest.raises(ValueError, match="int"):
        runner.release(bad)
    runner.execute(PrefillPlan(seq=1, token_ids=(5,), start=0))


def test_an_unknown_sequence_cannot_run_or_be_released(runner: Runner) -> None:
    with pytest.raises(ValueError, match="unknown"):
        runner.execute(PrefillPlan(seq=7, token_ids=(1, 2), start=0))
    with pytest.raises(ValueError, match="unknown"):
        runner.execute(DecodePlan(seqs=(7,), token_ids=(1,)))
    with pytest.raises(ValueError, match="unknown"):
        runner.release(7)


def test_a_released_sequence_is_gone_and_its_id_reusable(runner: Runner) -> None:
    runner.add(1, GREEDY)
    first = runner.execute(PrefillPlan(seq=1, token_ids=(5, 6, 7), start=0))
    runner.release(1)
    with pytest.raises(ValueError, match="unknown"):
        runner.execute(DecodePlan(seqs=(1,), token_ids=(5,)))
    with pytest.raises(ValueError, match="unknown"):
        runner.release(1)
    runner.add(1, GREEDY)
    assert runner.execute(PrefillPlan(seq=1, token_ids=(5, 6, 7), start=0)).tokens == first.tokens


# Invariant 2: contiguous positions and limits


def test_prefill_must_start_where_the_sequence_ends(runner: Runner) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="start"):
        runner.execute(PrefillPlan(seq=1, token_ids=(5, 6), start=1))
    runner.execute(PrefillPlan(seq=1, token_ids=(5, 6), start=0, sample=False))
    with pytest.raises(ValueError, match="start"):
        runner.execute(PrefillPlan(seq=1, token_ids=(7,), start=0))
    runner.execute(PrefillPlan(seq=1, token_ids=(7,), start=2))


def test_a_sequence_cannot_grow_past_max_model_len(runner: Runner) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="max_model_len"):
        runner.execute(PrefillPlan(seq=1, token_ids=(5,) * (MAX_LEN + 1), start=0))
    step = runner.execute(PrefillPlan(seq=1, token_ids=(5,) * (MAX_LEN - 1), start=0))
    step = runner.execute(DecodePlan(seqs=(1,), token_ids=step.tokens))
    with pytest.raises(ValueError, match="max_model_len"):
        runner.execute(DecodePlan(seqs=(1,), token_ids=step.tokens))


@pytest.mark.parametrize("bad", [-1, "vocab"])
def test_token_ids_must_be_in_the_vocabulary(runner: Runner, bad: int | str) -> None:
    token = _vocab(runner) if bad == "vocab" else int(bad)
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="vocab"):
        runner.execute(PrefillPlan(seq=1, token_ids=(5, token), start=0))
    runner.execute(PrefillPlan(seq=1, token_ids=(5,), start=0))
    with pytest.raises(ValueError, match="vocab"):
        runner.execute(DecodePlan(seqs=(1,), token_ids=(token,)))


def test_a_rejected_plan_leaves_the_sequence_intact(runner: Runner) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="vocab"):
        runner.execute(PrefillPlan(seq=1, token_ids=(5, -1), start=0))
    after_rejection = runner.execute(PrefillPlan(seq=1, token_ids=(5, 6), start=0))
    runner.release(1)
    runner.add(1, GREEDY)
    fresh = runner.execute(PrefillPlan(seq=1, token_ids=(5, 6), start=0))
    assert after_rejection.tokens == fresh.tokens


# Invariant 3: one token per sampled row


def test_prefill_samples_one_token_unless_told_not_to(runner: Runner) -> None:
    runner.add(1, GREEDY)
    assert runner.execute(PrefillPlan(seq=1, token_ids=(5, 6), start=0, sample=False)).tokens == ()
    (token,) = runner.execute(PrefillPlan(seq=1, token_ids=(7,), start=2)).tokens
    assert 0 <= token < _vocab(runner)
    (token,) = runner.execute(DecodePlan(seqs=(1,), token_ids=(token,))).tokens
    assert 0 <= token < _vocab(runner)


def test_a_chunked_prefill_samples_what_one_prefill_does(runner: Runner) -> None:
    runner.add(1, GREEDY)
    runner.execute(PrefillPlan(seq=1, token_ids=(5, 6), start=0, sample=False))
    chunked = runner.execute(PrefillPlan(seq=1, token_ids=(7,), start=2))
    runner.release(1)
    runner.add(1, GREEDY)
    whole = runner.execute(PrefillPlan(seq=1, token_ids=(5, 6, 7), start=0))
    assert chunked.tokens == whole.tokens


def test_timings_are_reported(runner: Runner) -> None:
    runner.add(1, GREEDY)
    step = runner.execute(PrefillPlan(seq=1, token_ids=(5, 6, 7), start=0))
    assert 0.0 <= step.forward_ms <= step.total_ms


# Invariant 4: a failed step fails its sequences


def test_after_a_failed_step_only_release_is_legal(built: tuple[Runner, Breakable]) -> None:
    runner, breakable = built
    runner.add(1, GREEDY)
    runner.add(2, GREEDY)
    runner.execute(PrefillPlan(seq=1, token_ids=(5, 6), start=0))
    breakable.armed = True
    with pytest.raises(RuntimeError, match="injected"):
        runner.execute(DecodePlan(seqs=(1,), token_ids=(5,)))
    with pytest.raises(ValueError, match="failed"):
        runner.execute(DecodePlan(seqs=(1,), token_ids=(5,)))
    with pytest.raises(ValueError, match="failed"):
        runner.execute(PrefillPlan(seq=1, token_ids=(5,), start=2))
    runner.execute(PrefillPlan(seq=2, token_ids=(5, 6), start=0))
    runner.release(1)
    runner.add(1, GREEDY)
    runner.execute(PrefillPlan(seq=1, token_ids=(5, 6), start=0))


def test_a_failed_batch_fails_every_sequence_in_it() -> None:
    sleep = Breakable(lambda seconds: None)
    runner = FakeRunner(INFO, NO_LATENCY, sleep=sleep)
    for seq in (1, 2, 3):
        runner.add(seq, GREEDY)
        runner.execute(PrefillPlan(seq=seq, token_ids=(5,), start=0))
    sleep.armed = True
    with pytest.raises(RuntimeError, match="injected"):
        runner.execute(DecodePlan(seqs=(1, 2), token_ids=(5, 5)))
    for seq in (1, 2):
        with pytest.raises(ValueError, match="failed"):
            runner.execute(DecodePlan(seqs=(seq,), token_ids=(5,)))
    runner.execute(DecodePlan(seqs=(3,), token_ids=(5,)))
