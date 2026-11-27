"""C3 v2: every runner keeps the protocol's invariants (decision 0008).

The same checks run on ``FakeRunner`` everywhere and on ``MLXRunner`` where
MLX and the model are present, so the core can be tested on the fake and trust
the real one. Each runner is built with ``max_model_len = 8`` so the length
limit is reachable without a long prompt, and with 4-token blocks so a
sequence crosses a block boundary.
"""

import dataclasses
from collections.abc import Callable
from typing import Any

import pytest
from tests.plans import Plans

from ullm.runner.base import DecodePlan, PrefillPlan, Runner, RunnerInfo, SamplingParams
from ullm.runner.fake_runner import FakeRunner, Latency

MAX_LEN = 8
GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)
INFO = RunnerInfo(
    model_id="fake",
    vocab_size=1000,
    eos_token_ids=frozenset({999}),
    max_model_len=MAX_LEN,
    block_size=4,
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
    info = dataclasses.replace(loaded.info(), max_model_len=MAX_LEN, block_size=INFO.block_size)
    return MLXRunner(model, info), model


@pytest.fixture
def plans() -> Plans:
    return Plans(block_size=INFO.block_size)


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
def test_sequence_ids_must_be_plain_ints(runner: Runner, bad: Any, plans: Plans) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="int"):
        runner.add(bad, GREEDY)
    with pytest.raises(ValueError, match="int"):
        runner.release(bad, plans.release(bad))
    runner.execute(plans.prefill(seq=1, token_ids=(5,), start=0))


def test_an_unknown_sequence_cannot_run_or_be_released(runner: Runner, plans: Plans) -> None:
    with pytest.raises(ValueError, match="unknown"):
        runner.execute(plans.prefill(seq=7, token_ids=(1, 2), start=0))
    with pytest.raises(ValueError, match="unknown"):
        runner.execute(plans.decode(seqs=(7,), token_ids=(1,)))
    with pytest.raises(ValueError, match="unknown"):
        runner.release(7, plans.release(7))


def test_a_released_sequence_is_gone_and_its_id_reusable(runner: Runner, plans: Plans) -> None:
    runner.add(1, GREEDY)
    first = runner.execute(plans.prefill(seq=1, token_ids=(5, 6, 7), start=0))
    runner.release(1, plans.release(1))
    with pytest.raises(ValueError, match="unknown"):
        runner.execute(plans.decode(seqs=(1,), token_ids=(5,)))
    with pytest.raises(ValueError, match="unknown"):
        runner.release(1, plans.release(1))
    runner.add(1, GREEDY)
    assert runner.execute(plans.prefill(seq=1, token_ids=(5, 6, 7), start=0)).tokens == first.tokens


# Invariant 2: contiguous positions and limits


def test_prefill_must_start_where_the_sequence_ends(runner: Runner, plans: Plans) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="start"):
        runner.execute(plans.prefill(seq=1, token_ids=(5, 6), start=1))
    runner.execute(plans.prefill(seq=1, token_ids=(5, 6), start=0, sample=False))
    with pytest.raises(ValueError, match="start"):
        runner.execute(plans.prefill(seq=1, token_ids=(7,), start=0))
    runner.execute(plans.prefill(seq=1, token_ids=(7,), start=2))


def test_a_sequence_cannot_grow_past_max_model_len(runner: Runner, plans: Plans) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="max_model_len"):
        runner.execute(plans.prefill(seq=1, token_ids=(5,) * (MAX_LEN + 1), start=0))
    step = runner.execute(plans.prefill(seq=1, token_ids=(5,) * (MAX_LEN - 1), start=0))
    step = runner.execute(plans.decode(seqs=(1,), token_ids=step.tokens))
    with pytest.raises(ValueError, match="max_model_len"):
        runner.execute(plans.decode(seqs=(1,), token_ids=step.tokens))


@pytest.mark.parametrize("bad", [-1, "vocab"])
def test_token_ids_must_be_in_the_vocabulary(runner: Runner, bad: int | str, plans: Plans) -> None:
    token = _vocab(runner) if bad == "vocab" else int(bad)
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="vocab"):
        runner.execute(plans.prefill(seq=1, token_ids=(5, token), start=0))
    runner.execute(plans.prefill(seq=1, token_ids=(5,), start=0))
    with pytest.raises(ValueError, match="vocab"):
        runner.execute(plans.decode(seqs=(1,), token_ids=(token,)))


def test_a_rejected_plan_leaves_the_sequence_intact(runner: Runner, plans: Plans) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match="vocab"):
        runner.execute(plans.prefill(seq=1, token_ids=(5, -1), start=0))
    after_rejection = runner.execute(plans.prefill(seq=1, token_ids=(5, 6), start=0))
    runner.release(1, plans.release(1))
    runner.add(1, GREEDY)
    fresh = runner.execute(plans.prefill(seq=1, token_ids=(5, 6), start=0))
    assert after_rejection.tokens == fresh.tokens


# Invariant 3: one token per sampled row


def test_prefill_samples_one_token_unless_told_not_to(runner: Runner, plans: Plans) -> None:
    runner.add(1, GREEDY)
    assert (
        runner.execute(plans.prefill(seq=1, token_ids=(5, 6), start=0, sample=False)).tokens == ()
    )
    (token,) = runner.execute(plans.prefill(seq=1, token_ids=(7,), start=2)).tokens
    assert 0 <= token < _vocab(runner)
    (token,) = runner.execute(plans.decode(seqs=(1,), token_ids=(token,))).tokens
    assert 0 <= token < _vocab(runner)


def test_a_chunked_prefill_samples_what_one_prefill_does(runner: Runner, plans: Plans) -> None:
    runner.add(1, GREEDY)
    runner.execute(plans.prefill(seq=1, token_ids=(5, 6), start=0, sample=False))
    chunked = runner.execute(plans.prefill(seq=1, token_ids=(7,), start=2))
    runner.release(1, plans.release(1))
    runner.add(1, GREEDY)
    whole = runner.execute(plans.prefill(seq=1, token_ids=(5, 6, 7), start=0))
    assert chunked.tokens == whole.tokens


def test_timings_are_reported(runner: Runner, plans: Plans) -> None:
    runner.add(1, GREEDY)
    step = runner.execute(plans.prefill(seq=1, token_ids=(5, 6, 7), start=0))
    assert 0.0 <= step.forward_ms <= step.total_ms


# Invariant 4: a failed step fails its sequences


def test_after_a_failed_step_only_release_is_legal(
    built: tuple[Runner, Breakable], plans: Plans
) -> None:
    runner, breakable = built
    runner.add(1, GREEDY)
    runner.add(2, GREEDY)
    runner.execute(plans.prefill(seq=1, token_ids=(5, 6), start=0))
    breakable.armed = True
    with pytest.raises(RuntimeError, match="injected"):
        runner.execute(plans.decode(seqs=(1,), token_ids=(5,)))
    with pytest.raises(ValueError, match="failed"):
        runner.execute(plans.decode(seqs=(1,), token_ids=(5,)))
    with pytest.raises(ValueError, match="failed"):
        runner.execute(plans.prefill(seq=1, token_ids=(5,), start=2))
    runner.execute(plans.prefill(seq=2, token_ids=(5, 6), start=0))
    runner.release(1, plans.release(1))
    runner.add(1, GREEDY)
    runner.execute(plans.prefill(seq=1, token_ids=(5, 6), start=0))


def test_a_failed_batch_fails_every_sequence_in_it(
    built: tuple[Runner, Breakable], plans: Plans
) -> None:
    runner, breakable = built
    for seq in (1, 2, 3):
        runner.add(seq, GREEDY)
        runner.execute(plans.prefill(seq=seq, token_ids=(5,), start=0))
    breakable.armed = True
    with pytest.raises(RuntimeError, match="injected"):
        runner.execute(plans.decode(seqs=(1, 2), token_ids=(5, 5)))
    for seq in (1, 2):
        with pytest.raises(ValueError, match="failed"):
            runner.execute(plans.decode(seqs=(seq,), token_ids=(5,)))
    runner.execute(plans.decode(seqs=(3,), token_ids=(5,)))


def test_a_failed_prefill_leaves_the_decoding_batch_intact(
    built: tuple[Runner, Breakable], plans: Plans
) -> None:
    runner, breakable = built

    def decode_seq_1(fail_a_prefill: bool) -> list[int]:
        runner.add(1, GREEDY)
        tokens = list(runner.execute(plans.prefill(seq=1, token_ids=(5, 6, 7), start=0)).tokens)
        tokens += runner.execute(plans.decode(seqs=(1,), token_ids=(tokens[-1],))).tokens
        if fail_a_prefill:
            runner.add(2, GREEDY)
            breakable.armed = True
            with pytest.raises(RuntimeError, match="injected"):
                runner.execute(plans.prefill(seq=2, token_ids=(5, 6), start=0))
            runner.release(2, plans.release(2))
        for _ in range(3):
            tokens += runner.execute(plans.decode(seqs=(1,), token_ids=(tokens[-1],))).tokens
        runner.release(1, plans.release(1))
        return tokens

    assert decode_seq_1(fail_a_prefill=True) == decode_seq_1(fail_a_prefill=False)


# Invariant 2: a batch decodes together (decision 0019)


def _decoding(runner: Runner, plans: Plans, *seqs: int) -> dict[int, int]:
    """Adds, prefills and decodes each sequence once, together; returns each last token."""
    for seq in seqs:
        runner.add(seq, GREEDY)
        runner.execute(plans.prefill(seq=seq, token_ids=(5, 6), start=0))
    step = runner.execute(plans.decode(seqs=seqs, token_ids=(5,) * len(seqs)))
    return dict(zip(seqs, step.tokens, strict=True))


def test_a_decode_must_name_every_decoding_sequence(runner: Runner, plans: Plans) -> None:
    last = _decoding(runner, plans, 1, 2)
    with pytest.raises(ValueError, match=r"omits decoding sequences \[2\]"):
        runner.execute(plans.decode(seqs=(1,), token_ids=(last[1],)))
    plans.undo()
    runner.execute(plans.decode(seqs=(2, 1), token_ids=(last[2], last[1])))
    runner.release(2, plans.release(2))
    runner.execute(plans.decode(seqs=(1,), token_ids=(last[1],)))


def test_a_prefill_comes_before_the_first_decode(runner: Runner, plans: Plans) -> None:
    last = _decoding(runner, plans, 1)
    with pytest.raises(ValueError, match="sequence 1 is decoding"):
        runner.execute(plans.prefill(seq=1, token_ids=(5,), start=3))
    plans.undo()
    runner.execute(plans.decode(seqs=(1,), token_ids=(last[1],)))


def test_a_decode_may_add_prefilling_sequences_to_the_batch(runner: Runner, plans: Plans) -> None:
    last = _decoding(runner, plans, 1)
    runner.add(2, GREEDY)
    runner.add(3, GREEDY)
    runner.execute(plans.prefill(seq=2, token_ids=(5, 6, 7), start=0))
    runner.execute(plans.prefill(seq=3, token_ids=(5,), start=0))
    step = runner.execute(plans.decode(seqs=(2, 1), token_ids=(5, last[1])))
    assert len(step.tokens) == 2
    with pytest.raises(ValueError, match=r"omits decoding sequences \[2\]"):
        runner.execute(plans.decode(seqs=(3, 1), token_ids=(5, 5)))
    plans.undo()
    runner.execute(plans.decode(seqs=(3, 2, 1), token_ids=(5, 5, 5)))


# Invariant 2: each plan carries each sequence's whole block table (decision 0034)


def test_a_prefill_names_one_block_per_block_size_tokens(runner: Runner, plans: Plans) -> None:
    runner.add(1, GREEDY)
    for wrong in ((), (0,), (0, 1, 2)):
        with pytest.raises(ValueError, match="block"):
            runner.execute(PrefillPlan(seq=1, token_ids=(5,) * 5, start=0, blocks=wrong))
    runner.execute(PrefillPlan(seq=1, token_ids=(5,) * 5, start=0, blocks=(0, 1)))


def test_a_decode_names_each_rows_table_after_the_step(runner: Runner, plans: Plans) -> None:
    runner.add(1, GREEDY)
    (token,) = runner.execute(plans.prefill(seq=1, token_ids=(5, 6, 7), start=0)).tokens
    (first,) = plans.table(1)
    with pytest.raises(ValueError, match="block"):
        runner.execute(DecodePlan(seqs=(1,), token_ids=(token,), blocks=((first, 9),)))
    runner.execute(DecodePlan(seqs=(1,), token_ids=(token,), blocks=((first,),)))
    with pytest.raises(ValueError, match="block"):
        runner.execute(DecodePlan(seqs=(1,), token_ids=(token,), blocks=((first,),)))
    runner.execute(DecodePlan(seqs=(1,), token_ids=(token,), blocks=((first, 9),)))


@pytest.mark.parametrize("bad", [[0], (True,), ("0",), None])
def test_release_takes_the_freed_blocks_as_a_tuple_of_ints(runner: Runner, bad: Any) -> None:
    runner.add(1, GREEDY)
    with pytest.raises(ValueError, match=r"tuple|int"):
        runner.release(1, bad)
    runner.release(1, ())
