"""FakeRunner: deterministic, batch-independent tokens and a latency model (E1, decision 0008)."""

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ullm.runner.base import DecodePlan, PrefillPlan, RunnerInfo, SamplingParams
from ullm.runner.fake_runner import FakeDetokenizer, FakeRunner, FakeTokenizer, Latency

INFO = RunnerInfo(
    model_id="fake", vocab_size=50, eos_token_ids=frozenset({0, 7, 49}), max_model_len=4096
)
LATENCY = Latency(
    prefill_base_ms=5.0, prefill_per_token_ms=0.5, decode_base_ms=10.0, decode_per_row_ms=2.0
)


def _sampling(seed: int | None) -> SamplingParams:
    return SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=seed)


def _generate(runner: FakeRunner, seq: int, prompt: tuple[int, ...], n: int) -> list[int]:
    out = list(runner.execute(PrefillPlan(seq=seq, token_ids=prompt, start=0)).tokens)
    while len(out) < n:
        out += runner.execute(DecodePlan(seqs=(seq,), token_ids=(out[-1],))).tokens
    return out


def _runner(sleeps: list[float] | None = None) -> FakeRunner:
    return FakeRunner(
        INFO, LATENCY, sleep=(sleeps.append if sleeps is not None else lambda s: None)
    )


def test_the_same_seed_and_sequence_give_the_same_tokens() -> None:
    a, b = _runner(), _runner()
    for runner in (a, b):
        runner.add(3, _sampling(42))
    assert _generate(a, 3, (1, 2, 3), 20) == _generate(b, 3, (1, 2, 3), 20)


def _alone(seq: int, seed: int | None, n: int = 20) -> list[int]:
    runner = _runner()
    runner.add(seq, _sampling(seed))
    return _generate(runner, seq, (1, 2, 3), n)


def test_the_seed_alone_changes_the_tokens() -> None:
    assert _alone(1, 42) != _alone(1, 43)
    assert _alone(1, None) == _alone(1, 0)


def test_the_sequence_id_alone_changes_the_tokens() -> None:
    assert _alone(1, 42) != _alone(2, 42)


def test_a_batch_returns_each_rows_own_tokens_in_plan_order() -> None:
    expected = {seq: _alone(seq, None, 10) for seq in (1, 2, 3)}

    batched = _runner()
    for seq in (1, 2, 3):
        batched.add(seq, _sampling(None))
    got = {
        seq: list(batched.execute(PrefillPlan(seq=seq, token_ids=(1, 2, 3), start=0)).tokens)
        for seq in (1, 2, 3)
    }
    order = (3, 1, 2)
    for _ in range(9):
        step = batched.execute(DecodePlan(seqs=order, token_ids=tuple(got[s][-1] for s in order)))
        for seq, token in zip(order, step.tokens, strict=True):
            got[seq].append(token)
    assert got == expected


@pytest.mark.parametrize(
    "eos",
    [frozenset({3, 9}), frozenset({9, 3, 0}), frozenset({5, 70})],
    ids=["two", "unsorted", "partly-outside"],
)
def test_every_non_eos_id_is_reachable_and_eos_never_is(eos: frozenset[int]) -> None:
    info = RunnerInfo(model_id="f", vocab_size=12, eos_token_ids=eos, max_model_len=4096)
    runner = FakeRunner(info, LATENCY, sleep=lambda s: None)
    seen: set[int] = set()
    for seq in range(200):
        runner.add(seq, _sampling(None))
        seen |= set(_generate(runner, seq, (1,), 8))
        runner.release(seq)  # a decode names every decoding sequence (0019)
    assert seen == set(range(12)) - eos


def test_a_refused_add_leaves_nothing_behind() -> None:
    runner = _runner()
    not_sampling: Any = None
    with pytest.raises(AttributeError):
        runner.add(1, not_sampling)
    runner.add(1, _sampling(None))
    runner.execute(PrefillPlan(seq=1, token_ids=(1,), start=0))


@given(
    seed=st.integers(min_value=0, max_value=2**32), seq=st.integers(0, 1000), n=st.integers(1, 64)
)
def test_tokens_are_in_the_vocabulary_and_never_eos(seed: int, seq: int, n: int) -> None:
    runner = _runner()
    runner.add(seq, _sampling(seed))
    for token in _generate(runner, seq, (1,), n):
        assert 0 <= token < INFO.vocab_size
        assert token not in INFO.eos_token_ids


def test_steps_sleep_and_report_the_modelled_time() -> None:
    sleeps: list[float] = []
    runner = _runner(sleeps)
    for seq in (1, 2):
        runner.add(seq, _sampling(None))
    prefill = runner.execute(PrefillPlan(seq=1, token_ids=(1,) * 10, start=0, sample=False))
    runner.execute(PrefillPlan(seq=2, token_ids=(1,), start=0))
    decode = runner.execute(DecodePlan(seqs=(1, 2), token_ids=(3, 4)))
    assert (prefill.forward_ms, prefill.total_ms) == (10.0, 10.0)
    assert (decode.forward_ms, decode.total_ms) == (14.0, 14.0)
    assert sleeps == pytest.approx([0.010, 0.0055, 0.014])


def test_a_token_is_keyed_by_its_position_however_it_was_reached() -> None:
    runner = _runner()
    runner.add(1, _sampling(None))
    prefilled = runner.execute(PrefillPlan(seq=1, token_ids=(1,) * 6, start=0)).tokens
    runner.release(1)
    runner.add(1, _sampling(None))
    runner.execute(PrefillPlan(seq=1, token_ids=(1,) * 5, start=0, sample=False))
    decoded = runner.execute(DecodePlan(seqs=(1,), token_ids=(1,))).tokens
    assert decoded == prefilled


def test_a_vocabulary_of_only_eos_is_refused() -> None:
    with pytest.raises(ValueError, match="EOS"):
        FakeRunner(
            RunnerInfo(
                model_id="f", vocab_size=2, eos_token_ids=frozenset({0, 1}), max_model_len=8
            ),
            LATENCY,
        )


def test_the_fake_detokenizer_spells_each_token_and_holds_nothing_back() -> None:
    detokenizer = FakeDetokenizer()
    assert [detokenizer.add(t) for t in (3, 0, 41)] == [" t3", " t0", " t41"]
    assert detokenizer.flush() == ""


def test_fake_tokenizer_spells_each_byte_as_one_token_and_never_eos() -> None:
    tokenizer = FakeTokenizer()
    ids = tokenizer.render(
        [{"role": "system", "content": "be"}, {"role": "user", "content": "é"}], {}
    )
    assert ids == tuple(b + 1 for b in "system\nbe\nuser\né\n".encode())
    assert tokenizer.render([{"role": "user", "content": ""}], {"enable_thinking": True}) == tuple(
        b + 1 for b in b"user\n\n"
    )
    assert isinstance(tokenizer.detokenizer(), FakeDetokenizer)
