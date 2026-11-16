"""FakeRunner: deterministic, batch-independent tokens and a latency model (E1, decision 0008)."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from ullm.runner.base import DecodePlan, PrefillPlan, RunnerInfo, SamplingParams
from ullm.runner.fake_runner import FakeRunner, Latency

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


def test_seed_and_sequence_id_both_change_the_tokens() -> None:
    runner = _runner()
    runner.add(1, _sampling(42))
    runner.add(2, _sampling(42))
    runner.add(3, _sampling(43))
    outs = [_generate(runner, seq, (1, 2, 3), 20) for seq in (1, 2, 3)]
    assert outs[0] != outs[1]
    assert outs[0] != outs[2]


def test_tokens_do_not_depend_on_the_batch() -> None:
    alone = _runner()
    alone.add(1, _sampling(None))
    expected = _generate(alone, 1, (4, 5), 10)

    batched = _runner()
    for seq in (1, 2, 3):
        batched.add(seq, _sampling(None))
    last = {
        seq: batched.execute(PrefillPlan(seq=seq, token_ids=(4, 5), start=0)).tokens[0]
        for seq in (1, 2, 3)
    }
    got = [last[1]]
    for _ in range(9):
        step = batched.execute(DecodePlan(seqs=(3, 1, 2), token_ids=(last[3], last[1], last[2])))
        last = dict(zip((3, 1, 2), step.tokens, strict=True))
        got.append(last[1])
    assert got == expected


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
