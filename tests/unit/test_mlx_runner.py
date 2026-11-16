"""MLXRunner beyond the shared conformance tests (decision 0008): what it refuses in M1,
how it samples, what ``load_mlx`` reports, and what its cache and tokenizer accept."""

from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from ullm.runner.base import DecodePlan, PrefillPlan, RunnerInfo, SamplingParams

if TYPE_CHECKING:
    from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer

pytestmark = pytest.mark.mlx

GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)
QWEN3_VOCAB = 151_936  # the logits width; the tokenizer has 151,669 entries (roadmap section 4)


@pytest.mark.parametrize("temperature", [0.7, -1.0, float("nan")])
def test_anything_but_temperature_zero_is_refused_and_adds_nothing(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", temperature: float
) -> None:
    runner, _ = mlx_loaded
    with pytest.raises(ValueError, match="M1-T2"):
        runner.add(100, SamplingParams(temperature=temperature, top_p=1.0, top_k=0, seed=1))
    runner.add(100, GREEDY)
    runner.release(100)


def test_batched_decode_is_refused_without_failing_the_sequences(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    runner, _ = mlx_loaded
    for seq in (101, 102):
        runner.add(seq, GREEDY)
        runner.execute(PrefillPlan(seq=seq, token_ids=(9707, 11), start=0))
    with pytest.raises(ValueError, match="M2-T1"):
        runner.execute(DecodePlan(seqs=(101, 102), token_ids=(11, 11)))
    for seq in (101, 102):
        runner.execute(DecodePlan(seqs=(seq,), token_ids=(11,)))
        runner.release(seq)


def test_greedy_picks_the_argmax_of_log_probabilities_not_of_logits() -> None:
    import mlx.core as mx

    from ullm.runner.mlx_runner import MLXRunner

    # In bf16, subtracting logsumexp rounds both of these to -11.4375, and argmax
    # then takes the lower index, as mlx-lm's greedy sampler does.
    logits = mx.zeros((1, 1, QWEN3_VOCAB), dtype=mx.bfloat16)
    logits[0, 0, 5] = 0.5
    logits[0, 0, 9] = 0.51

    class Stub:
        layers: ClassVar[list[Any]] = []

        def __call__(self, inputs: Any, cache: Any) -> Any:
            return logits

    info = RunnerInfo("stub", QWEN3_VOCAB, frozenset(), max_model_len=8)
    runner = MLXRunner(Stub(), info)
    runner.add(1, GREEDY)
    assert runner.execute(PrefillPlan(seq=1, token_ids=(1,), start=0)).tokens == (5,)


@pytest.mark.parametrize(
    ("plan", "forwards"),
    [
        (PrefillPlan(seq=1, token_ids=(9707, 11, 1879), start=0, sample=False), 1),
        (PrefillPlan(seq=1, token_ids=(9707, 11, 1879), start=0), 2),
        (PrefillPlan(seq=1, token_ids=(9707,), start=0), 1),
        (DecodePlan(seqs=(1,), token_ids=(9707,)), 1),
    ],
    ids=["unsampled", "sampled", "one-token", "decode"],
)
def test_only_a_sampled_multi_token_prefill_splits_its_forward(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", plan: PrefillPlan | DecodePlan, forwards: int
) -> None:
    from ullm.runner.mlx_runner import MLXRunner

    loaded, _ = mlx_loaded
    calls: list[int] = []

    class Counting:
        layers = loaded.model.layers

        def __call__(self, inputs: Any, cache: Any) -> Any:
            calls.append(inputs.shape[1])
            return loaded.model(inputs, cache=cache)

    runner = MLXRunner(Counting(), loaded.info())
    runner.add(1, GREEDY)
    runner.execute(plan)
    assert len(calls) == forwards
    assert sum(calls) == len(plan.token_ids)


def test_load_reports_the_models_own_limits(mlx_loaded: "tuple[MLXRunner, MLXTokenizer]") -> None:
    runner, _ = mlx_loaded
    assert runner.info() == RunnerInfo(
        model_id="qwen3-0.6b-bf16",
        vocab_size=QWEN3_VOCAB,
        eos_token_ids=frozenset({151645}),
        max_model_len=40_960,  # max_position_embeddings in the pinned config.json
    )


@pytest.mark.parametrize(
    "key",
    [
        "tokenize",
        "truncation",
        "max_length",
        "padding",
        "tokenizer_kwargs",
        "sanitize_control_tokens",
        "return_assistant_tokens_mask",
        "return_tensors",
        "return_dict",
        "add_generation_prompt",
        "continue_final_message",
        "chat_template",
    ],
)
def test_render_refuses_options_that_change_what_it_returns(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", key: str
) -> None:
    _, tokenizer = mlx_loaded
    with pytest.raises(ValueError, match=key):
        tokenizer.render([{"role": "user", "content": "hi"}], {key: False})


def test_the_cache_asks_for_no_mask_on_one_query_and_causal_otherwise() -> None:
    from ullm.runner.kv.contiguous import ContiguousKV

    kv = ContiguousKV()
    assert kv.make_mask(1) is None
    assert kv.make_mask(5) == "causal"
    with pytest.raises(ValueError, match="causal"):
        kv.make_mask(5, window_size=4)
    with pytest.raises(ValueError, match="causal"):
        kv.make_mask(5, return_array=True)
