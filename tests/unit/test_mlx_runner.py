"""MLXRunner beyond the shared conformance tests (decision 0008): what it refuses in M1,
how it samples, what ``load_mlx`` reports, and what its cache and tokenizer accept."""

import dataclasses
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import pytest

from ullm.runner.base import DecodePlan, PrefillPlan, RunnerInfo, SamplingParams

if TYPE_CHECKING:
    from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer

pytestmark = pytest.mark.mlx

GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)
QWEN3_VOCAB = 151_936  # the logits width; the tokenizer has 151,669 entries (roadmap section 4)


# "Write a poem." through the chat template, thinking off.
PROMPT = (
    151644,
    872,
    198,
    7985,
    264,
    32794,
    13,
    151645,
    198,
    151644,
    77091,
    198,
    151667,
    271,
    151668,
    271,
)


def _generate(runner: "MLXRunner", seq: int, sampling: SamplingParams, n: int = 24) -> list[int]:
    runner.add(seq, sampling)
    tokens = list(runner.execute(PrefillPlan(seq=seq, token_ids=PROMPT, start=0)).tokens)
    while len(tokens) < n:
        tokens += runner.execute(DecodePlan(seqs=(seq,), token_ids=(tokens[-1],))).tokens
    runner.release(seq)
    return tokens


def test_a_seed_reproduces_a_sampled_reply_under_any_sequence_id(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    runner, _ = mlx_loaded
    hot = SamplingParams(temperature=1.5, top_p=0.95, top_k=50, seed=1234)
    first = _generate(runner, 100, hot)
    assert _generate(runner, 101, hot) == first
    assert _generate(runner, 100, dataclasses.replace(hot, seed=1235)) != first


def test_without_a_seed_each_sequence_draws_its_own(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    runner, _ = mlx_loaded
    unseeded = SamplingParams(temperature=2.0, top_p=1.0, top_k=0, seed=None)
    assert _generate(runner, 100, unseeded) != _generate(runner, 101, unseeded)


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
        "conversation",
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


def test_render_refuses_a_batch_of_conversations(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    _, tokenizer = mlx_loaded
    batch: Any = [[{"role": "user", "content": "hi"}], [{"role": "user", "content": "yo"}]]
    with pytest.raises(ValueError, match="one conversation"):
        tokenizer.render(batch, {})


@pytest.mark.parametrize("sample", [True, False], ids=["sampled", "unsampled"])
def test_a_failure_inside_the_eval_fails_the_sequence(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", monkeypatch: pytest.MonkeyPatch, sample: bool
) -> None:
    import mlx.core as mx

    from ullm.runner.mlx_runner import MLXRunner

    loaded, _ = mlx_loaded
    runner = MLXRunner(loaded.model, loaded.info())
    runner.add(1, GREEDY)
    real_eval = mx.eval

    def fail_once(*args: Any) -> None:
        monkeypatch.setattr(mx, "eval", real_eval)
        raise RuntimeError("injected eval failure")

    monkeypatch.setattr(mx, "eval", fail_once)
    with pytest.raises(RuntimeError, match="injected"):
        runner.execute(PrefillPlan(seq=1, token_ids=(9707, 11), start=0, sample=sample))
    with pytest.raises(ValueError, match="failed"):
        runner.execute(PrefillPlan(seq=1, token_ids=(9707,), start=0))
    runner.release(1)


def test_the_cache_asks_for_no_mask_on_one_query_and_causal_otherwise() -> None:
    from ullm.runner.kv.contiguous import ContiguousKV

    kv = ContiguousKV()
    assert kv.make_mask(1) is None
    assert kv.make_mask(5) == "causal"
    with pytest.raises(ValueError, match="causal"):
        kv.make_mask(5, window_size=4)
    with pytest.raises(ValueError, match="causal"):
        kv.make_mask(5, return_array=True)


@pytest.mark.parametrize(
    "text",
    [
        "Hello wörld 你好 🎉🎉 ok",
        " leading space",
        "a , b . c 's n't  x\n\n y",
        "<think>\n\n</think>\n\nHi",
    ],
    ids=["multi-byte", "leading-space", "spacing", "special"],
)
def test_streamed_text_is_the_tokenizers_decoding(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", text: str
) -> None:
    _, tokenizer = mlx_loaded
    ids = tokenizer.wrapped.encode(text, add_special_tokens=False)
    detokenizer = tokenizer.detokenizer()
    chunks = [detokenizer.add(i) for i in ids]
    assert "".join(chunks) + detokenizer.flush() == tokenizer.wrapped.decode(ids) == text
    assert not any("�" in chunk for chunk in chunks)


def test_an_emoji_split_across_tokens_streams_whole(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    _, tokenizer = mlx_loaded
    ids = tokenizer.wrapped.encode("🫠", add_special_tokens=False)
    assert len(ids) == 3  # one byte-level token per byte group, in Qwen3's vocabulary
    detokenizer = tokenizer.detokenizer()
    assert [detokenizer.add(i) for i in ids] == ["", "", "🫠"]


def test_a_stop_string_spanning_qwen3_tokens_is_cut_from_the_stream(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    from ullm.core.stop import StopCheck

    _, tokenizer = mlx_loaded
    ids = tokenizer.wrapped.encode("The answer is 42.\n\nUser: next? 🫠", add_special_tokens=False)
    stop = "\n\nUser:"
    pieces = [tokenizer.wrapped.decode([i]) for i in ids]
    assert not any(stop in p for p in pieces)  # it spans tokens
    check = StopCheck(tokenizer.detokenizer(), frozenset({151645}), max_tokens=64, stop=(stop,))
    chunks = [check.add(i) for i in ids[: pieces.index(":") + 1]]
    assert "".join(c.text for c in chunks) == "The answer is 42."
    assert [c.finish_reason for c in chunks][-1] == "stop"


def test_streamed_text_is_the_decoding_of_any_token_ids(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    import random

    runner, tokenizer = mlx_loaded
    rng = random.Random(0)
    special = sorted(tokenizer.wrapped.added_tokens_decoder)
    for _ in range(300):
        ids = [rng.randrange(runner.info().vocab_size) for _ in range(rng.randrange(1, 40))]
        ids += rng.sample(special, 2)  # ids past the tokenizer decode to nothing
        rng.shuffle(ids)
        detokenizer = tokenizer.detokenizer()
        streamed = "".join(detokenizer.add(i) for i in ids) + detokenizer.flush()
        assert streamed == tokenizer.wrapped.decode(ids)


def test_a_token_that_ends_one_character_and_starts_the_next_keeps_streaming(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    _, tokenizer = mlx_loaded
    lead = [tokenizer.wrapped.convert_tokens_to_ids("áŀ")]  # bytes E1 9E, as Qwen3 spells them
    detokenizer = tokenizer.detokenizer()
    streamed = "".join(detokenizer.add(i) for i in lead)
    for _ in range(1000):
        streamed += detokenizer.add(72496)  # bytes 92 E1 9E: finishes one ធ, starts another
    assert streamed == "ធ" * 1000


def test_only_a_byte_level_tokenizer_is_accepted() -> None:
    from tokenizers import decoders

    from ullm.runner.mlx_runner import MLXTokenizer

    word_piece = SimpleNamespace(backend_tokenizer=SimpleNamespace(decoder=decoders.WordPiece()))
    with pytest.raises(ValueError, match="ByteLevel"):
        MLXTokenizer(word_piece, vocab_size=10)
