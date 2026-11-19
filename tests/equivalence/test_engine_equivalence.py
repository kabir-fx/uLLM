"""Rule 1 through the loop: greedy replies via ``EngineClient`` and ``MLXRunner`` against
mlx-lm, on this machine (decisions 0010 and 0021). The model loads on the test thread and
runs on the engine thread, as the server will run it.

At batch size 1 the tokens must be exactly mlx-lm's. Above 1 the batched forward runs at
another shape and can flip a near-tie, so each token is scored by mlx-lm on uLLM's own
prefix: it must be in mlx-lm's top 2 and within ``BATCH_NEAR_TIE`` of its best."""

import asyncio
import json
import os
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import pytest
from tests.equivalence.divergence import divergence
from tests.recording import RecordingRunner, tokens_by_rid

from ullm.core.client import EngineClient, RequestStream
from ullm.core.engine import EngineConfig, Finished, Output
from ullm.models import DEFAULT_MODEL
from ullm.runner.base import SamplingParams
from ullm.telemetry import events as ev
from ullm.telemetry.check import check_trace
from ullm.telemetry.trace import frames_from

if TYPE_CHECKING:
    from ullm.runner.mlx_runner import MLXRunner, MLXTokenizer

pytestmark = pytest.mark.mlx

pytest.importorskip("mlx.core")

TOKENS = int(os.environ.get("ULLM_GOLDEN_TOKENS", "32"))  # as test_runner_equivalence.py
GREEDY = SamplingParams(temperature=0.0, top_p=1.0, top_k=0, seed=None)
BATCH_NEAR_TIE = 0.25  # logprob; the worst measured at batch 8 and 128 tokens (decision 0021)


def _forced_misses(model: Any, prompt: list[int], tokens: Sequence[int]) -> list[str]:
    """mlx-lm's logprobs at each of ``tokens``, fed ``prompt`` and then ``tokens`` itself as
    ``generate_step`` feeds them, so a prefix that matches mlx-lm is scored with its own
    arithmetic. Names each token outside the top 2 or more than ``BATCH_NEAR_TIE`` behind."""
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache

    assert len(prompt) <= 2048, "generate_step prefills a longer prompt in chunks"
    cache = make_prompt_cache(model)
    model(mx.array(prompt[:-1])[None], cache=cache)
    misses = []
    for i, (fed, got) in enumerate(zip((prompt[-1], *tokens[:-1]), tokens, strict=True)):
        logits = model(mx.array([[fed]]), cache=cache)[:, -1, :]
        logprobs = (logits - mx.logsumexp(logits, keepdims=True)).squeeze(0).astype(mx.float32)
        gap = float(logprobs.max() - logprobs[got])
        above = int((logprobs > logprobs[got]).sum())
        if above > 1 or gap > BATCH_NEAR_TIE:
            misses.append(f"token {i}: {got} is {gap} behind, {above} tokens above it")
    return misses


@pytest.mark.parametrize("max_num_seqs", [1, 2, 4, 8])
def test_greedy_tokens_through_the_engine_match_mlx_lm_at_each_batch_size(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]", max_num_seqs: int
) -> None:
    from scripts.make_goldens import golden_path, reference

    runner, tokenizer = mlx_loaded
    cases: list[dict[str, Any]] = json.loads(golden_path(DEFAULT_MODEL).read_text())["cases"]
    recording = RecordingRunner(runner)
    events: list[ev.Event] = []

    async def main() -> list[tuple[str, list[Output]]]:
        config = EngineConfig(max_num_seqs=max_num_seqs)
        engine = EngineClient.start(recording, tokenizer.detokenizer, config, events.append)
        streams = [
            engine.submit(tuple(case["prompt_ids"]), GREEDY, max_tokens=TOKENS) for case in cases
        ]

        async def drain(stream: RequestStream) -> tuple[str, list[Output]]:
            async with stream:
                return stream.rid, [out async for out in stream]

        replies = await asyncio.gather(*(drain(s) for s in streams))
        await engine.close()
        return replies

    replies = asyncio.run(main())
    assert all(isinstance(outputs[-1], Finished) for _, outputs in replies)
    assert max(len(seqs) for seqs, _ in recording.steps) == max_num_seqs
    sampled = tokens_by_rid(events, recording.steps)
    eos = runner.info().eos_token_ids
    failures = []
    for (rid, _), case in zip(replies, cases, strict=True):
        got, name = sampled[rid], f"{case['id']} ({rid})"
        assert len(got) == TOKENS or got[-1] in eos, f"{name} stopped early"
        if max_num_seqs == 1:
            want, margins, _ = reference(
                runner.model, tokenizer.wrapped, case["prompt_ids"], TOKENS
            )
            if got != want:
                failures.append(divergence(name, got, want, margins))
        else:
            failures += [
                f"{name}, {m}" for m in _forced_misses(runner.model, case["prompt_ids"], got)
            ]
    assert not failures, "the engine differs from mlx-lm:\n" + "\n".join(failures)
    assert check_trace(frames_from(events, "r1")).problems == []
    assert recording.adds == recording.releases == len(cases)
