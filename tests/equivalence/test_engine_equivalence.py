"""Rule 1 through the loop: greedy replies via ``EngineClient`` and ``MLXRunner`` give exactly
mlx-lm's token IDs, on this machine (decision 0010). The model loads on the test thread and
runs on the engine thread, as the server will run it."""

import asyncio
import json
import os
from typing import TYPE_CHECKING, Any

import pytest
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


def test_greedy_tokens_through_the_engine_equal_mlx_lm(
    mlx_loaded: "tuple[MLXRunner, MLXTokenizer]",
) -> None:
    from scripts.make_goldens import golden_path, reference

    runner, tokenizer = mlx_loaded
    cases: list[dict[str, Any]] = json.loads(golden_path(DEFAULT_MODEL).read_text())["cases"]
    recording = RecordingRunner(runner)
    events: list[ev.Event] = []

    async def main() -> list[tuple[str, list[Output]]]:
        engine = EngineClient.start(recording, tokenizer.detokenizer, EngineConfig(), events.append)
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
    sampled = tokens_by_rid(events, recording.steps)
    failures = []
    for (rid, _), case in zip(replies, cases, strict=True):
        want, _, _ = reference(runner.model, tokenizer.wrapped, case["prompt_ids"], TOKENS)
        if sampled[rid] != want:
            failures.append(f"{case['id']} ({rid}): {sampled[rid]} != {want}")
    assert not failures, "the engine differs from mlx-lm:\n" + "\n".join(failures)
    assert check_trace(frames_from(events, "r1")).problems == []
    assert recording.adds == recording.releases == len(cases)
