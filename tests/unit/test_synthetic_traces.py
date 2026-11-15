"""The committed synthetic traces: consistent, deterministic, fresh, and labelled."""

import json
from collections import Counter
from pathlib import Path

import jsonschema
import pytest

from ullm.telemetry import events as ev
from ullm.telemetry import trace
from ullm.telemetry.check import check_trace
from ullm.telemetry.export import render_schema
from ullm.telemetry.synthetic import SCENARIOS, generate

FIXTURES = Path(__file__).resolve().parents[2] / "terminal" / "fixtures" / "telemetry"
NAMES = sorted(SCENARIOS)


@pytest.fixture(scope="module")
def generated() -> dict[str, list[ev.Frame]]:
    return {name: generate(SCENARIOS[name]) for name in NAMES}


@pytest.mark.parametrize("name", NAMES)
def test_trace_tells_a_consistent_story(name: str, generated: dict[str, list[ev.Frame]]) -> None:
    report = check_trace(generated[name])
    assert report.ok, report.problems[:10]
    assert report.requests > 50


@pytest.mark.parametrize("name", NAMES)
def test_generation_is_deterministic(name: str, generated: dict[str, list[ev.Frame]]) -> None:
    assert trace.dumps(generate(SCENARIOS[name])) == trace.dumps(generated[name])


@pytest.mark.parametrize("name", NAMES)
def test_committed_trace_is_fresh(name: str, generated: dict[str, list[ev.Frame]]) -> None:
    path = FIXTURES / f"{name}.jsonl.gz"
    assert trace.read_text(path) == trace.dumps(generated[name]), "run `make fixtures`"


@pytest.mark.parametrize("name", NAMES)
def test_committed_trace_matches_the_schema(name: str) -> None:
    validator = jsonschema.Draft202012Validator(json.loads(render_schema()))
    lines = trace.read_text(FIXTURES / f"{name}.jsonl.gz").splitlines()
    for line in lines:
        validator.validate(json.loads(line))
    assert [f.seq for f in trace.iter_frames(FIXTURES / f"{name}.jsonl.gz")] == list(
        range(len(lines))
    )


@pytest.mark.parametrize("name", NAMES)
def test_traces_say_they_are_synthetic(name: str, generated: dict[str, list[ev.Frame]]) -> None:
    hello = generated[name][0].events[0]
    assert isinstance(hello, ev.Hello)
    assert (hello.source, hello.model, hello.machine) == ("fixture", "synthetic", None)


def test_prefix_pressure_covers_what_the_kv_grid_needs(
    generated: dict[str, list[ev.Frame]],
) -> None:
    events = [e for f in generated["prefix-pressure"] for e in f.events]
    kinds = Counter(e.type for e in events)
    assert kinds["req.preempted"] > 0
    assert kinds["req.aborted"] > 0
    assert kinds["kv.snapshot"] > 100
    assert any(isinstance(e, ev.ReqAdmitted) and e.cached_tokens > 0 for e in events)
    states: set[int] = set()
    for e in events:
        if isinstance(e, ev.KvSnapshot):
            states.update(e.block_states())
    assert {ev.BLOCK_FREE, ev.BLOCK_CACHED, ev.BLOCK_SHARED, ev.BLOCK_OWNED_BASE} <= states
    steps = [e for e in events if isinstance(e, ev.Step)]
    assert max(len(s.rids) for s in steps) == SCENARIOS["prefix-pressure"].max_num_seqs
    assert max(s.tokens for s in steps if s.kind == "prefill") > 5000  # the long-prompt stall


def test_steady_chat_has_a_queue_and_no_blocks(generated: dict[str, list[ev.Frame]]) -> None:
    events = [e for f in generated["steady-chat"] for e in f.events]
    assert not any(isinstance(e, ev.KvSnapshot) for e in events)
    assert max(e.waiting for e in events if isinstance(e, ev.Step)) > 0
    hello = events[0]
    assert isinstance(hello, ev.Hello)
    assert hello.kv.num_blocks is None
