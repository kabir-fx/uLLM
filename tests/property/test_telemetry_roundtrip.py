"""C2 telemetry: every frame survives the wire, and agrees with the exported schema.

The terminal and the trace files only ever see the JSON. These properties
check that what pydantic emits parses back to the same frame, is a fixpoint
under a second round trip, and validates against the JSON Schema that the
TypeScript types are generated from.
"""

import json
from functools import cache
from typing import Any

import jsonschema
from hypothesis import given
from hypothesis import strategies as st

from ullm.telemetry import events as ev

# Times are serialised to microseconds, so generate them at that precision.
ms = st.integers(min_value=0, max_value=10**10).map(lambda us: us / 1000)
count = st.integers(min_value=0, max_value=10**7)
positive = st.integers(min_value=1, max_value=10**7)
ident = st.text(alphabet="abcdefghijklmnopqrstuvwxyz0123456789-", min_size=1, max_size=24)
maybe = st.none().__or__

hello = st.builds(
    ev.Hello,
    t=ms,
    replica=ident,
    model=ident,
    source=st.sampled_from(["engine", "sim", "fixture"]),
    version=ident,
    commit=maybe(ident),
    epoch_unix_ms=count,
    limits=st.builds(
        ev.Limits, max_num_seqs=positive, max_prefill_tokens=positive, max_model_len=positive
    ),
    kv=st.builds(
        ev.KvInfo, block_size=maybe(positive), num_blocks=maybe(count), capacity_tokens=maybe(count)
    ),
    machine=maybe(st.builds(ev.Machine, chip=ident, memory_gb=positive)),
)
arrived = st.builds(
    ev.ReqArrived, t=ms, rid=ident, prompt_tokens=count, max_tokens=positive, tenant=maybe(ident)
)
admitted = st.builds(ev.ReqAdmitted, t=ms, rid=ident, queue_ms=ms, cached_tokens=count)
first_token = st.builds(ev.ReqFirstToken, t=ms, rid=ident, ttft_ms=ms)
finished = st.builds(
    ev.ReqFinished,
    t=ms,
    rid=ident,
    reason=st.sampled_from(["stop", "length", "tool_calls"]),
    usage=st.builds(
        ev.TokenUsage, prompt_tokens=count, completion_tokens=count, cached_tokens=count
    ),
    timings=st.builds(ev.Timings, queue_ms=ms, ttft_ms=ms, tpot_ms=maybe(ms), e2e_ms=ms),
)
aborted = st.builds(
    ev.ReqAborted,
    t=ms,
    rid=ident,
    reason=st.sampled_from(["client_disconnect", "shutdown", "error"]),
    completion_tokens=count,
)
preempted = st.builds(ev.ReqPreempted, t=ms, rid=ident, completion_tokens=count, blocks_freed=count)
step = st.builds(
    ev.Step,
    t=ms,
    n=count,
    kind=st.sampled_from(["prefill", "decode"]),
    rids=st.lists(ident, max_size=16),
    tokens=count,
    ms=ms,
    waiting=count,
    running=count,
    kv_tokens=count,
)


@st.composite
def kv_snapshot(draw: st.DrawFn) -> ev.KvSnapshot:
    owners = draw(st.lists(ident, max_size=16, unique=True))
    top = ev.BLOCK_OWNED_BASE + len(owners) - 1
    states = draw(st.lists(st.integers(0, top), max_size=2400))
    return ev.KvSnapshot.build(draw(ms), owners, bytes(states))


decision = st.builds(
    ev.RouterDecision,
    t=ms,
    rid=ident,
    replica=ident,
    kind=st.sampled_from(["prefix", "least_load", "round_robin"]),
    matched_messages=count,
    retry=st.booleans(),
)
event = st.one_of(
    hello,
    arrived,
    admitted,
    first_token,
    finished,
    aborted,
    preempted,
    step,
    kv_snapshot(),
    decision,
)
frame = st.builds(
    ev.Frame, replica=ident, seq=count, t=ms, dropped=count, events=st.lists(event, max_size=12)
)


@cache
def validator() -> Any:
    from ullm.telemetry.export import render_schema

    schema = json.loads(render_schema())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


@given(frame)
def test_frame_round_trips_exactly(f: ev.Frame) -> None:
    wire = f.model_dump_json()
    assert ev.Frame.model_validate_json(wire) == f
    assert ev.Frame.model_validate_json(wire).model_dump_json() == wire


@given(frame)
def test_emitted_json_matches_the_exported_schema(f: ev.Frame) -> None:
    validator().validate(json.loads(f.model_dump_json()))


@given(event)
def test_single_events_parse_through_the_union(e: ev.Event) -> None:
    assert ev.EVENT_ADAPTER.validate_json(e.model_dump_json()) == e


@given(kv_snapshot())
def test_kv_snapshot_states_round_trip(snap: ev.KvSnapshot) -> None:
    again = ev.KvSnapshot.build(snap.t, list(snap.owners), snap.block_states())
    assert again == snap
    assert len(snap.block_states()) == snap.num_blocks
