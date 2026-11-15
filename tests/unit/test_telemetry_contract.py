"""C2 telemetry: the rules in decision 0003 that are not round trips."""

import json
from typing import Any

import pytest
from pydantic import ValidationError

from ullm.telemetry import events as ev
from ullm.telemetry.export import SCHEMA_PATH, render_schema

HELLO: dict[str, Any] = {
    "type": "hello",
    "t": 0,
    "replica": "r1",
    "model": "m",
    "source": "fixture",
    "version": "0.0.0",
    "commit": None,
    "epoch_unix_ms": 0,
    "limits": {"max_num_seqs": 16, "max_prefill_tokens": 2048, "max_model_len": 4096},
    "kv": {"block_size": None, "num_blocks": None, "capacity_tokens": None},
    "machine": None,
}

# Every string field in the protocol. Telemetry must never carry prompt or
# output text (recorded runs are published), so a new free-text field has to
# be added here on purpose, with a reason.
ALLOWED_STRING_FIELDS = {
    "Frame": {"replica"},
    "Hello": {"type", "replica", "model", "source", "version", "commit"},
    "Machine": {"chip"},
    "ReqArrived": {"type", "rid", "tenant"},
    "ReqAdmitted": {"type", "rid"},
    "ReqFirstToken": {"type", "rid"},
    "ReqFinished": {"type", "rid", "reason"},
    "ReqAborted": {"type", "rid", "reason"},
    "ReqPreempted": {"type", "rid"},
    "Step": {"type", "kind", "rids"},
    "KvSnapshot": {"type", "owners", "states"},
    "RouterDecision": {"type", "rid", "replica", "kind"},
}


def _holds_strings(prop: dict[str, Any]) -> bool:
    if prop.get("type") == "string":
        return True
    if prop.get("type") == "array":
        return _holds_strings(prop.get("items", {}))
    return any(_holds_strings(option) for option in prop.get("anyOf", []))


def test_committed_schema_is_fresh() -> None:
    assert SCHEMA_PATH.read_text() == render_schema(), "run `make contracts`"


def test_no_free_text_fields() -> None:
    schema = json.loads(render_schema())
    models = {"Frame": schema, **schema["$defs"]}
    found = {
        name: {field for field, prop in model["properties"].items() if _holds_strings(prop)}
        for name, model in models.items()
    }
    found = {name: fields for name, fields in found.items() if fields}
    assert found == ALLOWED_STRING_FIELDS


def test_every_event_type_is_required_and_literal() -> None:
    schema = json.loads(render_schema())
    mapping = schema["properties"]["events"]["items"]["discriminator"]["mapping"]
    for type_name, ref in mapping.items():
        model = schema["$defs"][ref.rsplit("/", 1)[1]]
        assert "type" in model["required"]
        assert model["properties"]["type"]["const"] == type_name


def test_producers_reject_misspelt_fields() -> None:
    with pytest.raises(ValidationError, match="extra"):
        ev.EVENT_ADAPTER.validate_python({**HELLO, "replcia": "r2"})


def test_unknown_event_type_is_rejected_by_the_producer_model() -> None:
    # Readers (the terminal) must ignore unknown types; the Python models are
    # the producer side and stay strict.
    with pytest.raises(ValidationError):
        ev.EVENT_ADAPTER.validate_python({"type": "req.teleported", "t": 0, "rid": "x"})


def test_times_serialise_to_microseconds() -> None:
    step = ev.Step(
        t=12.3456789,
        n=0,
        kind="decode",
        rids=["a"],
        tokens=1,
        ms=0.0004,
        waiting=0,
        running=1,
        kv_tokens=10,
    )
    wire = json.loads(step.model_dump_json())
    assert (wire["t"], wire["ms"]) == (12.346, 0.0)
    assert step.t == 12.3456789  # in-process values are untouched


def test_negative_times_and_counts_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ev.ReqFirstToken(t=-1, rid="a", ttft_ms=1)
    with pytest.raises(ValidationError):
        ev.ReqArrived(t=0, rid="a", prompt_tokens=-1, max_tokens=1)


def test_kv_snapshot_encoding() -> None:
    states = bytes([ev.BLOCK_FREE, ev.BLOCK_CACHED, ev.BLOCK_SHARED, ev.BLOCK_OWNED_BASE + 1])
    snap = ev.KvSnapshot.build(5.0, ["a", "b"], states)
    assert snap.num_blocks == 4
    assert snap.block_states() == states
    assert snap.block_states()[3] - ev.BLOCK_OWNED_BASE == snap.owners.index("b")


def test_kv_snapshot_rejects_inconsistent_data() -> None:
    good = ev.KvSnapshot.build(0, ["a"], bytes([ev.BLOCK_OWNED_BASE]))
    with pytest.raises(ValidationError, match="owner index"):
        ev.KvSnapshot.build(0, ["a"], bytes([ev.BLOCK_OWNED_BASE + 1]))
    with pytest.raises(ValidationError, match="num_blocks"):
        ev.KvSnapshot(t=0, num_blocks=2, owners=["a"], states=good.states)


def test_frame_defaults() -> None:
    frame = ev.Frame(replica="r1", seq=0, t=0, events=[ev.EVENT_ADAPTER.validate_python(HELLO)])
    wire = json.loads(frame.model_dump_json())
    assert (wire["v"], wire["dropped"]) == (ev.SCHEMA_VERSION, 0)
