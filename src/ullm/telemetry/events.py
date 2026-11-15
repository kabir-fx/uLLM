"""C2: telemetry protocol v1 (decision 0003).

The one definition of the telemetry wire format. The JSON Schema in
``schema/telemetry.v1.json`` and the terminal's TypeScript types are generated
from these models (``make contracts``); ``make check`` fails if they are stale.

Rules that keep the format safe to evolve and to publish:

- Within v1 only additive changes are allowed: new event types, and new fields
  that default to ``None``. Readers ignore what they do not know and treat a
  missing field as null. Anything else is v2.
- Producers are strict (``extra="forbid"``): a misspelt field fails a test.
- No event carries prompt or output text. Only counts, timings and ids.
- Times are milliseconds since the replica's telemetry epoch, monotonic clock,
  serialised to microsecond precision.
"""

import base64
from typing import Annotated, Final, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PlainSerializer,
    TypeAdapter,
    WithJsonSchema,
    model_validator,
)

SCHEMA_VERSION: Final = 1

# A custom serialiser hides the constraint from the exported schema, so it is restated.
Ms = Annotated[
    float,
    Field(ge=0),
    PlainSerializer(lambda v: round(v, 3), return_type=float, when_used="json"),
    WithJsonSchema({"type": "number", "minimum": 0}),
]
Count = Annotated[int, Field(ge=0)]
Positive = Annotated[int, Field(ge=1)]


class _Model(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        # Fields with defaults are always serialised, so the exported schema
        # marks them required. That makes ``type`` a required literal and gives
        # TypeScript a real discriminated union.
        json_schema_serialization_defaults_required=True,
    )


# --- hello -----------------------------------------------------------------------


class Limits(_Model):
    max_num_seqs: Positive
    max_prefill_tokens: Positive
    max_model_len: Positive


class KvInfo(_Model):
    """KV capacity. Block fields stay null until blocks exist (M3)."""

    block_size: Positive | None = None
    num_blocks: Count | None = None
    capacity_tokens: Count | None = None


class Machine(_Model):
    chip: str
    memory_gb: Positive


class Hello(_Model):
    """First event on every connection and in every recording."""

    type: Literal["hello"] = "hello"
    t: Ms
    replica: str
    model: str
    source: Literal["engine", "sim", "fixture"]  # the terminal labels sim and fixture data
    version: str
    commit: str | None
    epoch_unix_ms: Count  # wall-clock time of t = 0
    limits: Limits
    kv: KvInfo
    machine: Machine | None


# --- request life cycle -----------------------------------------------------------


class ReqArrived(_Model):
    type: Literal["req.arrived"] = "req.arrived"
    t: Ms
    rid: str
    prompt_tokens: Count
    max_tokens: Positive
    tenant: str | None = None  # from M8


class ReqAdmitted(_Model):
    type: Literal["req.admitted"] = "req.admitted"
    t: Ms
    rid: str
    queue_ms: Ms
    cached_tokens: Count = 0  # prefix-cache hit, from M3


class ReqFirstToken(_Model):
    type: Literal["req.first_token"] = "req.first_token"
    t: Ms
    rid: str
    ttft_ms: Ms  # from arrival


class TokenUsage(_Model):
    prompt_tokens: Count
    completion_tokens: Count
    cached_tokens: Count


class Timings(_Model):
    queue_ms: Ms
    ttft_ms: Ms
    tpot_ms: Ms | None  # null with fewer than two output tokens
    e2e_ms: Ms


class ReqFinished(_Model):
    type: Literal["req.finished"] = "req.finished"
    t: Ms
    rid: str
    reason: Literal["stop", "length", "tool_calls"]
    usage: TokenUsage
    timings: Timings


class ReqAborted(_Model):
    type: Literal["req.aborted"] = "req.aborted"
    t: Ms
    rid: str
    reason: Literal["client_disconnect", "shutdown", "error"]
    completion_tokens: Count


class ReqPreempted(_Model):
    type: Literal["req.preempted"] = "req.preempted"
    t: Ms
    rid: str
    completion_tokens: Count
    blocks_freed: Count


# --- engine -----------------------------------------------------------------------


class Step(_Model):
    """One engine step. ``t`` is its end, after ``mx.eval``; it started at ``t - ms``."""

    type: Literal["step"] = "step"
    t: Ms
    n: Count
    kind: Literal["prefill", "decode"]
    rids: list[str]  # rows, in batch order
    tokens: Count  # tokens processed this step
    ms: Ms
    waiting: Count
    running: Count
    kv_tokens: Count  # tokens held in KV after the step


# One byte per block in KvSnapshot.states.
BLOCK_FREE: Final = 0
BLOCK_CACHED: Final = 1  # ref count 0, contents kept for prefix hits
BLOCK_SHARED: Final = 2  # ref count above 1
BLOCK_OWNED_BASE: Final = 3  # 3 + i: owned by owners[i] alone
MAX_OWNERS: Final = 256 - BLOCK_OWNED_BASE


class KvSnapshot(_Model):
    """The state of every KV block, compactly (from M3)."""

    type: Literal["kv.snapshot"] = "kv.snapshot"
    t: Ms
    num_blocks: Count
    owners: list[str] = Field(max_length=MAX_OWNERS)
    states: str  # base64, one byte per block

    @classmethod
    def build(cls, t: float, owners: list[str], states: bytes) -> "KvSnapshot":
        return cls(
            t=t,
            num_blocks=len(states),
            owners=owners,
            states=base64.b64encode(states).decode("ascii"),
        )

    def block_states(self) -> bytes:
        return base64.b64decode(self.states, validate=True)

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        raw = self.block_states()
        if len(raw) != self.num_blocks:
            raise ValueError(f"states holds {len(raw)} blocks, num_blocks is {self.num_blocks}")
        if raw and max(raw) >= BLOCK_OWNED_BASE + len(self.owners):
            raise ValueError("a block names an owner index outside owners")
        return self


# --- router -----------------------------------------------------------------------


class RouterDecision(_Model):
    """Where the router sent a request, and why (from M5)."""

    type: Literal["router.decision"] = "router.decision"
    t: Ms
    rid: str
    replica: str
    kind: Literal["prefix", "least_load", "round_robin"]
    matched_messages: Count
    retry: bool


# --- frame ------------------------------------------------------------------------

Event = Annotated[
    Hello
    | ReqArrived
    | ReqAdmitted
    | ReqFirstToken
    | ReqFinished
    | ReqAborted
    | ReqPreempted
    | Step
    | KvSnapshot
    | RouterDecision,
    Field(discriminator="type"),
]
EVENT_ADAPTER: Final[TypeAdapter[Event]] = TypeAdapter(Event)


class Frame(_Model):
    """One flush of the telemetry stream: an SSE ``data:`` line, or one line of a trace file."""

    v: Literal[1] = 1
    replica: str
    seq: Count  # +1 per frame on a connection, so readers can see gaps
    t: Ms
    dropped: Count = 0  # events the bounded buffer dropped since the previous frame
    events: list[Event]
