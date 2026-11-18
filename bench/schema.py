"""The workload file (B2), the files of a C5 run directory (decision 0013), and the group file
and the files of a session directory (decision 0016)."""

from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

CONFIG = "config.json"
ENV = "env.json"
RAW = "raw.jsonl"
METRICS = "metrics.txt"
POWERMETRICS = "powermetrics.plist"
POWERMETRICS_GZ = "powermetrics.plist.gz"
TARGET_LOG = "target.log"
SUMMARY = "summary.md"
CHART = "latency.svg"

GROUP = "group.json"
PLAN = "plan.json"
PROBES = "probes.jsonl"
SESSION = "session.json"
TWIN = "-twin"
FIRST_PORT = 8100
# The request fields every target gets alike (rule 7); a field one target alone gets may not
# replace them.
COMMON_FIELDS = (
    "model",
    "messages",
    "max_tokens",
    "temperature",
    "stream",
    "stream_options",
    "chat_template_kwargs",
)


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Message(Frozen):
    role: Literal["system", "user", "assistant"]
    content: str


class WorkloadRequest(Frozen):
    id: str
    messages: tuple[Message, ...]
    max_tokens: int = Field(gt=0)
    prompt_tokens: int = Field(gt=0, description="the rendered prompt's length, thinking off")


class Source(Frozen):
    dataset: str
    revision: str
    file: str
    licence: str


class Workload(Frozen):
    name: str
    source: Source
    tokenizer: str = Field(description="the model repo and revision that counted prompt_tokens")
    chat_template_kwargs: dict[str, bool]
    seed: int
    requests: tuple[WorkloadRequest, ...]


def load_workload(path: Path) -> Workload:
    return Workload.model_validate_json(path.read_text())


class RunConfig(Frozen):
    target: str
    base_url: str
    launch: str | None
    served_model: str
    workload: str
    workload_sha256: str
    requests: int
    concurrency: int
    repeats: int
    warmup: int
    cooldown_s: float
    temperature: float
    chat_template_kwargs: dict[str, bool]
    extra_body: dict[str, JsonValue] = Field(
        default_factory=dict, description="request fields this target alone got"
    )
    smoke: bool


class Power(Frozen):
    on_ac: bool | None
    battery_percent: int | None
    low_power_mode: bool | None
    charging: bool | None
    lid_open: bool | None


class RunEnv(Frozen):
    commit: str
    dirty: bool
    chip: str
    cores: int
    memory_gb: float
    macos: str
    python: str
    libraries: dict[str, str]
    power: Power
    powermetrics: bool
    containers: int | None = Field(description="running Docker containers; None: no daemon")
    top_processes: tuple[str, ...]
    started: str
    ended: str


class Usage(Frozen):
    prompt_tokens: int
    completion_tokens: int


class Sent(Frozen):
    """One request as the load generator saw it. ``sent_unix`` is the wall clock, the clock
    powermetrics stamps; chunk and end times are ms after the send, on ``perf_counter``."""

    id: str
    sent_unix: float
    chunk_ms: tuple[float, ...]
    end_ms: float
    usage: Usage | None
    finish_reason: str | None
    error: str | None


class RequestRecord(Frozen):
    """One line of raw.jsonl. Times are seconds from run start, chunk times ms after send."""

    phase: Literal["warmup", "measure"]
    repeat: int
    id: str
    sent_s: float
    chunk_ms: tuple[float, ...]
    end_ms: float
    usage: Usage | None
    finish_reason: str | None
    error: str | None


class EngineSpec(Frozen):
    """One server of a group. ``launch`` may hold ``{model}``, the pinned snapshot's path, and
    ``{port}``; it runs through shlex, never a shell."""

    name: str
    launch: str
    model: str | None = Field(description='the id to send; "{model}" is the snapshot path')
    extra_body: dict[str, JsonValue] = Field(default_factory=dict)
    footprint_gb: float = Field(gt=0, description="limit on this server's peak footprint")

    @model_validator(mode="after")
    def _no_common_field(self) -> Self:
        if clash := sorted(self.extra_body.keys() & set(COMMON_FIELDS)):
            raise ValueError(f"{self.name}: extra_body replaces fields every target gets: {clash}")
        return self


class Gates(Frozen):
    twin_tpot_peak: float = Field(gt=0, description="largest |twin TPOT median ratio - 1|")
    twin_tpot_sustained: float = Field(gt=0)


class Group(Frozen):
    """Everything fixed before a session, committed before its first request. The twin is not
    listed: every session starts the baseline twice."""

    name: str
    concurrency: int = Field(ge=1)
    servers: Literal["resident", "per_chunk"]
    prompts_per_chunk: int = Field(ge=2)
    warmup_per_chunk: int = Field(ge=1)
    seed: int
    baseline: EngineSpec
    engines: tuple[EngineSpec, ...]
    gates: Gates

    @model_validator(mode="after")
    def _names(self) -> Self:
        names = [self.baseline.name, *(e.name for e in self.engines)]
        if len(set(names)) != len(names):
            raise ValueError(f"engine names repeat: {names}")
        if any(n.endswith(TWIN) for n in names):
            raise ValueError(f"no engine name may end in {TWIN!r}: the session derives the twin")
        return self

    @property
    def twin(self) -> EngineSpec:
        return self.baseline.model_copy(update={"name": self.baseline.name + TWIN})

    @property
    def servers_in_order(self) -> tuple[EngineSpec, ...]:
        """Baseline, twin, then the others; a server's port is ``FIRST_PORT`` plus its index."""
        return (self.baseline, self.twin, *self.engines)


class SessionRecord(Frozen):
    """One line of a session's raw.jsonl. Block -1 holds a resident session's liveness
    requests, sent before the first block."""

    engine: str
    block: int
    pos: int = Field(description="the engine's place in the block's order; -1 with block -1")
    phase: Literal["liveness", "warmup", "measure"]
    sent: Sent


class Probe(Frozen):
    """One line of probes.jsonl: the machine after a chunk, or before any server starts."""

    at_unix: float
    block: int
    engine: str | None = Field(description="whose chunk just ended; None only on the first line")
    footprint_bytes: int | None = Field(description="phys_footprint over its process group")
    peak_bytes: int | None = Field(description="phys_footprint_peak, summed the same way")
    swap_used_bytes: int
    power: Power

    @model_validator(mode="after")
    def _memory_needs_an_engine(self) -> Self:
        # A probe that names an engine may still lack its footprint, when the server is gone;
        # the memory gate then fails.
        if self.engine is None and (self.footprint_bytes, self.peak_bytes) != (None, None):
            raise ValueError("a probe has footprints only when it names an engine")
        return self


class SessionInfo(Frozen):
    """session.json, written last, however the session ended."""

    served_models: dict[str, str]
    smoke: bool
    settle_s: float
    started_unix: float
    ended_unix: float
    blocks_planned: int
    blocks_run: int
    ended: Literal["completed", "failed request", "interrupted"]
