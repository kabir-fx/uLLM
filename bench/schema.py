"""The workload file (B2) and the files of a C5 run directory (decision 0013)."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue

CONFIG = "config.json"
ENV = "env.json"
RAW = "raw.jsonl"
METRICS = "metrics.txt"
POWERMETRICS = "powermetrics.plist"
POWERMETRICS_GZ = "powermetrics.plist.gz"
TARGET_LOG = "target.log"
SUMMARY = "summary.md"
CHART = "latency.svg"


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
