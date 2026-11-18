"""A session against ``ullm serve --fake`` servers writes every file and stops every server, in
both server modes; each gate discards a session that breaks it; a session that breaks B5 never
starts (decision 0016). The machine is faked: power, swap, footprints and the clock log."""

import json
import plistlib
import shutil
import socket
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
from bench import analysis, conditions, schema
from bench.analysis import _twin_gate
from bench.paired import Estimate
from bench.regimes import Regime
from bench.schema import Group, Message, Power, Probe, Source, Workload, WorkloadRequest
from bench.session import main

from ullm.runner.fake_runner import FakeTokenizer

GOOD = Power(on_ac=True, battery_percent=100, low_power_mode=False, charging=False, lid_open=True)
MB = 2**20


def _sample(at: datetime) -> bytes:
    return plistlib.dumps(
        {
            "timestamp": at,
            "elapsed_ns": 1_000_000_000,
            "gpu": {
                "idle_ratio": 0.0,
                "dvfm_states": [{"freq": 1470, "used_ratio": 1.0}],
                "sw_requested_state": [{"sw_req_state": "P9", "used_ratio": 1.0}],
                "sw_state": [{"sw_state": "SW_P9", "used_ratio": 1.0}],
            },
        }
    )


class FakePowerMetrics:
    """A full clock at 1470 MHz, one sample a second from start to stop."""

    def __init__(self, path: Path) -> None:
        self.path, self.start = path, int(time.time()) - 1

    def stop(self) -> None:
        stop = int(time.time()) + 2
        first = datetime.fromtimestamp(self.start, UTC).replace(tzinfo=None)
        self.path.write_bytes(b"\x00".join(_sample(first) for _ in range(stop - self.start)))


def _free_ports(n: int) -> int:
    """The first of ``n`` free ports in a row."""
    while True:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            base = int(s.getsockname()[1])
        try:
            for port in range(base, base + n):
                with socket.socket() as s:
                    s.bind(("127.0.0.1", port))
            return base
        except OSError:
            continue


def _workload(path: Path, n: int, broken: int | None = None) -> Path:
    requests = []
    for i in range(n):
        content = f"question {i} " * (i + 1)
        prompt = FakeTokenizer().render([{"role": "user", "content": content}], {})
        requests.append(
            WorkloadRequest(
                id=f"t-{i}",
                messages=(Message(role="user", content=content),),
                max_tokens=10**6 if i == broken else 6 + i,
                prompt_tokens=len(prompt),
            )
        )
    workload = Workload(
        name="test",
        source=Source(dataset="none", revision="none", file="none", licence="none"),
        tokenizer="fake",
        chat_template_kwargs={"enable_thinking": False},
        seed=0,
        requests=tuple(requests),
    )
    path.write_text(workload.model_dump_json())
    return path


def _group(
    path: Path, servers: Literal["resident", "per_chunk"], launch: str | None = None
) -> Path:
    serve = launch or f"{sys.executable} -m ullm.cli serve --fake --port {{port}}"
    spec = {"launch": serve, "model": None, "footprint_gb": 1.0}
    group = {
        "name": f"test-{servers}",
        "concurrency": 2,
        "servers": servers,
        "prompts_per_chunk": 2,
        "warmup_per_chunk": 1,
        "seed": 0,
        "baseline": {"name": "base", **spec},
        "engines": [{"name": "other", **spec, "extra_body": {"seed": 0}}],
        "gates": {"twin_tpot_peak": 0.5, "twin_tpot_sustained": 0.5},
    }
    path.write_text(json.dumps(group))
    return path


@pytest.fixture
def machine(monkeypatch: pytest.MonkeyPatch) -> int:
    """A Mac that meets B5, on free ports; returns the first port."""
    monkeypatch.setattr(conditions, "power", lambda: GOOD)
    monkeypatch.setattr(conditions, "can_sudo", lambda: True)
    monkeypatch.setattr(conditions, "containers", lambda: 0)
    monkeypatch.setattr(conditions, "powermetrics_running", lambda: False)
    monkeypatch.setattr(conditions, "swap_used", lambda: 400 * MB)
    monkeypatch.setattr(conditions, "footprint", lambda pgid: (100 * MB, 200 * MB))
    monkeypatch.setattr(conditions, "PowerMetrics", FakePowerMetrics)
    first = _free_ports(3)
    monkeypatch.setattr(schema, "FIRST_PORT", first)
    return first


def _session(
    tmp_path: Path, servers: Literal["resident", "per_chunk"], *extra: str
) -> tuple[int, Path]:
    out = tmp_path / "session"
    code = main(
        [
            str(_group(tmp_path / "group.json", servers)),
            "--workload",
            str(_workload(tmp_path / "w.json", 5)),
            "--settle",
            "0",
            "--ready-timeout",
            "30",
            "--out",
            str(out),
            *extra,
        ]
    )
    return code, out


def _ports_free(first: int) -> bool:
    for port in range(first, first + 3):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) == 0:
                return False
    return True


@pytest.mark.parametrize("servers", ["resident", "per_chunk"])
def test_a_session_writes_every_file_and_stops_every_server(
    tmp_path: Path, machine: int, servers: Literal["resident", "per_chunk"]
) -> None:
    code, out = _session(tmp_path, servers)
    assert code == 0
    for name in (
        schema.GROUP,
        schema.PLAN,
        schema.ENV,
        schema.RAW,
        schema.PROBES,
        schema.POWERMETRICS_GZ,
        schema.SESSION,
        schema.SUMMARY,
        "base.log",
        "base-twin.log",
        "other.log",
    ):
        assert (out / name).is_file(), name
    assert not (out / schema.POWERMETRICS).exists()
    assert _ports_free(machine)

    session = analysis.load_session(out)
    assert session.verdict == "VALID"
    assert session.info.ended == "completed"
    assert session.info.blocks_run == session.info.blocks_planned == 2
    assert set(session.regimes.values()) == {Regime.PEAK}
    plan = json.loads((out / schema.PLAN).read_text())
    measured = [r for r in session.records if r.phase == "measure"]
    assert len(measured) == 2 * 2 * 3
    assert not {r.sent.id for r in measured} & set(plan["warmup"])
    for block, order in enumerate(plan["orders"][:2]):
        chunks = [r.engine for r in measured if r.block == block]
        assert list(dict.fromkeys(chunks)) == order
    liveness = [r.engine for r in session.records if r.phase == "liveness"]
    assert liveness == (["base", "base-twin", "other"] if servers == "resident" else [])
    assert [p.engine for p in session.probes][:2] == [None, plan["orders"][0][0]]
    assert len(session.probes) == 1 + 2 * 3
    starts = (out / "base.log").read_text().count("Uvicorn running")
    assert starts == (1 if servers == "resident" else 2)
    assert session.group.engines[0].extra_body == {"seed": 0}
    summary = (out / schema.SUMMARY).read_text()
    assert summary.startswith("# VALID: test-")
    assert "## Peak" in summary
    assert "| base-twin |" in summary


def test_a_failed_request_ends_the_session_after_its_chunk(tmp_path: Path, machine: int) -> None:
    out = tmp_path / "session"
    code = main(
        [
            str(_group(tmp_path / "group.json", "resident")),
            "--workload",
            str(_workload(tmp_path / "w.json", 5, broken=0)),
            "--settle",
            "0",
            "--ready-timeout",
            "30",
            "--out",
            str(out),
        ]
    )
    assert code == 1
    session = analysis.load_session(out)
    assert session.info.ended == "failed request"
    assert session.verdict == "DISCARDED"
    assert not next(g for g in session.gates if g.name == "Liveness").passed
    engines = {(r.block, r.engine) for r in session.records if r.phase == "measure"}
    assert len(engines) <= 1
    assert _ports_free(machine)


def test_a_server_that_never_becomes_ready_ends_the_session(tmp_path: Path, machine: int) -> None:
    out = tmp_path / "session"
    code = main(
        [
            str(_group(tmp_path / "group.json", "per_chunk", launch="sleep 60")),
            "--workload",
            str(_workload(tmp_path / "w.json", 5)),
            "--settle",
            "0",
            "--ready-timeout",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 1
    info = json.loads((out / schema.SESSION).read_text())
    assert (info["ended"], info["blocks_run"]) == ("failed request", 0)
    assert (out / schema.SUMMARY).read_text().startswith("# DISCARDED")


def test_a_smoke_session_is_never_valid(tmp_path: Path, machine: int) -> None:
    code, out = _session(tmp_path, "resident", "--smoke", "--blocks", "1")
    assert code == 1
    session = analysis.load_session(out)
    assert session.verdict == "SMOKE"
    assert session.info.blocks_run == 1
    assert (out / schema.SUMMARY).read_text().startswith("# SMOKE")


def test_a_session_that_breaks_b5_or_would_overwrite_never_starts(
    tmp_path: Path,
    machine: int,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(conditions, "powermetrics_running", lambda: True)
    code, out = _session(tmp_path, "resident")
    assert code == 2
    assert not out.exists()
    assert "another powermetrics" in capsys.readouterr().err

    out.mkdir()
    code, _ = _session(tmp_path, "resident", "--smoke")
    assert code == 2
    assert "never writes over another" in capsys.readouterr().err


def test_a_busy_port_refuses_the_session(tmp_path: Path, machine: int) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", machine + 1))
        busy.listen()
        code, out = _session(tmp_path, "resident")
    assert code == 2
    assert not out.exists()


@pytest.fixture(scope="module")
def valid(tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp_path = tmp_path_factory.mktemp("valid")
    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(conditions, "power", lambda: GOOD)
        monkeypatch.setattr(conditions, "can_sudo", lambda: True)
        monkeypatch.setattr(conditions, "containers", lambda: 0)
        monkeypatch.setattr(conditions, "powermetrics_running", lambda: False)
        monkeypatch.setattr(conditions, "swap_used", lambda: 400 * MB)
        monkeypatch.setattr(conditions, "footprint", lambda pgid: (100 * MB, 200 * MB))
        monkeypatch.setattr(conditions, "PowerMetrics", FakePowerMetrics)
        monkeypatch.setattr(schema, "FIRST_PORT", _free_ports(3))
        code, out = _session(tmp_path, "resident")
    assert code == 0
    return out


def _tampered(valid: Path, tmp_path: Path, change: dict[str, object], line: int = -1) -> Path:
    """A copy of the valid session with one probe changed."""
    copy = tmp_path / "copy"
    shutil.copytree(valid, copy)
    probes = [Probe.model_validate_json(p) for p in (copy / schema.PROBES).read_text().splitlines()]
    probes[line] = probes[line].model_copy(update=change)
    (copy / schema.PROBES).write_text("".join(p.model_dump_json() + "\n" for p in probes))
    return copy


def _failed(path: Path) -> list[str]:
    return [g.name for g in analysis.load_session(path).gates if not g.passed]


def test_the_valid_session_passes_every_gate(valid: Path) -> None:
    assert _failed(valid) == []


@pytest.mark.parametrize(
    ("change", "gate"),
    [
        ({"swap_used_bytes": 400 * MB + 1}, "Memory"),
        ({"peak_bytes": 2**30 + 1}, "Memory"),
        ({"peak_bytes": None, "footprint_bytes": None}, "Memory"),
        ({"power": GOOD.model_copy(update={"on_ac": False})}, "Conditions"),
        ({"power": GOOD.model_copy(update={"low_power_mode": True})}, "Conditions"),
    ],
)
def test_each_broken_probe_discards_the_session(
    valid: Path, tmp_path: Path, change: dict[str, object], gate: str
) -> None:
    copy = _tampered(valid, tmp_path, change)
    assert _failed(copy) == [gate]
    assert analysis.load_session(copy).verdict == "DISCARDED"


def test_a_missing_clock_log_discards_the_session(valid: Path, tmp_path: Path) -> None:
    copy = tmp_path / "copy"
    shutil.copytree(valid, copy)
    (copy / schema.POWERMETRICS_GZ).unlink()
    assert "Conditions" in _failed(copy)


def test_the_twin_gate_checks_the_tpot_median_in_each_regime() -> None:
    group = Group.model_validate_json(
        (Path(__file__).parents[2] / "bench/groups/m1-c1.json").read_text()
    )

    def tpot(point: float) -> Estimate:
        return Estimate(point, None, None, 36, 3)

    assert _twin_gate(group, {Regime.PEAK: tpot(1.004), Regime.SUSTAINED: tpot(0.975)}).passed
    assert not _twin_gate(group, {Regime.PEAK: tpot(1.006), Regime.SUSTAINED: None}).passed
    assert not _twin_gate(group, {Regime.PEAK: None, Regime.SUSTAINED: tpot(1.031)}).passed
    assert _twin_gate(group, {Regime.PEAK: None, Regime.SUSTAINED: None}).passed
