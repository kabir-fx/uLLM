"""A session against ``ullm serve --fake`` servers writes every file and stops every server, in
both server modes; each gate discards a session that breaks it; a session that breaks B5 never
starts (decision 0016). The machine is faked: power, swap, footprints and the clock log."""

import json
import plistlib
import shutil
import socket
import sys
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest
from bench import analysis, conditions, schema, steps
from bench.analysis import _twin_gate
from bench.paired import Estimate
from bench.regimes import Regime
from bench.schema import (
    Group,
    Message,
    Power,
    Probe,
    SessionRecord,
    Source,
    Workload,
    WorkloadRequest,
)
from bench.session import main

from ullm.runner.fake_runner import FakeTokenizer
from ullm.telemetry.check import check_trace
from ullm.telemetry.events import ReqArrived
from ullm.telemetry.trace import iter_frames

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
    assert _failed_apart_from_twin(out) == []
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
    assert code == (0 if session.verdict == "VALID" else 1)
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
    assert summary.startswith(f"# {session.verdict}: test-")
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
    assert _failed_apart_from_twin(out) == []
    assert code == (0 if analysis.load_session(out).verdict == "VALID" else 1)
    # On a GitHub runner two fake servers' TPOT can differ by 2x, so the twin
    # gate is tested on a twin that repeats the baseline's timings exactly.
    return _rewritten(out, tmp_path, _baseline_timings("base-twin", out))


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


def _failed_apart_from_twin(path: Path) -> list[str]:
    return [g for g in _failed(path) if g != "Twin"]


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


def test_compare_prints_each_sessions_ratios_with_its_lifecycle(
    valid: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert analysis.main([str(valid), "--compare"]) == 0
    out = capsys.readouterr().out
    assert out.startswith("### Peak: test-resident, concurrency 2, servers stay loaded, VALID")
    assert "| other |" in out
    assert "| base-twin |" in out


def test_a_session_that_crashes_is_recorded_as_interrupted(
    tmp_path: Path, machine: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = iter(range(100))

    def swap_used() -> int:
        if next(calls) == 2:
            raise RuntimeError("sysctl vm.swapusage gave no swap figure")
        return 400 * MB

    monkeypatch.setattr(conditions, "swap_used", swap_used)
    with pytest.raises(RuntimeError, match="swapusage"):
        _session(tmp_path, "resident")
    info = json.loads((tmp_path / "session" / schema.SESSION).read_text())
    assert info["ended"] == "interrupted"
    assert _ports_free(machine)


def _rewritten(
    valid: Path, tmp_path: Path, change: Callable[[SessionRecord], SessionRecord]
) -> Path:
    """A copy of the valid session with its raw.jsonl records changed."""
    copy = tmp_path / "copy"
    shutil.copytree(valid, copy)
    raw = copy / schema.RAW
    rows = [
        change(SessionRecord.model_validate_json(line)) for line in raw.read_text().splitlines()
    ]
    raw.write_text("".join(r.model_dump_json() + "\n" for r in rows))
    return copy


def _baseline_timings(engine: str, session: Path) -> Callable[[SessionRecord], SessionRecord]:
    base = {
        (r.block, r.sent.id): r.sent
        for line in (session / schema.RAW).read_text().splitlines()
        if (r := SessionRecord.model_validate_json(line)).engine == "base" and r.phase == "measure"
    }

    def change(r: SessionRecord) -> SessionRecord:
        if r.engine != engine or r.phase != "measure":
            return r
        b = base[(r.block, r.sent.id)]
        sent = r.sent.model_copy(
            update={"chunk_ms": b.chunk_ms, "end_ms": b.end_ms, "usage": b.usage}
        )
        return r.model_copy(update={"sent": sent})

    return change


def _slower(engine: str, phase: str, factor: float) -> Callable[[SessionRecord], SessionRecord]:
    def change(r: SessionRecord) -> SessionRecord:
        if r.engine != engine or r.phase != phase:
            return r
        sent = r.sent.model_copy(
            update={
                "chunk_ms": tuple(t * factor for t in r.sent.chunk_ms),
                "end_ms": r.sent.end_ms * factor,
            }
        )
        return r.model_copy(update={"sent": sent})

    return change


def test_ratios_pair_each_engine_with_the_baseline_prompt_by_prompt(
    valid: Path, tmp_path: Path
) -> None:
    session = analysis.load_session(_rewritten(valid, tmp_path, _slower("other", "measure", 2.0)))
    est = analysis.ratios(session, "other", Regime.PEAK)
    before = analysis.ratios(analysis.load_session(valid), "other", Regime.PEAK)
    assert est["TPOT"].point == pytest.approx(2 * before["TPOT"].point)
    assert est["E2E"].point == pytest.approx(2 * before["E2E"].point)
    assert est["TPOT"].prompts == 4


def test_a_twin_that_disagrees_with_the_baseline_discards_the_session(
    valid: Path, tmp_path: Path
) -> None:
    copy = _rewritten(valid, tmp_path, _slower("base-twin", "measure", 2.0))
    gates = analysis.load_session(copy).gates
    assert [g.name for g in gates if not g.passed] == ["Twin"], gates
    assert "Peak twin TPOT median 2.0000" in gates[-1].detail


def test_warmup_requests_are_never_measured(valid: Path, tmp_path: Path) -> None:
    session = analysis.load_session(_rewritten(valid, tmp_path, _slower("other", "warmup", 50.0)))
    before = analysis.ratios(analysis.load_session(valid), "other", Regime.PEAK)
    after = analysis.ratios(session, "other", Regime.PEAK)
    assert after["TPOT"] == before["TPOT"]
    assert after["TPOT"].prompts == 4


def test_a_reply_whose_prompt_length_differs_discards_the_session(
    valid: Path, tmp_path: Path
) -> None:
    done = False

    def change(r: SessionRecord) -> SessionRecord:
        nonlocal done
        if done or r.phase != "measure" or r.sent.usage is None:
            return r
        done = True
        usage = r.sent.usage.model_copy(update={"prompt_tokens": r.sent.usage.prompt_tokens + 1})
        return r.model_copy(update={"sent": r.sent.model_copy(update={"usage": usage})})

    assert _failed(_rewritten(valid, tmp_path, change)) == ["Liveness"]


def test_the_summary_states_its_conditions_and_keeps_streams_inside_a_regime(
    valid: Path,
) -> None:
    summary = (valid / schema.SUMMARY).read_text()
    env = json.loads((valid / schema.ENV).read_text())
    assert f"Commit `{env['commit'][:12]}`" in summary
    assert f"macOS {env['macos']}" in summary
    assert "lid open" in summary
    peak = summary.index("## Peak")
    assert summary.index("Largest gap") > peak
    assert "## Streams" not in summary


def test_a_missing_model_snapshot_refuses_the_session(
    tmp_path: Path, machine: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    import scripts.fetch_models

    monkeypatch.setattr(scripts.fetch_models, "local_path", lambda spec: None)
    out = tmp_path / "session"
    code = main(
        [
            str(_group(tmp_path / "group.json", "resident", launch="serve --model {model}")),
            "--workload",
            str(_workload(tmp_path / "w.json", 5)),
            "--out",
            str(out),
        ]
    )
    assert code == 2
    assert not out.exists()


def test_a_launch_may_write_into_the_session_directory(tmp_path: Path, machine: int) -> None:
    group = _group(tmp_path / "group.json", "per_chunk")
    spec = json.loads(group.read_text())
    spec["engines"][0]["launch"] += " --record {out}/other.jsonl.gz"
    group.write_text(json.dumps(spec))
    out = tmp_path / "a session"
    main(
        [
            str(group),
            *("--workload", str(_workload(tmp_path / "w.json", 5))),
            *("--settle", "0", "--ready-timeout", "30", "--blocks", "1", "--out", str(out)),
        ]
    )
    frames = list(iter_frames(out / "other.jsonl.gz"))
    assert check_trace(frames).problems == []


def test_a_server_started_per_chunk_records_each_block_apart(tmp_path: Path, machine: int) -> None:
    group = _group(tmp_path / "group.json", "per_chunk")
    spec = json.loads(group.read_text())
    spec["engines"][0]["launch"] += " --record {out}/other-{block}.jsonl.gz"
    group.write_text(json.dumps(spec))
    out = tmp_path / "session"
    code = main(
        [
            str(group),
            *("--workload", str(_workload(tmp_path / "w.json", 5))),
            *("--settle", "0", "--ready-timeout", "30", "--out", str(out)),
        ]
    )
    assert code == 0
    measured: dict[int, set[str]] = {}
    for line in (out / schema.RAW).read_text().splitlines():
        record = SessionRecord.model_validate_json(line)
        if record.engine == "other" and record.phase == "measure":
            measured.setdefault(record.block, set()).add(record.sent.id)
    assert sorted(measured) == [0, 1]
    for block, ids in measured.items():
        frames = list(iter_frames(out / f"other-{block}.jsonl.gz"))
        assert check_trace(frames).problems == []
        arrived = {e.rid for f in frames for e in f.events if isinstance(e, ReqArrived)}
        assert len(arrived) == len(ids) + 1  # the chunk's one warm-up request, then its block


def test_steps_reads_each_blocks_recording_inside_its_measured_span(
    tmp_path: Path, machine: int
) -> None:
    group = _group(tmp_path / "group.json", "per_chunk")
    spec = json.loads(group.read_text())
    spec["engines"][0]["launch"] += " --record {out}/other-{block}.jsonl.gz"
    group.write_text(json.dumps(spec))
    out = tmp_path / "session"
    main(
        [
            str(group),
            *("--workload", str(_workload(tmp_path / "w.json", 5))),
            *("--settle", "0", "--ready-timeout", "30", "--out", str(out)),
        ]
    )
    session = analysis.load_session(out)
    assert set(session.regimes.values()) == {Regime.PEAK}
    s = steps.split(
        [
            steps.chunk(iter_frames(out / f"other-{b}.jsonl.gz"), session.spans[b, "other"])
            for b in sorted(session.regimes)
        ]
    )
    measured = [r for r in session.records if r.engine == "other" and r.phase == "measure"]
    assert s.requests == len(measured) == 4
    assert s.dropped == 0
    assert s.prefill_ms_per_1k
    assert s.decode_ms_by_rows
    assert steps.main([str(out), "--engine", "other"]) == 0
    text = (out / steps.STEPS).read_text()
    assert "## Peak: 2 blocks, 0 events dropped" in text
    assert "| Requests | 4 |" in text


def _at_concurrency(valid: Path, tmp_path: Path, concurrency: int, engines: bool) -> Path:
    copy = tmp_path / f"c{concurrency}"
    shutil.copytree(valid, copy)
    group = json.loads((copy / schema.GROUP).read_text())
    group["concurrency"] = concurrency
    if not engines:
        group["engines"] = []
    (copy / schema.GROUP).write_text(json.dumps(group))
    return copy


def test_sweep_puts_each_concurrency_in_a_column(
    valid: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    low = _at_concurrency(valid, tmp_path, 2, engines=True)
    high = _at_concurrency(valid, tmp_path, 4, engines=False)
    assert analysis.main([str(high), str(low), "--sweep"]) == 0
    out = capsys.readouterr().out
    rate = analysis.ratios(analysis.load_session(low), "other", Regime.PEAK)["Output tok/s"]
    table = out.split("### Peak: Output tok/s, ratio to `base`")[1].split("###")[0]
    assert "| Engine | c2 | c4 |" in table
    assert f"| other | {analysis._estimate(rate)}" in table
    [row] = [line for line in table.splitlines() if line.startswith("| other |")]
    assert row.endswith("| not run |")
    # Two blocks give the twin no interval, so no gap can be claimed against it.
    assert f"| other | {analysis._estimate(rate)}, within noise |" in row
    assert "| base-twin |" in table
    assert "Below 1 is faster for TTFT, TPOT and E2E; above 1 is faster for output tok/s." in out
    assert out.count("- c2, VALID, servers stay loaded") == 1
    assert out.count("- c4, VALID, servers stay loaded") == 1
    assert out.count("Low Power Mode False") == 2


def test_sweep_names_a_discarded_sessions_column(
    valid: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    low = _at_concurrency(valid, tmp_path, 2, engines=True)
    swapped = _tampered(valid, tmp_path, {"swap_used_bytes": 400 * MB + 1})
    high = _at_concurrency(swapped, tmp_path, 4, engines=True)
    assert analysis.main([str(low), str(high), "--sweep"]) == 0
    assert "| Engine | c2 | c4, DISCARDED |" in capsys.readouterr().out
