"""B1 and B5: a whole benchmark run against ``ullm serve --fake`` writes a C5 directory, and a
run that breaks B5 never starts (decision 0013)."""

import json
import socket
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from bench import analysis, conditions, schema
from bench.run import main
from bench.schema import Message, Power, Source, Workload, WorkloadRequest

from ullm.runner.fake_runner import FakeTokenizer


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _workload(path: Path, n: int) -> Path:
    requests = []
    for i in range(n):
        content = f"question {i} " * (i + 1)
        prompt = FakeTokenizer().render([{"role": "user", "content": content}], {})
        requests.append(
            WorkloadRequest(
                id=f"t-{i}",
                messages=(Message(role="user", content=content),),
                max_tokens=4 + i,
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


def test_a_smoke_run_writes_every_c5_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    port = _free_port()
    out = tmp_path / "run"
    code = main(
        [
            "--target",
            "fake",
            "--launch",
            f"{sys.executable} -m ullm.cli serve --fake --port {port}",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--workload",
            str(_workload(tmp_path / "w.json", 5)),
            "--concurrency",
            "2",
            "--repeats",
            "2",
            "--warmup",
            "2",
            "--cooldown",
            "0.1",
            "--ready-timeout",
            "30",
            "--smoke",
            "--out",
            str(out),
        ]
    )
    assert code == 0
    for name in (schema.CONFIG, schema.ENV, schema.RAW, schema.SUMMARY, schema.CHART):
        assert (out / name).is_file(), name
    assert "Uvicorn running" in (out / schema.TARGET_LOG).read_text()
    assert not (out / schema.METRICS).exists()  # ullm serve has no /metrics until M2-T5
    assert not (out / schema.POWERMETRICS).exists()

    config = json.loads((out / schema.CONFIG).read_text())
    assert (config["served_model"], config["requests"], config["smoke"]) == ("fake", 5, True)
    raw = [json.loads(line) for line in (out / schema.RAW).read_text().splitlines()]
    assert [(r["phase"], r["repeat"]) for r in raw] == [("warmup", -1)] * 2 + [
        ("measure", 0)
    ] * 5 + [("measure", 1)] * 5
    assert all(r["error"] is None for r in raw)
    summary = (out / schema.SUMMARY).read_text()
    assert "smoke: not a B5 run" in summary
    assert "| **median** |" in summary

    capsys.readouterr()
    assert analysis.main([str(out), str(out), "--compare"]) == 0
    rows = [line for line in capsys.readouterr().out.splitlines() if line.startswith("| fake")]
    assert len(rows) == 2

    _workload(tmp_path / "w.json", 6)
    with pytest.raises(SystemExit, match="SHA-256"):
        analysis.load_run(out)


def test_a_full_run_that_breaks_b5_never_starts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "run"
    code = main(
        [
            "--target",
            "nothing",
            "--workload",
            str(_workload(tmp_path / "w.json", 1)),
            "--repeats",
            "1",
            "--out",
            str(out),
        ]
    )
    assert code == 2
    assert not out.exists()
    assert "1 repeats" in capsys.readouterr().err


def test_a_launch_onto_a_busy_port_never_starts(tmp_path: Path) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        out = tmp_path / "run"
        code = main(
            [
                "--target",
                "fake",
                "--launch",
                f"{sys.executable} -m ullm.cli serve --fake --port {port}",
                "--base-url",
                f"http://127.0.0.1:{port}",
                "--workload",
                str(_workload(tmp_path / "w.json", 1)),
                "--smoke",
                "--out",
                str(out),
            ]
        )
    assert code == 2
    assert not out.exists()


class _SilentPowerMetrics:
    """powermetrics that started and logged nothing, as when root was refused."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def stop(self) -> None:
        return


def test_a_full_run_without_a_powermetrics_log_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(
        conditions, "power", lambda: Power(on_ac=True, battery_percent=100, low_power_mode=False)
    )
    monkeypatch.setattr(conditions, "can_sudo", lambda: True)
    monkeypatch.setattr(conditions, "containers", lambda: 0)
    monkeypatch.setattr(conditions, "PowerMetrics", _SilentPowerMetrics)
    port = _free_port()
    out = tmp_path / "run"
    code = main(
        [
            "--target",
            "fake",
            "--launch",
            f"{sys.executable} -m ullm.cli serve --fake --port {port}",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--workload",
            str(_workload(tmp_path / "w.json", 2)),
            "--repeats",
            "3",
            "--warmup",
            "1",
            "--cooldown",
            "0.1",
            "--ready-timeout",
            "30",
            "--out",
            str(out),
        ]
    )
    assert code == 1
    assert json.loads((out / schema.ENV).read_text())["powermetrics"] is False
    assert "powermetrics" in capsys.readouterr().err
    assert "| fake (not B5: no powermetrics log) |" in analysis.compare([analysis.load_run(out)])


@pytest.mark.parametrize(
    "wrong",
    [
        pytest.param(lambda r: {"prompt_tokens": r.prompt_tokens + 1}, id="prompt-mismatch"),
        pytest.param(lambda r: {"max_tokens": 10**6}, id="refused-request"),
    ],
)
def test_a_run_with_a_mismatch_or_a_failure_exits_1(
    wrong: Callable[[WorkloadRequest], dict[str, int]],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    workload = Workload.model_validate_json(_workload(tmp_path / "w.json", 2).read_text())
    first, second = workload.requests
    broken = first.model_copy(update=wrong(first))
    path = tmp_path / "wrong.json"
    path.write_text(workload.model_copy(update={"requests": (broken, second)}).model_dump_json())
    port = _free_port()
    code = main(
        [
            "--target",
            "fake",
            "--launch",
            f"{sys.executable} -m ullm.cli serve --fake --port {port}",
            "--base-url",
            f"http://127.0.0.1:{port}",
            "--workload",
            str(path),
            "--warmup",
            "1",
            "--repeats",
            "1",
            "--ready-timeout",
            "30",
            "--smoke",
            "--out",
            str(tmp_path / "run"),
        ]
    )
    assert code == 1
    assert "1 measured requests failed or rendered another prompt" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--concurrency", "--requests"])
def test_a_run_of_nothing_is_refused(flag: str, tmp_path: Path) -> None:
    out = tmp_path / "run"
    with pytest.raises(SystemExit) as refused:
        main(["--target", "fake", flag, "0", "--ready-timeout", "1", "--smoke", "--out", str(out)])
    assert refused.value.code == 2
    assert not out.exists()
