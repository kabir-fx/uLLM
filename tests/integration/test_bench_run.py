"""B1: a whole smoke run against ``ullm serve --fake`` writes a C5 directory (decision 0013)."""

import json
import socket
import sys
from collections.abc import Callable
from pathlib import Path

import pytest
from bench import analysis, schema
from bench.run import main
from bench.schema import Message, Source, Workload, WorkloadRequest
from tests.prometheus import metric_totals

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
            "--out",
            str(out),
        ]
    )
    assert code == 0
    for name in (schema.CONFIG, schema.ENV, schema.RAW, schema.SUMMARY, schema.CHART):
        assert (out / name).is_file(), name
    assert "Uvicorn running" in (out / schema.TARGET_LOG).read_text()
    assert not (out / schema.POWERMETRICS).exists()
    assert not (out / schema.POWERMETRICS_GZ).exists()

    config = json.loads((out / schema.CONFIG).read_text())
    assert (config["served_model"], config["requests"], config["smoke"]) == ("fake", 5, True)
    raw = [json.loads(line) for line in (out / schema.RAW).read_text().splitlines()]
    assert [(r["phase"], r["repeat"]) for r in raw] == [("warmup", -1)] * 2 + [
        ("measure", 0)
    ] * 5 + [("measure", 1)] * 5
    assert all(r["error"] is None for r in raw)
    # O2's proof: the server's counters equal the load generator's totals, warm-up included.
    totals = metric_totals((out / schema.METRICS).read_text())
    assert totals["ullm_requests_total"] == len(raw)
    assert totals["ullm_prompt_tokens_total"] == sum(r["usage"]["prompt_tokens"] for r in raw)
    assert totals["ullm_generation_tokens_total"] == sum(
        r["usage"]["completion_tokens"] for r in raw
    )
    assert totals["ullm_ttft_seconds_count"] == len(raw)
    assert totals["ullm_aborts_total"] == 0
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
                "--out",
                str(out),
            ]
        )
    assert code == 2
    assert not out.exists()


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
        main(["--target", "fake", flag, "0", "--ready-timeout", "1", "--out", str(out)])
    assert refused.value.code == 2
    assert not out.exists()


def _fake_run(tmp_path: Path, *extra: str) -> tuple[int, Path]:
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
            "1",
            "--warmup",
            "1",
            "--ready-timeout",
            "30",
            "--out",
            str(out),
            *extra,
        ]
    )
    return code, out


def test_extra_body_and_model_are_sent_and_recorded(tmp_path: Path) -> None:
    code, out = _fake_run(tmp_path, "--model", "fake", "--extra-body", '{"seed": 0}')
    assert code == 0
    config = json.loads((out / schema.CONFIG).read_text())
    assert (config["served_model"], config["extra_body"]) == ("fake", {"seed": 0})
    assert "seed" in (out / schema.SUMMARY).read_text()


def test_a_model_the_target_does_not_serve_is_refused(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="not-served"):
        _fake_run(tmp_path, "--model", "not-served")


@pytest.mark.parametrize("body", ['{"temperature": 1}', "[0]", "not json"])
def test_extra_body_that_is_not_a_new_field_object_is_rejected(tmp_path: Path, body: str) -> None:
    with pytest.raises(SystemExit) as exit_:
        _fake_run(tmp_path, "--extra-body", body)
    assert exit_.value.code == 2
    assert not (tmp_path / "run").exists()


def test_a_run_without_extra_body_records_none(tmp_path: Path) -> None:
    code, out = _fake_run(tmp_path)
    assert code == 0
    assert json.loads((out / schema.CONFIG).read_text())["extra_body"] == {}
