"""Q5: what an eval run writes and which server it starts, and when it refuses to start
(decision 0036)."""

import json
from collections.abc import Sequence
from pathlib import Path

import pytest
from evals import data, exact_match, perplexity
from evals.data import Item
from evals.run import main
from evals.schema import EvalResult, ExactMatch, ItemResult, Perplexity

PPL = Perplexity(
    window=1024, windows=2, tokens=2046, mean_nll=3.0, perplexity=20.0, standard_error=0.1
)


def test_a_run_starts_one_sequence_at_a_time_and_writes_every_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launches: list[str] = []

    class Tokenizer:
        def encode(self, text: str) -> list[int]:
            return list(range(3000))

    def fake_run(
        launch: str, port: int, items: Sequence[Item], log: Path
    ) -> tuple[ExactMatch, list[ItemResult]]:
        launches.append(launch)
        log.write_text("served\n")
        results = [
            ItemResult(
                index=i.index,
                gold=str(i.gold),
                answer=None,
                correct=False,
                finish_reason="stop",
                completion_tokens=5,
                reply="",
            )
            for i in items
        ]
        return ExactMatch.of(results, 2.0, 123), results

    monkeypatch.setattr(perplexity, "load", lambda spec: (object(), Tokenizer()))
    monkeypatch.setattr(perplexity, "measure", lambda model, chunks: PPL)
    monkeypatch.setattr(exact_match, "run", fake_run)
    out = tmp_path / "run"

    assert main(["--model", "qwen3-0.6b-4bit", "--out", str(out)]) == 0

    [launch] = launches
    assert "serve --model qwen3-0.6b-4bit" in launch
    assert "--max-num-seqs 1" in launch.split(" --port")[1]
    assert sorted(p.name for p in out.iterdir()) == [
        "env.json",
        "items.jsonl",
        "result.json",
        "server.log",
    ]
    result = EvalResult.model_validate_json((out / "result.json").read_text())
    assert (result.model, result.dtype, result.perplexity) == ("qwen3-0.6b-4bit", "4bit", PPL)
    assert result.exact_match.items == 200
    assert result.datasets == data.SHA256
    lines = (out / "items.jsonl").read_text().splitlines()
    assert [json.loads(line)["index"] for line in lines] == list(range(200))


def test_a_results_directory_with_files_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "result.json").write_text("{}")
    assert main(["--model", "qwen3-0.6b-4bit", "--out", str(tmp_path)]) == 2
    assert "already holds" in capsys.readouterr().err
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_a_file_in_place_of_the_directory_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    taken = tmp_path / "taken"
    taken.write_text("")
    assert main(["--model", "qwen3-0.6b-4bit", "--out", str(taken)]) == 2
    assert "not a directory" in capsys.readouterr().err


def test_a_missing_model_stops_the_run_and_names_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(perplexity, "local_path", lambda spec: None)
    with pytest.raises(SystemExit, match=r"fetch_models\.py qwen3-0\.6b-4bit"):
        main(["--model", "qwen3-0.6b-4bit", "--out", str(tmp_path / "run")])
    assert not (tmp_path / "run" / "server.log").exists()
