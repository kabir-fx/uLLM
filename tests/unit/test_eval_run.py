"""Q5: an eval run never writes over a results directory that already holds files, and stops
before the server when the model is not downloaded (decision 0035)."""

from pathlib import Path

import pytest
from evals import perplexity
from evals.run import main


def test_a_results_directory_with_files_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "result.json").write_text("{}")
    assert main(["--model", "qwen3-0.6b-4bit", "--out", str(tmp_path)]) == 2
    assert "already holds" in capsys.readouterr().err
    assert [p.name for p in tmp_path.iterdir()] == ["result.json"]


def test_a_missing_model_stops_the_run_and_names_the_download(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(perplexity, "local_path", lambda spec: None)
    with pytest.raises(SystemExit, match=r"fetch_models\.py qwen3-0\.6b-4bit"):
        main(["--model", "qwen3-0.6b-4bit", "--out", str(tmp_path / "run")])
    assert not (tmp_path / "run" / "server.log").exists()
