"""`bench.readme.table` keeps out of the README any figure that rule 8 or decision 0030 would
not let it claim. Sessions are stubs and `ratios` is replaced, so no committed run loads."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
from bench import readme
from bench.analysis import Session
from bench.paired import Estimate

BASELINE = "uv run python scripts/mlx_lm_server.py --model {model} --prompt-cache-size 0"
ULLM = "uv run ullm serve --port {port} --max-num-seqs 16"


def _session(concurrency: int, **change: Any) -> Session:
    s: dict[str, Any] = {
        "verdict": "VALID",
        "baseline": "mlx-lm-batched-no-cache",
        "twin": "mlx-lm-batched-no-cache-twin",
        "commit": "c51200e208123ab0608558839ba50fa539efcf9d",
        "dirty": False,
        "mlx": "0.32.4",
        "model": "mlx-community/Qwen3-0.6B-bf16",
        "baseline_launch": BASELINE,
        "ullm_launch": ULLM,
    } | change
    return cast(
        Session,
        SimpleNamespace(
            verdict=s["verdict"],
            baseline=s["baseline"],
            twin=s["twin"],
            group=SimpleNamespace(
                concurrency=concurrency,
                baseline=SimpleNamespace(name=s["baseline"], launch=s["baseline_launch"]),
                engines=(SimpleNamespace(name="ullm", launch=s["ullm_launch"]),),
            ),
            env=SimpleNamespace(
                commit=s["commit"],
                dirty=s["dirty"],
                chip="Apple M4",
                memory_gb=16.0,
                macos="27.2.1",
                libraries={"mlx": s["mlx"], "mlx-lm": "0.32.2"},
            ),
            info=SimpleNamespace(served_models={"ullm": s["model"]}),
        ),
    )


@pytest.fixture(autouse=True)
def fixed_ratios(monkeypatch: pytest.MonkeyPatch) -> None:
    def ratios(session: Session, engine: str, regime: object) -> dict[str, Estimate]:
        point = 1.0 if engine == session.twin else 1.2
        return {m: Estimate(point, point - 0.01, point + 0.01, 12, 3) for m in readme.ROWS}

    monkeypatch.setattr(readme, "ratios", ratios)


def test_a_table_of_valid_sessions_has_one_column_per_concurrency_in_order() -> None:
    text = readme.table([_session(4), _session(1)])
    assert text.splitlines()[0].endswith("| c1 | c4 |")
    assert "| TTFT | 1.200 [1.190, 1.210] | 1.200 [1.190, 1.210] |" in text


@pytest.mark.parametrize("verdict", ["DISCARDED", "SMOKE"])
def test_a_session_that_is_not_valid_is_refused(verdict: str) -> None:
    with pytest.raises(SystemExit, match=f"VALID sessions only: c2 {verdict}"):
        readme.table([_session(1), _session(2, verdict=verdict)])


def test_two_baselines_are_refused() -> None:
    with pytest.raises(SystemExit, match="one baseline"):
        readme.table([_session(1), _session(2, baseline="mlx-lm-sequential-no-cache")])


@pytest.mark.parametrize(
    "change",
    [
        {"commit": "41671f1"},
        {"dirty": True},
        {"mlx": "0.31.0"},
        {"model": "mlx-community/Qwen3-1.7B-bf16"},
        {"baseline_launch": BASELINE.replace(" --prompt-cache-size 0", "")},
        {"ullm_launch": ULLM.replace("16", "8")},
    ],
)
def test_sessions_with_another_setup_are_refused(change: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="one setup"):
        readme.table([_session(1), _session(2, **change)])


def test_the_conditions_say_when_the_tree_was_dirty() -> None:
    assert "commit `c51200e`, dirty tree" in readme.table([_session(1, dirty=True)])
    assert "dirty" not in readme.table([_session(1)])
