"""O3: the provisioned Grafana dashboard reads only metrics that ``/metrics`` exports."""

import json
import re
from pathlib import Path

from prometheus_client.parser import text_string_to_metric_families

from ullm.core.block_manager import KvUsage
from ullm.telemetry.metrics import Metrics

DASHBOARD = Path(__file__).parents[2] / "deploy/grafana/dashboards/ullm.json"


def _exported() -> set[str]:
    text = (
        Metrics()
        .render(waiting=0, running=0, kv=KvUsage.idle(num_blocks=1, block_size=16))
        .decode()
    )
    return {s.name for family in text_string_to_metric_families(text) for s in family.samples}


def _queries() -> list[str]:
    panels = json.loads(DASHBOARD.read_text())["panels"]
    return [target["expr"] for panel in panels for target in panel["targets"]]


def test_every_metric_the_dashboard_queries_is_exported() -> None:
    queried = {name for expr in _queries() for name in re.findall(r"\bullm_\w+", expr)}
    assert queried
    assert queried - _exported() == set()


def test_percentiles_come_from_histograms() -> None:
    latency = [expr for expr in _queries() if "_seconds" in expr]
    assert latency
    assert all("histogram_quantile" in expr and "_bucket" in expr for expr in latency)


def test_the_kv_usage_panel_shows_the_pool_in_use_cached_and_wasted() -> None:
    panels = json.loads(DASHBOARD.read_text())["panels"]
    (kv,) = [p for p in panels if p["title"] == "KV usage"]
    exprs = " ".join(t["expr"] for t in kv["targets"])
    assert 'state="in_use"' in exprs
    assert 'state="cached"' in exprs
    assert "ullm_kv_waste_ratio" in exprs
    assert kv["fieldConfig"]["defaults"]["unit"] == "percentunit"
