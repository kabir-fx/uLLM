"""Reads Prometheus text exposition in tests (O2, decision 0024)."""

from prometheus_client.parser import text_string_to_metric_families


def metric_totals(text: str) -> dict[str, float]:
    """Each sample but histogram buckets, summed over its labels."""
    totals: dict[str, float] = {}
    for family in text_string_to_metric_families(text):
        for s in family.samples:
            if "le" not in s.labels:
                totals[s.name] = totals.get(s.name, 0.0) + s.value
    return totals
