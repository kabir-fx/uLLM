"""O2: Prometheus metrics, read from the C2 events the hub drains (decision 0024).

Only the event loop's thread calls ``observe`` and ``render``, so the engine thread never
touches a metric (rules 2 and 5). Labels are low-cardinality, never a request id (rule 6).
"""

from collections.abc import Sequence
from typing import get_args

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from ullm.telemetry import events as ev

NAMESPACE = "ullm"

# Bucket bounds in seconds, set around M1's figures (TTFT p50 175 ms, TPOT p50 15.6 ms at c1,
# bench/results/m1); a design choice, not a measurement.
TTFT_BUCKETS = (0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
TPOT_BUCKETS = (0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.05, 0.075, 0.1, 0.25, 0.5, 1.0)
E2E_BUCKETS = (0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0)
STEP_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
QUEUE_BUCKETS = (0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)


def _values(model: type[ev.ReqFinished | ev.ReqAborted | ev.Step], field: str) -> tuple[str, ...]:
    return get_args(model.model_fields[field].annotation)


class Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()

        def histogram(
            name: str, doc: str, buckets: Sequence[float], labels: Sequence[str] = ()
        ) -> Histogram:
            return Histogram(
                name, doc, labels, namespace=NAMESPACE, registry=self.registry, buckets=buckets
            )

        def counter(name: str, doc: str, labels: Sequence[str] = ()) -> Counter:
            return Counter(name, doc, labels, namespace=NAMESPACE, registry=self.registry)

        def gauge(name: str, doc: str) -> Gauge:
            return Gauge(name, doc, namespace=NAMESPACE, registry=self.registry)

        self._ttft = histogram("ttft_seconds", "Arrival to first token.", TTFT_BUCKETS)
        self._tpot = histogram(
            "tpot_seconds", "First to last token over tokens after the first.", TPOT_BUCKETS
        )
        self._e2e = histogram(
            "e2e_seconds", "Arrival to the end of a finished request.", E2E_BUCKETS
        )
        self._step = histogram(
            "step_seconds", "One engine step, after mx.eval.", STEP_BUCKETS, ["kind"]
        )
        self._queue = histogram("queue_wait_seconds", "Arrival to admission.", QUEUE_BUCKETS)
        self._queue_depth = gauge("queue_depth", "Requests waiting for admission.")
        self._batch_size = gauge("batch_size", "Requests admitted and running.")
        self._prompt = counter("prompt_tokens", "Prompt tokens prefilled.")
        self._generation = counter("generation_tokens", "Tokens sampled.")
        self._cache_hit = counter("prefix_cache_hit_tokens", "Prompt tokens served from cache.")
        self._requests = counter("requests", "Requests finished.", ["finish_reason"])
        self._preemptions = counter("preemptions", "Requests preempted.")
        self._aborts = counter("aborts", "Requests aborted.", ["reason"])
        self._dropped = counter("telemetry_dropped", "Events the telemetry bus dropped.")

        # Every series exists from the start, so a rate over a fresh one is 0, not absent.
        for kind in _values(ev.Step, "kind"):
            self._step.labels(kind)
        for reason in _values(ev.ReqFinished, "reason"):
            self._requests.labels(reason)
        for reason in _values(ev.ReqAborted, "reason"):
            self._aborts.labels(reason)

    def observe(self, events: Sequence[ev.Event], dropped: int) -> None:
        self._dropped.inc(dropped)
        for event in events:
            match event:
                case ev.Step(kind="prefill", tokens=tokens, ms=ms):
                    self._prompt.inc(tokens)
                    self._step.labels("prefill").observe(ms / 1000)
                case ev.Step(kind="decode", rids=rids, ms=ms):
                    self._generation.inc(len(rids))
                    self._step.labels("decode").observe(ms / 1000)
                case ev.ReqFirstToken(ttft_ms=ttft_ms):
                    # The first token is counted here, not at the prefill step: a chunked
                    # prefill step (E13) will sample no token.
                    self._generation.inc()
                    self._ttft.observe(ttft_ms / 1000)
                case ev.ReqAdmitted(queue_ms=queue_ms, cached_tokens=cached):
                    self._queue.observe(queue_ms / 1000)
                    self._cache_hit.inc(cached)
                case ev.ReqFinished(reason=reason, timings=timings):
                    self._requests.labels(reason).inc()
                    self._e2e.observe(timings.e2e_ms / 1000)
                    if timings.tpot_ms is not None:
                        self._tpot.observe(timings.tpot_ms / 1000)
                case ev.ReqAborted(reason=reason):
                    self._aborts.labels(reason).inc()
                case ev.ReqPreempted():
                    self._preemptions.inc()

    def render(self, *, waiting: int, running: int) -> bytes:
        """The text exposition, with the gauges set to the engine's load now."""
        self._queue_depth.set(waiting)
        self._batch_size.set(running)
        return generate_latest(self.registry)
