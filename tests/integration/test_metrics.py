"""O2: ``GET /metrics`` on a real uvicorn server (decision 0024)."""

import asyncio
from collections.abc import Awaitable, Callable

from prometheus_client.parser import text_string_to_metric_families
from tests.prometheus import metric_totals
from tests.recording import RecordingRunner
from tests.serving import SERVED, serving, serving_app

from ullm.core.engine import EngineConfig
from ullm.runner.base import RunnerInfo
from ullm.runner.fake_runner import FakeRunner, FakeTokenizer, Latency
from ullm.server.app import ServeConfig, create_app
from ullm.telemetry import events as ev

INFO = RunnerInfo(model_id="fake", vocab_size=300, eos_token_ids=frozenset({0}), max_model_len=4096)
FAST = Latency(
    prefill_base_ms=1.0, prefill_per_token_ms=0.0, decode_base_ms=1.0, decode_per_row_ms=0.0
)


def _run(main: Callable[[], Awaitable[None]]) -> None:
    asyncio.run(asyncio.wait_for(main(), timeout=20))


def test_a_scrape_right_after_each_reply_already_counts_it() -> None:
    async def main() -> None:
        app = create_app(FakeRunner(INFO, FAST), FakeTokenizer(), ServeConfig(served_model=SERVED))
        prompt = completion = 0
        async with serving_app(app) as client:
            for i in range(1, 6):
                reply = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": SERVED,
                        "messages": [{"role": "user", "content": "hi " * i}],
                        "max_tokens": 3 + i,
                    },
                )
                usage = reply.json()["usage"]
                prompt += usage["prompt_tokens"]
                completion += usage["completion_tokens"]
                scrape = await client.get("/metrics")
                assert scrape.headers["content-type"].startswith("text/plain")
                totals = metric_totals(scrape.text)
                assert totals["ullm_requests_total"] == i
                assert totals["ullm_prompt_tokens_total"] == prompt
                assert totals["ullm_generation_tokens_total"] == completion
                assert totals["ullm_e2e_seconds_count"] == i
                assert (totals["ullm_queue_depth"], totals["ullm_batch_size"]) == (0, 0)

    _run(main)


def test_prometheus_in_docker_may_scrape_metrics_and_nothing_else() -> None:
    async def main() -> None:
        app = create_app(FakeRunner(INFO, FAST), FakeTokenizer(), ServeConfig(served_model=SERVED))
        async with serving_app(app) as client:
            docker = {"host": "host.docker.internal:8000"}
            scrape = await client.get("/metrics", headers=docker)
            assert scrape.status_code == 200
            assert metric_totals(scrape.text)["ullm_requests_total"] == 0
            foreign = await client.get("/metrics", headers={"host": "evil.test"})
            assert foreign.status_code == 400
            refused = [
                await client.get(path, headers=docker, timeout=2)
                for path in ("/health", "/v1/models", "/telemetry/stream", "/metrics/")
            ]
            chat = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}]}
            refused.append(await client.post("/v1/chat/completions", json=chat, headers=docker))
            assert [(r.status_code, r.text) for r in refused] == [(400, "Invalid host header")] * 5
            assert (
                await client.get("/health", headers={"host": "localhost:8000"})
            ).status_code == 200

    _run(main)


def test_an_app_without_its_own_telemetry_serves_no_metrics() -> None:
    async def main() -> None:
        events: list[ev.Event] = []
        async with serving(RecordingRunner(FakeRunner(INFO, FAST)), events) as client:
            assert (await client.get("/metrics")).status_code == 404

    _run(main)


def test_the_gauges_show_a_running_request_and_one_waiting_behind_it() -> None:
    slow = Latency(
        prefill_base_ms=1.0, prefill_per_token_ms=0.0, decode_base_ms=50.0, decode_per_row_ms=0.0
    )
    config = ServeConfig(served_model=SERVED, engine=EngineConfig(max_num_seqs=1))
    body = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}

    async def main() -> None:
        app = create_app(FakeRunner(INFO, slow), FakeTokenizer(), config)
        async with serving_app(app) as client:

            async def load_reaches(want: tuple[float, float]) -> tuple[float, float]:
                for _ in range(200):
                    totals = metric_totals((await client.get("/metrics")).text)
                    seen = (totals["ullm_queue_depth"], totals["ullm_batch_size"])
                    if seen == want:
                        break
                    await asyncio.sleep(0.01)
                return seen

            def post() -> asyncio.Task[object]:
                return asyncio.create_task(client.post("/v1/chat/completions", json=body))

            first = post()
            assert await load_reaches((0, 1)) == (0, 1)
            second = post()
            assert await load_reaches((1, 1)) == (1, 1)
            await asyncio.gather(first, second)

    _run(main)


def test_the_kv_gauges_show_a_running_requests_blocks_and_an_idle_pool_after() -> None:
    slow = Latency(
        prefill_base_ms=1.0, prefill_per_token_ms=0.0, decode_base_ms=20.0, decode_per_row_ms=0.0
    )
    config = ServeConfig(served_model=SERVED, engine=EngineConfig(max_num_seqs=1))
    body = {"model": SERVED, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 20}
    pool = -(-EngineConfig().max_kv_slots // INFO.block_size)

    async def main() -> None:
        app = create_app(FakeRunner(INFO, slow), FakeTokenizer(), config)
        async with serving_app(app) as client:

            async def kv() -> dict[str, float]:
                text = (await client.get("/metrics")).text
                states = {
                    f"kv_{s.labels['state']}": s.value
                    for family in text_string_to_metric_families(text)
                    for s in family.samples
                    if s.name == "ullm_kv_blocks"
                }
                return metric_totals(text) | states

            reply = asyncio.create_task(client.post("/v1/chat/completions", json=body))
            for _ in range(200):
                running = await kv()
                if running["ullm_batch_size"] == 1:
                    break
                await asyncio.sleep(0.01)
            assert running["kv_in_use"] >= 1
            assert running["ullm_kv_blocks"] == pool  # free + in_use + cached
            # Ten requests of max_model_len 4,096 fit the pool; max_num_seqs caps it at one.
            assert running["ullm_kv_max_concurrent_seqs"] == 1
            await reply
            idle = await kv()
            assert (idle["kv_free"], idle["kv_in_use"], idle["ullm_kv_waste_ratio"]) == (pool, 0, 0)

    _run(main)
