"""The control process that the terminal's launcher drives, against the real server on
FakeRunner over real sockets (decision 0024)."""

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx2
import pytest
import uvicorn
from bench.control import create_control_app, loopback_target, target_client
from bench.loadgen import open_loop
from bench.schema import Message, Source, Workload, WorkloadRequest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import StreamingResponse
from starlette.routing import Route
from tests.recording import RecordingRunner
from tests.serving import serving, serving_app

from ullm.runner.base import RunnerInfo
from ullm.runner.fake_runner import FakeRunner, FakeTokenizer, Latency
from ullm.telemetry import events as ev

INFO = RunnerInfo(model_id="fake", vocab_size=300, eos_token_ids=frozenset({0}), max_model_len=512)
FAST = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=1.0, decode_per_row_ms=0.0
)
SLOW = Latency(
    prefill_base_ms=0.2, prefill_per_token_ms=0.0, decode_base_ms=50.0, decode_per_row_ms=0.0
)
TERMINAL = "http://localhost:5173"
CLOSED = {"kind": "closed", "concurrency": 2}


def _workload(n: int) -> Workload:
    requests = []
    for i in range(n):
        content = "x" * (i + 1)
        prompt = len(FakeTokenizer().render([{"role": "user", "content": content}], {}))
        requests.append(
            WorkloadRequest(
                id=f"t-{i:03d}",
                messages=(Message(role="user", content=content),),
                max_tokens=40,
                prompt_tokens=prompt,
            )
        )
    return Workload(
        name="test",
        source=Source(dataset="none", revision="0", file="none", licence="none"),
        tokenizer="fake",
        chat_template_kwargs={"enable_thinking": False},
        seed=0,
        requests=tuple(requests),
    )


@pytest.fixture
def workloads(tmp_path: Path) -> Path:
    (tmp_path / "t.json").write_text(_workload(6).model_dump_json())
    return tmp_path


def _spec(**change: object) -> dict[str, object]:
    return {"workload": "t", "arrival": CLOSED, "max_tokens": 3, "requests": 6, **change}


@asynccontextmanager
async def _control(
    workloads: Path, latency: Latency, events: list[ev.Event]
) -> AsyncIterator[httpx2.AsyncClient]:
    async with (
        serving(RecordingRunner(FakeRunner(INFO, latency)), events) as target,
        serving_app(create_control_app(target, workloads)) as control,
    ):
        yield control


Test = Callable[[httpx2.AsyncClient, list[ev.Event]], Awaitable[None]]


def _run(test: Test, workloads: Path, latency: Latency = FAST) -> None:
    async def main() -> None:
        events: list[ev.Event] = []
        async with _control(workloads, latency, events) as control:
            await test(control, events)

    asyncio.run(asyncio.wait_for(main(), timeout=30))


async def _settled(control: httpx2.AsyncClient) -> dict[str, object]:
    while True:
        status: dict[str, object] = (await control.get("/load")).json()
        if status["state"] != "running":
            return status
        await asyncio.sleep(0.02)


async def _post(control: httpx2.AsyncClient, spec: dict[str, object]) -> httpx2.Response:
    return await control.post("/load", json=spec)


def test_it_lists_the_workloads_by_file_name(workloads: Path) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        response = await control.get("/workloads")
        assert response.status_code == 200
        assert response.json() == [{"name": "t", "requests": 6}]

    _run(test, workloads)


def test_it_is_idle_before_any_run(workloads: Path) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        status = (await control.get("/load")).json()
        assert status == {"state": "idle", "spec": None, "ok": 0, "errors": 0, "first_error": None}

    _run(test, workloads)


@pytest.mark.parametrize("arrival", [CLOSED, {"kind": "open", "rate_per_s": 40.0}])
def test_a_run_sends_the_first_n_requests_with_its_max_tokens(
    workloads: Path, arrival: dict[str, object]
) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        started = await _post(control, _spec(arrival=arrival, requests=4))
        assert started.status_code == 202
        assert started.json()["state"] == "running"
        status = await _settled(control)
        assert status["state"] == "done"
        assert (status["ok"], status["errors"], status["first_error"]) == (4, 0, None)
        assert status["spec"] == _spec(arrival=arrival, requests=4)
        arrived = [e for e in events if isinstance(e, ev.ReqArrived)]
        assert len(arrived) == 4
        assert {e.max_tokens for e in arrived} == {3}
        assert sum(isinstance(e, ev.ReqFinished) for e in events) == 4

    _run(test, workloads)


def test_a_second_start_while_one_runs_is_refused(workloads: Path) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        assert (await _post(control, _spec(max_tokens=20))).status_code == 202
        second = await _post(control, _spec())
        assert second.status_code == 409
        assert "running" in second.json()["detail"]
        assert (await control.delete("/load")).json()["state"] == "stopped"
        assert (await _post(control, _spec())).status_code == 202
        assert (await _settled(control))["state"] == "done"

    _run(test, workloads, SLOW)


def test_a_stop_closes_the_streams_so_the_server_aborts_them(workloads: Path) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        assert (await _post(control, _spec(max_tokens=30))).status_code == 202
        while not any(isinstance(e, ev.ReqFirstToken) for e in events):
            await asyncio.sleep(0.01)
        stopped = (await control.delete("/load")).json()
        assert stopped["state"] == "stopped"
        assert (await control.get("/load")).json()["state"] == "stopped"
        while sum(isinstance(e, ev.ReqAborted) for e in events) < 2:
            await asyncio.sleep(0.01)
        arrived = sum(isinstance(e, ev.ReqArrived) for e in events)
        await asyncio.sleep(0.3)
        assert sum(isinstance(e, ev.ReqArrived) for e in events) == arrived
        assert not any(isinstance(e, ev.ReqFinished) for e in events)

    _run(test, workloads, SLOW)


def test_a_stop_with_nothing_running_changes_nothing(workloads: Path) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        assert (await control.delete("/load")).json()["state"] == "idle"
        await _post(control, _spec(requests=1))
        await _settled(control)
        assert (await control.delete("/load")).json()["state"] == "done"

    _run(test, workloads)


def test_refused_replies_are_counted_with_the_first_error(workloads: Path) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        # The fake model's context is 512 tokens, so this max_tokens makes every request a 400.
        await _post(control, _spec(max_tokens=600, requests=3))
        status = await _settled(control)
        assert (status["state"], status["ok"], status["errors"]) == ("done", 0, 3)
        assert str(status["first_error"]).startswith("HTTP 400")

    _run(test, workloads)


@pytest.mark.parametrize(
    "change",
    [
        {"workload": "nope"},
        {"workload": "../t"},
        {"requests": 0},
        {"requests": 7},
        {"max_tokens": 0},
        {"max_tokens": 4097},
        {"arrival": {"kind": "closed", "concurrency": 0}},
        {"arrival": {"kind": "closed", "concurrency": 65}},
        {"arrival": {"kind": "open", "rate_per_s": 0}},
        {"arrival": {"kind": "open", "rate_per_s": 50.5}},
        {"arrival": {"kind": "open", "concurrency": 2}},
        {"target": "http://example.com"},
    ],
)
def test_a_bad_spec_is_a_400_and_starts_nothing(workloads: Path, change: dict[str, object]) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        response = await _post(control, _spec(**change))
        assert response.status_code == 400
        assert isinstance(response.json()["detail"], str)
        assert (await control.get("/load")).json()["state"] == "idle"

    _run(test, workloads)


def test_a_body_that_is_not_json_is_refused(workloads: Path) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        body = json.dumps(_spec())
        plain = await control.post("/load", content=body, headers={"content-type": "text/plain"})
        assert plain.status_code == 400
        assert (await control.get("/load")).json()["state"] == "idle"

    _run(test, workloads)


def test_another_host_name_is_refused(workloads: Path) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        rebound = await control.post("/load", json=_spec(), headers={"host": "evil.example"})
        assert rebound.status_code == 400
        assert (await control.get("/load")).json()["state"] == "idle"

    _run(test, workloads)


@pytest.mark.parametrize(("origin", "allowed"), [(TERMINAL, True), ("http://evil.example", False)])
def test_only_the_terminal_origin_passes_the_preflight(
    workloads: Path, origin: str, allowed: bool
) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        for method in ("POST", "DELETE"):
            preflight = await control.options(
                "/load",
                headers={
                    "origin": origin,
                    "access-control-request-method": method,
                    "access-control-request-headers": "content-type",
                },
            )
            assert (preflight.status_code == 200) is allowed
            assert (preflight.headers.get("access-control-allow-origin") == origin) is allowed

    _run(test, workloads)


def test_a_target_that_does_not_answer_fails_the_run(workloads: Path) -> None:
    async def main() -> None:
        async with (
            httpx2.AsyncClient(base_url="http://127.0.0.1:9") as target,
            serving_app(create_control_app(target, workloads)) as control,
        ):
            assert (await _post(control, _spec())).status_code == 202
            status = await _settled(control)
            assert status["state"] == "failed"
            assert status["first_error"]

    asyncio.run(asyncio.wait_for(main(), timeout=30))


@pytest.mark.parametrize("url", ["http://127.0.0.1:8000", "http://localhost:8000"])
def test_a_loopback_target_is_accepted(url: str) -> None:
    assert loopback_target(url) == url


@pytest.mark.parametrize(
    "url", ["http://example.com:8000", "http://10.0.0.2:8000", "ftp://127.0.0.1"]
)
def test_any_other_target_is_refused(url: str) -> None:
    with pytest.raises(ValueError, match="loopback"):
        loopback_target(url)


def test_a_body_over_the_limit_is_refused(workloads: Path) -> None:
    async def test(control: httpx2.AsyncClient, events: list[ev.Event]) -> None:
        body = json.dumps(_spec()) + " " * 4096
        big = await control.post(
            "/load", content=body, headers={"content-type": "application/json"}
        )
        assert big.status_code == 400
        assert (await control.get("/load")).json()["state"] == "idle"

    _run(test, workloads)


def test_an_error_nobody_foresaw_fails_the_run_and_frees_the_slot(workloads: Path) -> None:
    def deep(request: httpx2.Request) -> httpx2.Response:
        # Nesting this deep makes the JSON parser raise RecursionError.
        return httpx2.Response(200, text="[" * 100_000 + "]" * 100_000)

    async def main() -> None:
        transport = httpx2.MockTransport(deep)
        async with (
            httpx2.AsyncClient(base_url="http://127.0.0.1:9", transport=transport) as target,
            serving_app(create_control_app(target, workloads)) as control,
        ):
            assert (await _post(control, _spec())).status_code == 202
            status = await _settled(control)
            assert status["state"] == "failed"
            assert "RecursionError" in str(status["first_error"])
            assert (await _post(control, _spec())).status_code == 202

    asyncio.run(asyncio.wait_for(main(), timeout=30))


def test_the_target_client_lets_the_open_loop_hold_more_than_100_open() -> None:
    held = 0
    peak = 0

    async def chat(request: Request) -> StreamingResponse:
        nonlocal held, peak
        await request.body()
        held += 1
        peak = max(peak, held)

        async def reply() -> AsyncIterator[bytes]:
            nonlocal held
            try:
                await asyncio.sleep(1.0)
                yield b'data: {"choices":[{"delta":{"content":"x"},"finish_reason":"stop"}]}\n\n'
                yield b'data: {"choices":[],"usage":{"prompt_tokens":1,"completion_tokens":1}}\n\n'
                yield b"data: [DONE]\n\n"
            finally:
                held -= 1

        return StreamingResponse(reply(), media_type="text/event-stream")

    stub = Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"])])
    requests = _workload(120).requests

    async def main() -> None:
        server = uvicorn.Server(uvicorn.Config(stub, host="127.0.0.1", port=0, log_level="warning"))
        task = asyncio.create_task(server.serve())
        while not server.started:
            await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        try:
            async with target_client(f"http://127.0.0.1:{port}") as target:
                sent = await open_loop(target, "m", requests, 1000.0, 0, {}, extra_body={})
        finally:
            server.should_exit = True
            await task
        assert all(s.error is None for s in sent)

    asyncio.run(asyncio.wait_for(main(), timeout=30))
    assert peak == 120
