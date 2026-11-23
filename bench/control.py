"""The control process behind the terminal's load launcher (T3, decision 0024). It runs one
load at a time against a target fixed at startup. Launched load is not a benchmark: it writes
no C5 directory, and no claim comes from it (rule 7).

    uv run python -m bench.control --target http://127.0.0.1:8000
"""

import argparse
import asyncio
import json
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Literal

import httpx2
import uvicorn
from bench.loadgen import closed_loop, open_loop
from bench.schema import Frozen, Sent, Workload, load_workload
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import Field, ValidationError
from pydantic.json_schema import models_json_schema

from ullm.server.app import LOCAL_HOSTS, TERMINAL_ORIGINS
from ullm.telemetry.export import ModelTitlesOnly

WORKLOADS = Path("bench/workloads")
PORT = 8090  # below bench's FIRST_PORT, so a session never takes it
SCHEMA_PATH = Path(__file__).with_name("control.schema.json")
OPEN_LOOP_SEED = 0
MAX_BODY_BYTES = 4096


class Closed(Frozen):
    kind: Literal["closed"]
    concurrency: int = Field(ge=1, le=64)


class Open(Frozen):
    kind: Literal["open"]
    rate_per_s: float = Field(gt=0, le=50)


class LoadSpec(Frozen):
    workload: str = Field(description="a name GET /workloads lists")
    arrival: Annotated[Closed | Open, Field(discriminator="kind")]
    max_tokens: int = Field(ge=1, le=4096, description="replaces every request's max_tokens")
    requests: int = Field(ge=1, description="the workload's first N, in order")


class WorkloadInfo(Frozen):
    name: str
    requests: int


class LoadStatus(Frozen):
    state: Literal["idle", "running", "done", "stopped", "failed"]
    spec: LoadSpec | None
    ok: int = Field(description="replies without an error; counted when the run ends")
    errors: int
    first_error: str | None


IDLE = LoadStatus(state="idle", spec=None, ok=0, errors=0, first_error=None)


def loopback_target(url: str) -> str:
    parsed = httpx2.URL(url)
    if parsed.scheme != "http" or parsed.host not in LOCAL_HOSTS:
        raise ValueError(f"the target must be an http loopback URL, not {url!r}")
    return url


def target_client(base_url: str) -> httpx2.AsyncClient:
    # httpx2's default pool holds 100 connections, which would cap an open loop's arrivals.
    limits = httpx2.Limits(max_connections=None, max_keepalive_connections=20)
    return httpx2.AsyncClient(base_url=base_url, timeout=None, limits=limits)


def create_control_app(target: httpx2.AsyncClient, workloads: Path) -> FastAPI:
    """``target`` has the server's base URL. Workloads are ``workloads/*.json``, named by stem."""
    found = {path.stem: load_workload(path) for path in sorted(workloads.glob("*.json"))}
    status = IDLE
    task: asyncio.Task[None] | None = None

    async def run(spec: LoadSpec, workload: Workload) -> None:
        nonlocal status
        requests = [
            r.model_copy(update={"max_tokens": spec.max_tokens})
            for r in workload.requests[: spec.requests]
        ]
        try:
            models = await target.get("/v1/models")
            models.raise_for_status()
            model = str(models.json()["data"][0]["id"])
            match spec.arrival:
                case Closed(concurrency=n):
                    sent = await closed_loop(
                        target, model, requests, n, workload.chat_template_kwargs, extra_body={}
                    )
                case Open(rate_per_s=rate):
                    sent = await open_loop(
                        target,
                        model,
                        requests,
                        rate,
                        OPEN_LOOP_SEED,
                        workload.chat_template_kwargs,
                        extra_body={},
                    )
        except asyncio.CancelledError:
            status = status.model_copy(update={"state": "stopped"})
            raise
        except (httpx2.HTTPError, ValueError, KeyError, IndexError, TypeError) as e:
            status = status.model_copy(
                update={"state": "failed", "first_error": f"{type(e).__name__}: {e}"}
            )
            return
        status = _ended(spec, sent)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        yield
        if task is not None:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    app = FastAPI(title="uLLM load control", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=LOCAL_HOSTS)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=TERMINAL_ORIGINS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["content-type"],
    )

    @app.get("/workloads")
    async def list_workloads() -> list[WorkloadInfo]:
        return [WorkloadInfo(name=n, requests=len(w.requests)) for n, w in found.items()]

    @app.get("/load")
    async def get_load() -> LoadStatus:
        return status

    @app.post("/load", status_code=202)
    async def start_load(request: Request) -> LoadStatus:
        nonlocal status, task
        # A page on any origin may POST text/plain with no CORS preflight; JSON needs one.
        if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
            raise HTTPException(400, "content-type must be application/json")
        body = await request.body()
        if len(body) > MAX_BODY_BYTES:
            raise HTTPException(400, f"the body is over {MAX_BODY_BYTES} bytes")
        try:
            spec = LoadSpec.model_validate_json(body)
        except ValidationError as e:
            raise HTTPException(400, _first_problem(e)) from None
        workload = found.get(spec.workload)
        if workload is None:
            raise HTTPException(400, f"no workload {spec.workload!r}; there are {sorted(found)}")
        if spec.requests > len(workload.requests):
            raise HTTPException(
                400, f"workload {spec.workload!r} has only {len(workload.requests)} requests"
            )
        if status.state == "running":
            raise HTTPException(409, "a run is running; stop it first")
        status = LoadStatus(state="running", spec=spec, ok=0, errors=0, first_error=None)
        task = asyncio.create_task(run(spec, workload))
        return status

    @app.delete("/load")
    async def stop_load() -> LoadStatus:
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        return status

    return app


def _ended(spec: LoadSpec, sent: Sequence[Sent]) -> LoadStatus:
    errors = [s.error for s in sent if s.error is not None]
    return LoadStatus(
        state="done",
        spec=spec,
        ok=len(sent) - len(errors),
        errors=len(errors),
        first_error=errors[0] if errors else None,
    )


def _first_problem(e: ValidationError) -> str:
    problem = e.errors()[0]
    where = ".".join(str(p) for p in problem["loc"]) or "body"
    return f"{where}: {problem['msg']}"


def render_schema() -> str:
    """The JSON Schema the terminal's ``src/contracts/control.ts`` is generated from."""
    _, schema = models_json_schema(
        [(LoadSpec, "validation"), (LoadStatus, "serialization"), (WorkloadInfo, "serialization")],
        schema_generator=ModelTitlesOnly,
    )
    return json.dumps({"title": "Control", **schema}, indent=2, sort_keys=True) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m bench.control", description=__doc__)
    parser.add_argument("--target", required=True, help="the server's base URL, on loopback")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--workloads", type=Path, default=WORKLOADS)
    args = parser.parse_args(argv)
    try:
        base_url = loopback_target(args.target)
    except ValueError as e:
        parser.error(str(e))

    async def serve() -> None:
        async with target_client(base_url) as target:
            app = create_control_app(target, args.workloads)
            # Local only (rule 13): no option binds another address.
            config = uvicorn.Config(app, host="127.0.0.1", port=args.port)
            await uvicorn.Server(config).serve()

    asyncio.run(serve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
