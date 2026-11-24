"""One benchmark session: engines take turns on the same prompts, block by block, while
powermetrics logs the GPU (decision 0016). The only writer of a session directory.

    uv run python -m bench.session bench/groups/m1-c1.json --settle 240 --out bench/results/m1/c1
    uv run python -m bench.session bench/groups/m1-c8.json --smoke --blocks 1 --out <dir>

Exits 0 when the session is VALID, 1 when it is not, and 2 when it refuses to start.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Literal

import httpx2
from bench import analysis, conditions, schema
from bench.loadgen import closed_loop
from bench.plan import Plan, plan
from bench.schema import (
    EngineSpec,
    Group,
    Probe,
    Sent,
    SessionInfo,
    SessionRecord,
    Workload,
    WorkloadRequest,
)
from bench.target import Target, port_in_use, ready
from pydantic import JsonValue

WORKLOAD = Path("bench/workloads/a.json")
MODEL = "{model}"
PORT = "{port}"
OUT = "{out}"
BLOCK = "{block}"


class ServerFailed(Exception):
    """A server that exited or never became ready."""


@dataclass(frozen=True)
class Live:
    """A server that answered ``/health`` and named the model to send."""

    target: Target
    client: httpx2.AsyncClient
    log: IO[bytes]
    served: str
    extra_body: dict[str, JsonValue]

    async def send(
        self, requests: Sequence[WorkloadRequest], concurrency: int, kwargs: dict[str, bool]
    ) -> list[Sent]:
        return await closed_loop(
            self.client, self.served, requests, concurrency, kwargs, extra_body=self.extra_body
        )

    async def stop(self) -> None:
        await self.client.aclose()
        self.target.stop()
        self.log.close()


@dataclass
class Server:
    spec: EngineSpec
    port: int
    launch: str
    model: str | None
    log_path: Path
    live: Live | None = None
    served: str | None = None

    async def start(self, timeout_s: float, block: int | None = None) -> Live:
        log = self.log_path.open("ab")
        launch = self.launch if block is None else self.launch.replace(BLOCK, str(block))
        target = Target(launch, log)
        client = httpx2.AsyncClient(base_url=f"http://127.0.0.1:{self.port}", timeout=None)
        try:
            served = await ready(client, target, timeout_s, self.model)
        except (SystemExit, asyncio.CancelledError) as e:
            await client.aclose()
            target.stop()
            log.close()
            if isinstance(e, SystemExit):
                raise ServerFailed(f"{self.spec.name}: {e}") from None
            raise
        self.served = served
        self.live = Live(target, client, log, served, self.spec.extra_body)
        return self.live

    async def stop(self) -> None:
        if self.live is not None:
            await self.live.stop()
            self.live = None


def _servers(group: Group, out: Path, model_dir: str | None) -> dict[str, Server]:
    servers = {}
    for i, spec in enumerate(group.servers_in_order):
        port = schema.FIRST_PORT + i
        launch = spec.launch.replace(PORT, str(port)).replace(OUT, shlex.quote(str(out)))
        model = spec.model
        if model_dir is not None:
            launch = launch.replace(MODEL, model_dir)
            model = None if model is None else model.replace(MODEL, model_dir)
        servers[spec.name] = Server(spec, port, launch, model, out / f"{spec.name}.log")
    return servers


def _model_dir(group: Group) -> str | None:
    """The pinned snapshot's path, when a launch or a model names it."""
    specs = group.servers_in_order
    if not any(MODEL in s.launch or s.model == MODEL for s in specs):
        return None
    from scripts.fetch_models import local_path

    from ullm.models import DEFAULT_MODEL

    path = local_path(DEFAULT_MODEL)
    if path is None:
        raise LookupError("no local snapshot of the default model: run make models")
    return str(path)


def _plan_json(p: Plan, workload_path: Path) -> str:
    return json.dumps(
        {
            "workload": str(workload_path),
            "workload_sha256": hashlib.sha256(workload_path.read_bytes()).hexdigest(),
            "blocks": [[r.id for r in block] for block in p.blocks],
            "warmup": [r.id for r in p.warmup],
            "orders": [list(o) for o in p.orders],
        },
        indent=2,
    )


def _p50(values: Sequence[float]) -> str:
    return f"{statistics.median(values):.1f}" if values else "n/a"


class Session:
    def __init__(
        self,
        group: Group,
        p: Plan,
        workload: Workload,
        servers: dict[str, Server],
        out: Path,
        args: argparse.Namespace,
    ) -> None:
        self.group, self.plan, self.servers, self.args = group, p, servers, args
        self.kwargs = workload.chat_template_kwargs
        self.raw = (out / schema.RAW).open("w")
        self.probes = (out / schema.PROBES).open("w")
        self.blocks_run = 0
        # Interrupted until the last block ends: a crash leaves it so.
        self.ended: Literal["completed", "failed request", "interrupted"] = "interrupted"

    def close(self) -> None:
        self.raw.close()
        self.probes.close()

    def record(
        self,
        engine: str,
        block: int,
        pos: int,
        phase: Literal["liveness", "warmup", "measure"],
        sent: Sequence[Sent],
    ) -> None:
        for s in sent:
            row = SessionRecord(engine=engine, block=block, pos=pos, phase=phase, sent=s)
            self.raw.write(row.model_dump_json() + "\n")
        self.raw.flush()

    def probe(self, block: int, engine: str | None, live: Live | None) -> None:
        memory = None if live is None else conditions.footprint(live.target.proc.pid)
        row = Probe(
            at_unix=time.time(),
            block=block,
            engine=engine,
            footprint_bytes=None if memory is None else memory[0],
            peak_bytes=None if memory is None else memory[1],
            swap_used_bytes=conditions.swap_used(),
            power=conditions.power(),
        )
        self.probes.write(row.model_dump_json() + "\n")
        self.probes.flush()

    async def run(self) -> None:
        warmup = self.plan.warmup[: self.group.warmup_per_chunk]
        concurrency = self.group.concurrency
        try:
            self.probe(-1, None, None)
            await asyncio.sleep(self.args.settle)
            if self.group.servers == "resident":
                live = {n: await s.start(self.args.ready_timeout) for n, s in self.servers.items()}
                for name, server in live.items():
                    sent = await server.send(warmup, concurrency, self.kwargs)
                    self.record(name, -1, -1, "liveness", sent)
                    if any(s.error is not None for s in sent):
                        self.ended = "failed request"
                        return
            for block in range(min(self.args.blocks, len(self.plan.blocks))):
                for pos, name in enumerate(self.plan.order(block)):
                    if not await self.chunk(block, pos, self.servers[name], warmup):
                        self.ended = "failed request"
                        return
                self.blocks_run += 1
            self.ended = "completed"
        except ServerFailed as e:
            print(f"server failed: {e}", file=sys.stderr)
            self.ended = "failed request"
        except asyncio.CancelledError:
            pass

    async def chunk(
        self, block: int, pos: int, server: Server, warmup: Sequence[WorkloadRequest]
    ) -> bool:
        """One engine's turn: start it if it runs per chunk, warm it up, send the block, probe.
        False when a request failed."""
        per_chunk = self.group.servers == "per_chunk"
        live = server.live or await server.start(self.args.ready_timeout, block)
        name, concurrency = server.spec.name, self.group.concurrency
        try:
            warm = await live.send(warmup, concurrency, self.kwargs)
            self.record(name, block, pos, "warmup", warm)
            measured = await live.send(self.plan.blocks[block], concurrency, self.kwargs)
            self.record(name, block, pos, "measure", measured)
            self.probe(block, name, live)
        finally:
            if per_chunk:
                await server.stop()
        failed = sum(s.error is not None for s in (*warm, *measured))
        ttfts = [v for s in measured if (v := analysis.per_request("TTFT", s)) is not None]
        tpots = [v for s in measured if (v := analysis.per_request("TPOT", s)) is not None]
        print(
            f"block {block} {name} place {pos}: {len(measured)} requests, {failed} failed, "
            f"TTFT p50 {_p50(ttfts)} ms, "
            f"TPOT p50 {_p50(tpots)} ms",
            flush=True,
        )
        return failed == 0


def _gzip(plist: Path) -> None:
    """Replace the powermetrics log with its gzip: about 32 KB a sample raw, 2 KB gzipped."""
    if plist.exists():
        gz = plist.with_name(schema.POWERMETRICS_GZ)
        gz.write_bytes(gzip.compress(plist.read_bytes(), mtime=0))
        plist.unlink()


def _refusals(group: Group, out: Path, smoke: bool) -> list[str]:
    found = []
    if not smoke:
        found += conditions.violations(
            conditions.power(),
            sudo=conditions.can_sudo(),
            containers=conditions.containers(),
            powermetrics_running=conditions.powermetrics_running(),
        )
    for i, spec in enumerate(group.servers_in_order):
        url = f"http://127.0.0.1:{schema.FIRST_PORT + i}"
        if port_in_use(url):
            found.append(f"something already listens at {url}, where {spec.name} would")
    if out.exists():
        found.append(f"{out} exists: a session never writes over another")
    return found


async def _run(session: Session) -> None:
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    assert task is not None
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        await session.run()
    finally:
        for server in session.servers.values():
            await server.stop()


def run(args: argparse.Namespace) -> int:
    group = Group.model_validate_json(args.group.read_text())
    workload = schema.load_workload(args.workload)
    names = [s.name for s in group.servers_in_order]
    try:
        p = plan(
            workload.requests, names, group.prompts_per_chunk, group.warmup_per_chunk, group.seed
        )
    except ValueError as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 2
    broken = _refusals(group, args.out, args.smoke)
    if broken:
        print("refusing to start:", file=sys.stderr)
        for problem in broken:
            print(f"  - {problem}", file=sys.stderr)
        if not args.smoke:
            print("use --smoke for a session that is not B5", file=sys.stderr)
        return 2
    try:
        model_dir = _model_dir(group)
    except LookupError as e:
        print(f"refusing to start: {e}", file=sys.stderr)
        return 2
    out = args.out
    out.mkdir(parents=True)
    started = time.time()
    (out / schema.GROUP).write_text(group.model_dump_json(indent=2) + "\n")
    (out / schema.PLAN).write_text(_plan_json(p, args.workload) + "\n")
    env = conditions.environment(
        used_powermetrics=True, started=datetime.fromtimestamp(started, UTC)
    ).model_dump_json(indent=2)
    (out / schema.ENV).write_text(env + "\n")
    servers = _servers(group, out, model_dir)
    session = Session(group, p, workload, servers, out, args)
    awake = None
    if shutil.which("caffeinate"):
        awake = subprocess.Popen(["caffeinate", "-ims", "-w", str(os.getpid())])
    powermetrics = conditions.PowerMetrics(out / schema.POWERMETRICS)
    try:
        asyncio.run(_run(session))
    finally:
        powermetrics.stop()
        if awake is not None:
            awake.terminate()
        session.close()
        _gzip(out / schema.POWERMETRICS)
        info = SessionInfo(
            served_models={n: s.served for n, s in servers.items() if s.served is not None},
            smoke=args.smoke,
            settle_s=args.settle,
            started_unix=started,
            ended_unix=time.time(),
            blocks_planned=len(p.blocks),
            blocks_run=session.blocks_run,
            ended=session.ended,
        )
        (out / schema.SESSION).write_text(info.model_dump_json(indent=2) + "\n")
    result = analysis.load_session(out)
    analysis.write_session(result)
    print((out / schema.SUMMARY).read_text())
    return 0 if result.verdict == "VALID" else 1


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not at least 1")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.session")
    parser.add_argument("group", type=Path, help="a group file, as bench/groups/m1-c1.json")
    parser.add_argument("--out", type=Path, required=True, help="the session directory, new")
    parser.add_argument("--workload", type=Path, default=WORKLOAD)
    parser.add_argument("--settle", type=float, default=240.0, help="idle seconds, logged, first")
    parser.add_argument("--blocks", type=_positive, default=sys.maxsize, help="the first N blocks")
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--smoke", action="store_true", help="skip the B5 check; never VALID")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
