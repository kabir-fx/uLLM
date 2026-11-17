"""One benchmark run, end to end, into a C5 directory (B1, B5, decision 0013).

    uv run python -m bench.run --target ullm --launch "uv run ullm serve"
    uv run python -m bench.run --target ullm --launch "uv run ullm serve" --smoke --requests 16

A full run refuses to start unless B5 holds. ``--smoke`` skips that check and says so in
config.json and summary.md.
"""

import argparse
import asyncio
import hashlib
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import IO

import httpx2
from bench import analysis, conditions, schema
from bench.loadgen import TEMPERATURE, closed_loop
from bench.schema import RequestRecord, RunConfig, WorkloadRequest

WORKLOAD = Path("bench/workloads/a.json")
RESULTS = Path("bench/results")
SCRATCH = Path("bench/.scratch")


class Target:
    """The server under test, started from a command line in its own process group."""

    def __init__(self, command: str, log: IO[bytes]) -> None:
        self.proc = subprocess.Popen(
            shlex.split(command), stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )

    def stop(self) -> None:
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGINT)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()


def _port_in_use(base_url: str) -> bool:
    url = httpx2.URL(base_url)
    try:
        with socket.create_connection((url.host, url.port or 80), timeout=1):
            return True
    except OSError:
        return False


async def _ready(client: httpx2.AsyncClient, target: Target | None, timeout_s: float) -> str:
    """Wait for ``GET /health``, then return the served model id from ``GET /v1/models``."""
    deadline = time.monotonic() + timeout_s
    while True:
        if target is not None and target.proc.poll() is not None:
            raise SystemExit(f"the target exited with {target.proc.returncode} before it was ready")
        try:
            healthy = (await client.get("/health")).status_code == 200
        except httpx2.TransportError:
            healthy = False
        if healthy:
            break
        if time.monotonic() > deadline:
            raise SystemExit(f"the target was not healthy after {timeout_s:g} s")
        await asyncio.sleep(0.5)
    models = (await client.get("/v1/models")).json()
    return str(models["data"][0]["id"])


async def _measure(
    args: argparse.Namespace,
    requests: Sequence[WorkloadRequest],
    template_kwargs: dict[str, bool],
    target: Target | None,
    out: Path,
) -> tuple[str, list[RequestRecord]]:
    async with httpx2.AsyncClient(base_url=args.base_url, timeout=None) as client:
        model = await _ready(client, target, args.ready_timeout)
        start = time.perf_counter()
        records = await closed_loop(
            client,
            model,
            requests[: args.warmup],
            args.concurrency,
            template_kwargs,
            "warmup",
            -1,
            start,
        )
        for repeat in range(args.repeats):
            if repeat:
                await asyncio.sleep(args.cooldown)
            records += await closed_loop(
                client,
                model,
                requests,
                args.concurrency,
                template_kwargs,
                "measure",
                repeat,
                start,
            )
        metrics = await client.get("/metrics")
        if metrics.status_code == 200:
            (out / schema.METRICS).write_text(metrics.text)
    return model, records


def _out_dir(args: argparse.Namespace, started: datetime) -> Path:
    if args.out is not None:
        return Path(args.out)
    stamp = started.strftime("%Y%m%d-%H%M%S")
    return (SCRATCH if args.smoke else RESULTS) / f"{stamp}-{args.target}-c{args.concurrency}"


def run(args: argparse.Namespace) -> int:
    workload = schema.load_workload(args.workload)
    requests = workload.requests[: args.requests] if args.requests else workload.requests
    if not args.smoke:
        plan = conditions.Plan(args.repeats, args.warmup, args.cooldown)
        broken = conditions.violations(
            conditions.power(),
            sudo=conditions.can_sudo(),
            containers=conditions.containers(),
            plan=plan,
        )
        if broken:
            print("refusing a full run; B5 is broken:", file=sys.stderr)
            for problem in broken:
                print(f"  - {problem}", file=sys.stderr)
            print("use --smoke for a run that is not B5", file=sys.stderr)
            return 2
    if args.launch and _port_in_use(args.base_url):
        print(f"refusing to launch: something already listens at {args.base_url}", file=sys.stderr)
        return 2
    started = datetime.now(UTC)
    out = _out_dir(args, started)
    out.mkdir(parents=True)
    powermetrics = None if args.smoke else conditions.PowerMetrics(out / schema.POWERMETRICS)
    target = None
    with (out / schema.TARGET_LOG).open("wb") as log:
        try:
            if args.launch:
                target = Target(args.launch, log)
            model, records = asyncio.run(
                _measure(args, requests, workload.chat_template_kwargs, target, out)
            )
        finally:
            if target is not None:
                target.stop()
            if powermetrics is not None:
                powermetrics.stop()
    config = RunConfig(
        target=args.target,
        base_url=args.base_url,
        launch=args.launch,
        served_model=model,
        workload=str(args.workload),
        workload_sha256=hashlib.sha256(args.workload.read_bytes()).hexdigest(),
        requests=len(requests),
        concurrency=args.concurrency,
        repeats=args.repeats,
        warmup=args.warmup,
        cooldown_s=args.cooldown,
        temperature=TEMPERATURE,
        chat_template_kwargs=workload.chat_template_kwargs,
        smoke=args.smoke,
    )
    plist = out / schema.POWERMETRICS
    logged = plist.exists() and plist.stat().st_size > 0
    env = conditions.environment(used_powermetrics=logged, started=started)
    (out / schema.CONFIG).write_text(config.model_dump_json(indent=2) + "\n")
    (out / schema.ENV).write_text(env.model_dump_json(indent=2) + "\n")
    (out / schema.RAW).write_text("".join(r.model_dump_json() + "\n" for r in records))
    result = analysis.load_run(out)
    analysis.write(result)
    print((out / schema.SUMMARY).read_text())
    failed = sum(s.failed + s.prompt_mismatches for s in result.stats)
    if failed:
        print(f"{failed} measured requests failed or rendered another prompt", file=sys.stderr)
    if not args.smoke and not logged:
        print("powermetrics wrote no log: this run does not meet B5", file=sys.stderr)
    return 1 if failed or (not args.smoke and not logged) else 0


def _positive(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"{value} is not at least 1")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.run")
    parser.add_argument("--target", required=True, help="a short name for the server under test")
    parser.add_argument("--launch", help="the command that starts the target; omit if it runs")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--workload", type=Path, default=WORKLOAD)
    parser.add_argument("--requests", type=_positive, help="the first N requests; default all")
    parser.add_argument("--concurrency", type=_positive, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=4, help="requests sent first, not counted")
    parser.add_argument("--cooldown", type=float, default=120.0, help="seconds between repeats")
    parser.add_argument("--ready-timeout", type=float, default=300.0)
    parser.add_argument("--smoke", action="store_true", help="skip the B5 check; not for claims")
    parser.add_argument("--out", type=Path, help="the run directory; default a new one")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
