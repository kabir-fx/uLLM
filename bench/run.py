"""One benchmark run, end to end, into a C5 directory (B1, B5, decision 0013).

    uv run python -m bench.run --target ullm --launch "uv run ullm serve"
    uv run python -m bench.run --target ullm --launch "uv run ullm serve" --smoke --requests 16
    uv run python -m bench.run --target mlx-lm --launch "uv run mlx_lm.server --model <dir>" \
        --model <dir> --extra-body '{"seed": 0}'

A full run refuses to start unless B5 holds. ``--smoke`` skips that check and says so in
config.json and summary.md.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import httpx2
from bench import analysis, conditions, schema
from bench.loadgen import COMMON_FIELDS, TEMPERATURE, closed_loop
from bench.schema import RequestRecord, RunConfig, WorkloadRequest
from bench.target import Target, port_in_use, ready

WORKLOAD = Path("bench/workloads/a.json")
RESULTS = Path("bench/results")
SCRATCH = Path("bench/.scratch")


async def _measure(
    args: argparse.Namespace,
    requests: Sequence[WorkloadRequest],
    template_kwargs: dict[str, bool],
    target: Target | None,
    out: Path,
) -> tuple[str, list[RequestRecord]]:
    async with httpx2.AsyncClient(base_url=args.base_url, timeout=None) as client:
        model = await ready(client, target, args.ready_timeout, args.model)
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
            extra_body=args.extra_body,
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
                extra_body=args.extra_body,
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


def _gzip_powermetrics(out: Path) -> bool:
    """Replace the powermetrics log with its gzip: about 32 KB a sample raw, 2 KB gzipped
    (decision 0014). True when the log held anything."""
    plist = out / schema.POWERMETRICS
    if not plist.exists():
        return False
    data = plist.read_bytes()
    if data:
        (out / schema.POWERMETRICS_GZ).write_bytes(gzip.compress(data, mtime=0))
    plist.unlink()
    return bool(data)


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
    if args.launch and port_in_use(args.base_url):
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
        extra_body=args.extra_body,
        smoke=args.smoke,
    )
    logged = _gzip_powermetrics(out)
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


def _extra_body(text: str) -> dict[str, object]:
    try:
        body = json.loads(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"not JSON: {e}") from e
    if not isinstance(body, dict):
        raise argparse.ArgumentTypeError("not a JSON object")
    if clash := sorted(body.keys() & set(COMMON_FIELDS)):
        raise argparse.ArgumentTypeError(f"replaces fields every target gets: {clash}")
    return body


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.run")
    parser.add_argument("--target", required=True, help="a short name for the server under test")
    parser.add_argument("--launch", help="the command that starts the target; omit if it runs")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--workload", type=Path, default=WORKLOAD)
    parser.add_argument(
        "--model", help="the model id to send; default the first that GET /v1/models lists"
    )
    parser.add_argument(
        "--extra-body",
        type=_extra_body,
        default={},
        help="a JSON object of request fields for this target only, recorded in config.json",
    )
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
