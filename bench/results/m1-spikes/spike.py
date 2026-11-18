"""Thermal spike driver: one cold-to-hot trajectory with three servers taking turns.

    uv run --locked python bench/.scratch/thermal-spike/spike.py --out <dir> [--smoke]

Engines: ullm, and two identical mlx_lm.server processes (mlxA, mlxB) for an A/A comparison.
Every block gives each engine the same K prompts, in a rotating engine order. Phases:

  cold_hot  blocks back to back from a cool machine until it is heat-soaked
  reload1   rest, then one block: how much full clock does a rest buy when soaked
  lpm       Low Power Mode on (needs lpm_helper.sh), blocks back to back, then off
  reload2/3 rest, then one block, twice

Logs: raw.jsonl (one line a request), events.jsonl, sensors.jsonl, powermetrics.plist.
"""

import argparse
import asyncio
import json
import os
import random
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path("~/phoenix/perplexity/uLLM")
HERE = Path(__file__).resolve().parent
CTL = HERE / "ctl"
sys.path.insert(0, str(REPO))
os.chdir(REPO)

import httpx2  # noqa: E402
from bench import conditions, schema  # noqa: E402
from bench.loadgen import send  # noqa: E402
from bench.run import Target, _port_in_use, _ready  # noqa: E402

MODEL = (
    "~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-bf16/"
    "snapshots/42096995f6402fde107068cf530136fe64b604f8"
)
UV = "uv run --locked"
MLX = f"{UV} python scripts/mlx_lm_server.py --model {MODEL} --prompt-cache-size 0 --port"
ENGINES = {
    "ullm": {"port": 8000, "launch": f"{UV} ullm serve --port 8000", "model": None, "extra": {}},
    "mlxA": {"port": 8001, "launch": f"{MLX} 8001", "model": MODEL, "extra": {"seed": 0}},
    "mlxB": {"port": 8002, "launch": f"{MLX} 8002", "model": MODEL, "extra": {"seed": 0}},
}
ORDERS = (
    ("ullm", "mlxA", "mlxB"),
    ("mlxA", "mlxB", "ullm"),
    ("mlxB", "ullm", "mlxA"),
    ("ullm", "mlxB", "mlxA"),
    ("mlxB", "mlxA", "ullm"),
    ("mlxA", "ullm", "mlxB"),
)
# Offsets in seconds from driver start. The full plan fits in 44.5 minutes.
FULL = {
    "k": 12,
    "cold_hot_end": 17.0 * 60,
    "rest1": 180,
    "hot2_end": 29.5 * 60,
    "deadline": 44.25 * 60,
    "block_est": 125.0,
    "samples": 2760,
}
SMOKE = {
    "k": 2,
    "cold_hot_end": 80,
    "rest1": 5,
    "hot2_end": 115,
    "deadline": 150,
    "block_est": 15.0,
    "samples": 200,
}


def make_blocks(requests: list, k: int, seed: int) -> list[list]:
    """Blocks of k prompts with one prompt from each of k prompt-length strata."""
    rng = random.Random(seed)
    by_len = sorted(requests, key=lambda r: r.prompt_tokens)
    n = len(by_len) // k
    blocks: list[list] = [[] for _ in range(n)]
    for s in range(k):
        stratum = by_len[s * n : (s + 1) * n]
        rng.shuffle(stratum)
        for b, request in enumerate(stratum):
            blocks[b].append(request)
    for block in blocks:
        rng.shuffle(block)
    return blocks


def sh(*argv: str) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=15).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def lpm_state() -> bool | None:
    m = re.search(r"^\s*lowpowermode\s+(\d)", sh("pmset", "-g"), re.MULTILINE)
    return None if m is None else m.group(1) == "1"


def helper_alive() -> bool:
    try:
        return time.time() - (CTL / "helper_alive").stat().st_mtime < 6
    except OSError:
        return False


class Spike:
    def __init__(self, out: Path, plan: dict) -> None:
        self.out, self.plan = out, plan
        self.t0 = time.monotonic()
        self.raw = (out / "raw.jsonl").open("w")
        self.events = (out / "events.jsonl").open("w")
        self.targets: dict[str, Target] = {}
        self.clients: dict[str, httpx2.AsyncClient] = {}
        self.models: dict[str, str] = {}
        self.block = 0
        self.block_est = plan["block_est"]
        self.lpm_on = False
        workload = schema.load_workload(Path("bench/workloads/a.json"))
        self.template_kwargs = workload.chat_template_kwargs
        self.warmups = list(workload.requests[:4])
        self.blocks = make_blocks(list(workload.requests), plan["k"], seed=0)

    def now(self) -> float:
        return time.monotonic() - self.t0

    def event(self, kind: str, /, **detail: object) -> None:
        row = {"t": round(time.time(), 3), "at_s": round(self.now(), 1), "event": kind, **detail}
        self.events.write(json.dumps(row) + "\n")
        self.events.flush()
        print(f"[{self.now():7.1f}s] {kind} {json.dumps(detail) if detail else ''}", flush=True)

    def power_note(self, label: str) -> None:
        batt = " ".join(sh("pmset", "-g", "batt").split())
        level = sh("notifyutil", "-g", "com.apple.system.thermalpressurelevel").strip()
        swap = sh("sysctl", "-n", "vm.swapusage").strip()
        self.event("state", label=label, batt=batt, lpm=lpm_state(), thermal=level, swap=swap)

    async def start_engines(self) -> None:
        for name, spec in ENGINES.items():
            url = f"http://127.0.0.1:{spec['port']}"
            if _port_in_use(url):
                raise SystemExit(f"something already listens at {url}")
            log = (self.out / f"{name}.log").open("wb")
            self.targets[name] = Target(spec["launch"], log)
            self.clients[name] = httpx2.AsyncClient(base_url=url, timeout=None)
        for name, spec in ENGINES.items():
            self.models[name] = await _ready(self.clients[name], self.targets[name], 300.0, spec["model"])
            self.event("ready", engine=name)

    async def one(self, engine: str, request, row: dict) -> dict:
        row = {**row, "engine": engine, "id": request.id, "prompt_tokens": request.prompt_tokens,
               "max_tokens": request.max_tokens, "t_send": round(time.time(), 4)}
        try:
            reply, end_ms = await asyncio.wait_for(
                send(self.clients[engine], self.models[engine], request, self.template_kwargs,
                     extra_body=ENGINES[engine]["extra"]),
                timeout=180,
            )
            usage = reply.usage
            row |= {
                "chunk_ms": [round(x, 2) for x in reply.chunk_ms],
                "end_ms": round(end_ms, 2),
                "usage_prompt": usage.prompt_tokens if usage else None,
                "completion": usage.completion_tokens if usage else None,
                "finish": reply.finish_reason,
                "error": reply.outcome_error(),
            }
        except TimeoutError:
            row |= {"chunk_ms": [], "end_ms": None, "completion": None, "error": "timeout 180 s"}
        self.raw.write(json.dumps(row) + "\n")
        self.raw.flush()
        return row

    async def warm_up(self) -> None:
        for name in ENGINES:
            for i, request in enumerate(self.warmups):
                await self.one(name, request, {"phase": "warmup", "block": -1, "pos": 0, "i": i})
            self.event("warmed", engine=name)

    async def run_block(self, phase: str) -> None:
        started = time.monotonic()
        prompts = self.blocks[self.block % len(self.blocks)]
        order = ORDERS[self.block % len(ORDERS)]
        for pos, engine in enumerate(order):
            if self.targets[engine].proc.poll() is not None:
                self.event("engine_dead", engine=engine)
                continue
            rows = [
                await self.one(engine, request, {"phase": phase, "block": self.block, "pos": pos, "i": i})
                for i, request in enumerate(prompts)
            ]
            ok = [r for r in rows if r["error"] is None and len(r["chunk_ms"]) > 1]
            ttft = sorted(r["chunk_ms"][0] for r in ok)
            tpot = sorted((r["chunk_ms"][-1] - r["chunk_ms"][0]) / (r["completion"] - 1) for r in ok)
            mid = len(ok) // 2
            self.event(
                "chunk", phase=phase, block=self.block, engine=engine, pos=pos, ok=len(ok),
                failed=len(rows) - len(ok),
                ttft_p50=round(ttft[mid], 1) if ok else None,
                tpot_p50=round(tpot[mid], 2) if ok else None,
            )
        self.block += 1
        self.block_est = time.monotonic() - started

    async def blocks_until(self, phase: str, end_s: float) -> None:
        while self.now() + self.block_est <= end_s + 15:
            await self.run_block(phase)

    async def rest(self, seconds: float, label: str) -> None:
        self.event("rest_start", label=label, seconds=round(seconds))
        await asyncio.sleep(seconds)
        self.power_note(f"after {label}")

    async def set_lpm(self, on: bool) -> bool:
        if not helper_alive():
            self.event("lpm_helper_missing", wanted=on)
            return False
        (CTL / ("lpm_on" if on else "lpm_off")).write_text("")
        for _ in range(12):
            await asyncio.sleep(1)
            if lpm_state() == on:
                self.lpm_on = on
                self.event("lpm", on=on)
                return True
        self.event("lpm_failed", wanted=on)
        return False

    async def run(self) -> None:
        p = self.plan
        self.power_note("start")
        await self.start_engines()
        await self.warm_up()
        self.power_note("after warm-up")

        self.event("phase", name="cold_hot")
        await self.blocks_until("cold_hot", p["cold_hot_end"])
        self.power_note("after cold_hot")

        await self.rest(p["rest1"], "rest1")
        await self.run_block("reload1")

        self.event("phase", name="hot2")
        await self.blocks_until("hot2", p["hot2_end"])
        self.power_note("after hot2")

        if await self.set_lpm(True):
            self.event("phase", name="lpm")
            self.block_est *= 1.3
            await self.blocks_until("lpm", p["deadline"])
            await self.set_lpm(False)
        else:
            # No helper: keep the hot rotation going instead.
            self.event("phase", name="hot3")
            await self.blocks_until("hot3", p["deadline"])
        self.power_note("end")

    async def close(self) -> None:
        if self.lpm_on:
            (CTL / "lpm_off").write_text("")
        for client in self.clients.values():
            await client.aclose()
        for target in self.targets.values():
            target.stop()
        self.raw.close()
        self.events.close()


async def amain(out: Path, plan: dict) -> None:
    spike = Spike(out, plan)
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, task.cancel)
    try:
        await spike.run()
    finally:
        await spike.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--settle", type=int, default=0, help="idle seconds logged before the start")
    args = parser.parse_args()
    args.out.mkdir(parents=True)
    CTL.mkdir(exist_ok=True)
    plan = SMOKE if args.smoke else FULL
    meta = {
        "plan": plan, "engines": ENGINES, "orders": ORDERS, "smoke": args.smoke,
        "started_unix": time.time(), "settle_s": args.settle, "commit": sh("git", "rev-parse", "HEAD").strip(),
        "can_sudo_powermetrics": conditions.can_sudo(),
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    sensors = subprocess.Popen([sys.executable, str(HERE / "sensors.py"), str(args.out / "sensors.jsonl")])
    powermetrics = subprocess.Popen(
        ["sudo", "-n", conditions.POWERMETRICS, "--samplers", "gpu_power,cpu_power,thermal",
         "--sample-rate", "1000", "--sample-count", str(plan["samples"] + args.settle), "--format", "plist",
         "--output-file", str(args.out / "powermetrics.plist")],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True,
    )
    try:
        if args.settle:
            print(f"settling for {args.settle} s with the loggers running", flush=True)
            time.sleep(args.settle)
        asyncio.run(amain(args.out, plan))
    except asyncio.CancelledError:
        print("cancelled", flush=True)
    finally:
        if powermetrics.poll() is None:
            os.kill(powermetrics.pid, signal.SIGTERM)
        try:
            powermetrics.wait(timeout=15)
        except subprocess.TimeoutExpired:
            print("powermetrics still running; it ends at its sample count", flush=True)
        sensors.send_signal(signal.SIGINT)
        sensors.wait(timeout=10)
    print("spike finished", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
