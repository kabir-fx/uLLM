"""Throwaway: do all servers of a concurrency-8 group fit in memory while every one stays loaded?

Run from the repo root: uv run --locked python <this file>
Starts the five candidate c8 servers, then gives each in turn the 24 longest workload A prompts at
concurrency 8, and reads every server's footprint and peak after each turn, and the swap in use.
"""

import asyncio
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
import httpx2
from bench import schema
from bench.loadgen import closed_loop
from bench.run import Target, _ready

SCRATCH = Path(__file__).resolve().parent
MODEL = subprocess.run(
    ["uv", "run", "--locked", "python", "scripts/fetch_models.py"], capture_output=True, text=True
).stdout
MODEL = re.search(r"qwen3-0.6b-bf16: (\S+)", MODEL).group(1)
MLX = f"uv run --locked python scripts/mlx_lm_server.py --model {MODEL} --port"
SEQ = {"seed": 0}
SERVERS = {
    "ullm": (8100, "uv run --locked ullm serve --port 8100", None, {}),
    "mlx-lm-batched": (8101, f"{MLX} 8101", MODEL, {}),
    "mlx-lm-batched-no-cache": (8102, f"{MLX} 8102 --prompt-cache-size 0", MODEL, {}),
    "mlx-lm-batched-no-cache-twin": (8103, f"{MLX} 8103 --prompt-cache-size 0", MODEL, {}),
    "mlx-lm-sequential-no-cache": (8104, f"{MLX} 8104 --prompt-cache-size 0", MODEL, SEQ),
}


def sh(*argv: str) -> str:
    return subprocess.run(argv, capture_output=True, text=True).stdout


def footprint_mb(pgid: int) -> tuple[float, float]:
    """phys_footprint and its peak, summed over the process group, in MB."""
    pids = sh("pgrep", "-g", str(pgid)).split()
    now = peak = 0.0
    for pid in pids:
        out = sh("footprint", "-p", pid)
        for key, acc in (("phys_footprint", "now"), ("phys_footprint_peak", "peak")):
            m = re.search(rf"{key}:\s*([\d.]+) (KB|MB|GB)", out)
            if m:
                mb = float(m.group(1)) * {"KB": 1 / 1024, "MB": 1, "GB": 1024}[m.group(2)]
                if acc == "now":
                    now += mb
                else:
                    peak += mb
    return now, peak


def swap() -> str:
    return sh("sysctl", "-n", "vm.swapusage").strip()


def report(targets: dict[str, Target], label: str) -> None:
    total = 0.0
    print(f"--- {label}   swap: {swap()}")
    for name, t in targets.items():
        now, peak = footprint_mb(t.proc.pid)
        total += now
        print(f"   {name:30s} footprint {now:7.0f} MB   peak {peak:7.0f} MB")
    print(f"   {'sum':30s} footprint {total:7.0f} MB")


async def main() -> None:
    w = schema.load_workload(Path("bench/workloads/a.json"))
    longest = sorted(w.requests, key=lambda r: r.prompt_tokens + r.max_tokens)[-24:]
    targets: dict[str, Target] = {}
    clients: dict[str, httpx2.AsyncClient] = {}
    models: dict[str, str] = {}
    print("swap before:", swap())
    try:
        for name, (port, launch, _, _) in SERVERS.items():
            targets[name] = Target(launch, (SCRATCH / f"probe_{name}.log").open("wb"))
            clients[name] = httpx2.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=None)
        for name, (_, _, model, _) in SERVERS.items():
            models[name] = await _ready(clients[name], targets[name], 300.0, model)
        report(targets, "all loaded, idle")
        for name, (_, _, _, extra) in SERVERS.items():
            records = await closed_loop(
                clients[name], models[name], longest, 8, w.chat_template_kwargs, "measure", 0, 0.0,
                extra_body=extra,
            )
            failed = sum(r.error is not None for r in records)
            report(targets, f"after {name}'s 24 longest prompts at c8 ({failed} failed)")
    finally:
        for c in clients.values():
            await c.aclose()
        for t in targets.values():
            t.stop()
    print("swap after stop:", swap())


asyncio.run(main())
