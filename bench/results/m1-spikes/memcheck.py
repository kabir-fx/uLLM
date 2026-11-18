"""How much memory does each server hold on its own? Launch one, send prompts, read footprint."""
import asyncio, os, re, subprocess, sys, time
from pathlib import Path
REPO = Path("~/phoenix/perplexity/uLLM"); sys.path.insert(0, str(REPO)); os.chdir(REPO)
import httpx2
from bench import schema
from bench.loadgen import send
from bench.run import Target, _ready
MODEL = "~/.cache/huggingface/hub/models--mlx-community--Qwen3-0.6B-bf16/snapshots/42096995f6402fde107068cf530136fe64b604f8"
ENG = {
    "ullm": ("uv run --locked ullm serve --port 8000", 8000, None, {}),
    "mlx-lm": (f"uv run --locked python scripts/mlx_lm_server.py --model {MODEL} --prompt-cache-size 0 --port 8001", 8001, MODEL, {"seed": 0}),
}
def sh(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
def footprint(port):
    pid = sh(f"lsof -nP -tiTCP:{port} -sTCP:LISTEN").split()[0]
    out = sh(f"footprint -p {pid} 2>/dev/null")
    m = re.search(r"phys_footprint:\s*([\d.]+ \w+)", out)
    peak = re.search(r"phys_footprint_peak:\s*([\d.]+ \w+)", out)
    big = [l.strip() for l in out.splitlines() if re.match(r"\s*[\d.]+ (GB|MB)", l)][:5]
    rss = sh(f"ps -o rss= -p {pid}").strip()
    return pid, (m.group(1) if m else "?"), (peak.group(1) if peak else "?"), int(rss or 0) // 1024, big
async def main():
    w = schema.load_workload(Path("bench/workloads/a.json"))
    reqs = list(w.requests)[40:76]
    for name, (launch, port, model, extra) in ENG.items():
        log = open(f"/tmp/mem_{name}.log", "wb")
        t = Target(launch, log)
        async with httpx2.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=None) as c:
            mid = await _ready(c, t, 300.0, model)
            print(f"\n{name}: swap at start: {sh('sysctl -n vm.swapusage').strip()}")
            for n, r in enumerate(reqs):
                if n in (0, 6, 12, 24):
                    pid, fp, peak, rss, big = footprint(port)
                    print(f"  after {n:2d} requests: footprint {fp:>10s}  peak {peak:>10s}  rss {rss} MB")
                await send(c, mid, r, w.chat_template_kwargs, extra_body=extra)
            pid, fp, peak, rss, big = footprint(port)
            print(f"  after {len(reqs)} requests: footprint {fp:>10s}  peak {peak:>10s}  rss {rss} MB")
            print("   largest categories:", " | ".join(big))
            print(f"  swap at end: {sh('sysctl -n vm.swapusage').strip()}")
        t.stop()
        time.sleep(2)
asyncio.run(main())
