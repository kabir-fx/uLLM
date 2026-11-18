"""Paired statistics for the spike: A/A noise floor, uLLM against mlx-lm, by thermal segment."""
import json, random, statistics as st, sys
from pathlib import Path
R = Path(sys.argv[1])
rows = [json.loads(l) for l in (R / "raw.jsonl").open()]
CH = json.loads((R / "chunks.json").read_text())
ck = {(c["block"], c["engine"]): c for c in CH}
for r in rows:
    c = r["chunk_ms"]
    r["ttft"] = c[0]
    r["tpot"] = (c[-1] - c[0]) / (r["completion"] - 1)
    r["e2e"] = r["end_ms"]
    r["stalls"] = sum(1 for a, b in zip(c, c[1:]) if b - a > 100)
by = {}
for r in rows:
    if r["phase"] != "warmup":
        by[(r["block"], r["engine"], r["id"])] = r
blocks = sorted({r["block"] for r in rows if r["phase"] != "warmup"})

def seg(b):
    """Thermal segment of a block, from the pressure its mlx chunks saw."""
    ph = ck[(b, "mlxA")]["phase"]
    if ph == "lpm": return "lpm"
    n = min(ck[(b, e)]["pshare"]["N"] for e in ("ullm", "mlxA", "mlxB"))
    if n >= 0.99: return "full clock"
    if max(ck[(b, e)]["pshare"]["N"] for e in ("ullm", "mlxA", "mlxB")) > 0: return "mixed"
    return "hot"
SEG = {b: seg(b) for b in blocks}
print("segments:", {s: [b for b in blocks if SEG[b] == s] for s in ("full clock", "mixed", "hot", "lpm")})

def ratios(a, b, metric, bs, skip_first=0):
    out = {}
    for blk in bs:
        ids = [k[2] for k in by if k[0] == blk and k[1] == a]
        rs = [by[(blk, a, i)][metric] / by[(blk, b, i)][metric] for i in ids
              if by[(blk, a, i)]["i"] >= skip_first and (blk, b, i) in by]
        out[blk] = rs
    return out

def boot(per_block, n=4000, seed=1):
    rng = random.Random(seed)
    keys = list(per_block)
    meds = []
    for _ in range(n):
        pick = [rng.choice(keys) for _ in keys]
        vals = [v for k in pick for v in per_block[k]]
        meds.append(st.median(vals))
    meds.sort()
    allv = [v for k in keys for v in per_block[k]]
    return st.median(allv), meds[int(0.025 * n)], meds[int(0.975 * n)], len(allv)

def show(title, a, b, bs):
    if not bs: return
    print(f"\n{title}  blocks {bs}")
    for metric in ("ttft", "tpot", "e2e"):
        m, lo, hi, n = boot(ratios(a, b, metric, bs))
        print(f"   {metric:5s} median ratio {a}/{b} = {m:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={n} prompts")
    # block throughput ratio
    tr = [ck[(blk, a)]["toks"] / ck[(blk, b)]["toks"] for blk in bs]
    print(f"   tok/s ratio by block: " + " ".join(f"{x:.3f}" for x in tr) + f"   median {st.median(tr):.3f}")

for s in ("full clock", "mixed", "hot", "lpm"):
    bs = [b for b in blocks if SEG[b] == s]
    show(f"A/A  [{s}]", "mlxA", "mlxB", bs)
allhot = [b for b in blocks if SEG[b] in ("mixed", "hot")]
show("A/A  [all non-LPM blocks after the first throttle]", "mlxA", "mlxB", allhot)
show("A/A  [every block]", "mlxA", "mlxB", blocks)
for s in ("full clock", "mixed", "hot", "lpm"):
    bs = [b for b in blocks if SEG[b] == s]
    show(f"uLLM vs mlxA [{s}]", "ullm", "mlxA", bs)
    show(f"uLLM vs mlxB [{s}]", "ullm", "mlxB", bs)

# absolute medians by segment and engine
print("\nABSOLUTE medians by segment (requests pooled)")
print(" segment     engine  n   TTFT   TTFT/ptok  TPOT   tok/s(chunk med)  stalls/1k tok  clock MHz (chunk med)")
for s in ("full clock", "mixed", "hot", "lpm"):
    for e in ("ullm", "mlxA", "mlxB"):
        rs = [r for r in rows if r["phase"] != "warmup" and SEG[r["block"]] == s and r["engine"] == e]
        cs = [ck[(b, e)] for b in blocks if SEG[b] == s]
        if not rs: continue
        toks = sum(r["completion"] for r in rs)
        print(f" {s:11s} {e:6s} {len(rs):3d} {st.median(r['ttft'] for r in rs):6.0f}  {st.median(r['ttft']/r['prompt_tokens'] for r in rs):7.3f}  {st.median(r['tpot'] for r in rs):6.2f}  {st.median(c['toks'] for c in cs):6.1f}   {1000*sum(r['stalls'] for r in rs)/toks:8.1f}   {st.median(c['mhz'] for c in cs):6.0f}")

# uLLM by how long it sat idle before its chunk
print("\nuLLM chunk vs idle time before it (non-LPM)")
last_end = {}
for c in sorted(CH, key=lambda c: c["at"]):
    e = c["engine"]
    idle = c["at"] - last_end.get(e, c["at"])
    if c["phase"] != "lpm" and c["block"] > 0:
        print(f'   block {c["block"]:2d} {e:5s} idle {idle:5.0f} s  TPOT {c["tpot"]:6.2f}  TTFT/ptok {c["ttft_tok"]:.3f}  stalls {c["stalls"]:3d}  cpuW {c["cpu_w"]:.2f}  MHz {c["mhz"]:.0f}') if e == "ullm" else None
    last_end[e] = c["at"] + c["dur"]

# clock-mix distance (TVD) between chunks of the same block
def tvd(p, q):
    return 0.5 * sum(abs(p.get(k, 0) - q.get(k, 0)) for k in set(p) | set(q))
print("\nClock-mix distance (TVD) inside each block")
aa, ab = [], []
for b in blocks:
    s = {e: {int(k): v for k, v in ck[(b, e)]["steps"].items()} for e in ("ullm", "mlxA", "mlxB")}
    d_aa = tvd(s["mlxA"], s["mlxB"]); d_ua = tvd(s["ullm"], s["mlxA"]); d_ub = tvd(s["ullm"], s["mlxB"])
    print(f"   block {b:2d} [{SEG[b]:10s}]  mlxA-mlxB {d_aa:.2f}   ullm-mlxA {d_ua:.2f}   ullm-mlxB {d_ub:.2f}")
