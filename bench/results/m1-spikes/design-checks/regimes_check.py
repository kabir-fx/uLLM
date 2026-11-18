"""Throwaway: apply the Final Approach's capped rule to both spikes and test 0016's choices.

python3 regimes_check.py <spike dir> [margin] [busy_min]
"""
import json
import random
import statistics as st
import sys
from pathlib import Path

HERE = Path(sys.argv[1])
MARGIN = float(sys.argv[2]) if len(sys.argv) > 2 else 0.3
BUSY_MIN = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0

RUNS = ("run-20261118-141243", "run-20261118-163540")


def load(run):
    rows = [json.loads(l) for l in open(HERE / run / "raw.jsonl")]
    pm = json.load(open(HERE / run / "pm.json"))
    for r in rows:
        c = r["chunk_ms"]
        r["ttft"] = c[0]
        r["tpot"] = (c[-1] - c[0]) / (r["completion"] - 1)
        r["e2e"] = r["end_ms"]
        r["t_end"] = r["t_send"] + r["end_ms"] / 1000
    return rows, pm


def window(pm, a, b):
    return [s for s in pm if s["t"] > a and s["t"] - s["el"] < b]


def is_capped(s):
    return s["req"] is not None and s["app"] is not None and (1 - s["idle"]) >= BUSY_MIN and s["app"] < s["req"] - MARGIN


def capped_share(samples):
    busy = sum((1 - s["idle"]) * s["el"] for s in samples)
    cap = sum((1 - s["idle"]) * s["el"] for s in samples if is_capped(s))
    return cap / busy if busy else 0.0, cap, busy


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


for run in RUNS:
    rows, pm = load(run)
    meas = [r for r in rows if r["phase"] != "warmup"]
    blocks = sorted({r["block"] for r in meas})
    chunk = {}
    for b in blocks:
        for e in ("ullm", "mlxA", "mlxB"):
            rs = [r for r in meas if r["block"] == b and r["engine"] == e]
            a, z = min(r["t_send"] for r in rs), max(r["t_end"] for r in rs)
            w = window(pm, a, z)
            share, cap, busy = capped_share(w)
            chunk[(b, e)] = dict(phase=rs[0]["phase"], share=share, any=any(is_capped(s) for s in w),
                                 n_capped=sum(is_capped(s) for s in w), pos=rs[0]["pos"],
                                 toks=sum(r["completion"] for r in rs) / (z - a), a=a, z=z)
    cls_each, cls_pool = {}, {}
    for b in blocks:
        anyc = any(chunk[(b, e)]["any"] for e in ("ullm", "mlxA", "mlxB"))
        each = min(chunk[(b, e)]["share"] for e in ("mlxA", "mlxB"))
        # pooled: busy-weighted over both baseline chunks
        wa = window(pm, chunk[(b, "mlxA")]["a"], chunk[(b, "mlxA")]["z"])
        wb = window(pm, chunk[(b, "mlxB")]["a"], chunk[(b, "mlxB")]["z"])
        pooled = capped_share(wa + wb)[0]
        cls_each[b] = "P" if not anyc else ("S" if each >= 0.5 else "T")
        cls_pool[b] = "P" if not anyc else ("S" if pooled >= 0.5 else "T")
    print(f"\n=== {run}  margin {MARGIN} busy_min {BUSY_MIN}")
    print(" blk phase     ullm  mlxA  mlxB  (capped share; * = any capped sample)  each pooled")
    for b in blocks:
        cells = "  ".join(f'{chunk[(b,e)]["share"]:.2f}{"*" if chunk[(b,e)]["any"] else " "}' for e in ("ullm", "mlxA", "mlxB"))
        print(f" {b:3d} {chunk[(b,'mlxA')]['phase']:8s} {cells}   {cls_each[b]}    {cls_pool[b]}")
    non_lpm = [b for b in blocks if chunk[(b, "mlxA")]["phase"] != "lpm"]
    for name, cls in (("each", cls_each), ("pooled", cls_pool)):
        counts = {k: sum(cls[b] == k for b in non_lpm) for k in "PTS"}
        print(f" non-LPM counts ({name}): {counts}")
    by = {(r["block"], r["engine"], r["id"]): r for r in meas}
    for regime in ("P", "S"):
        bs = [b for b in non_lpm if cls_each[b] == regime]
        if not bs:
            continue
        print(f" regime {regime} blocks {bs}")
        for a_, b_ in (("mlxA", "mlxB"), ("ullm", "mlxA"), ("ullm", "mlxB")):
            out = []
            for m in ("ttft", "tpot", "e2e"):
                per = {blk: [by[(blk, a_, k[2])][m] / by[(blk, b_, k[2])][m] for k in by if k[0] == blk and k[1] == a_] for blk in bs}
                med, lo, hi, n = boot(per)
                out.append(f"{m} {med:.3f} [{lo:.3f},{hi:.3f}]")
            tr = st.median(chunk[(blk, a_)]["toks"] / chunk[(blk, b_)]["toks"] for blk in bs)
            print(f"   {a_}/{b_}: " + "  ".join(out) + f"  tok/s {tr:.3f}  n={n}")
    lpm = [b for b in blocks if chunk[(b, "mlxA")]["phase"] == "lpm"]
    print(" LPM blocks capped share mlxA:", [round(chunk[(b, 'mlxA')]['share'], 2) for b in lpm])
    # stalls in this run (uLLM)
    for e in ("ullm", "mlxA"):
        rs = [r for r in meas if r["engine"] == e and r["phase"] != "lpm"]
        gaps = [y - x for r in rs for x, y in zip(r["chunk_ms"], r["chunk_ms"][1:])]
        stream = sum(r["chunk_ms"][-1] - r["chunk_ms"][0] for r in rs)
        stall = sum(g for g in gaps if g > 100)
        toks = sum(r["completion"] for r in rs)
        print(f" {e}: stalls>100ms {sum(g > 100 for g in gaps)} ({1000*sum(g>100 for g in gaps)/toks:.1f}/1k tok), max gap {max(gaps):.0f} ms, stall share {stall/stream:.3%}")
