"""Join the spike's requests to the powermetrics samples and print per-chunk and timeline tables.

    python analyze.py <run dir> [section ...]     sections: chunks timeline sensors
"""
import json, statistics as st, sys
from pathlib import Path

R = Path(sys.argv[1])
SECTIONS = set(sys.argv[2:]) or {"chunks", "timeline", "sensors"}
pm = json.loads((R / "pm.json").read_text())
rows = [json.loads(l) for l in (R / "raw.jsonl").open()]
events = [json.loads(l) for l in (R / "events.jsonl").open()]
T0 = events[0]["t"] - events[0]["at_s"]          # unix time of driver start
LVL = {"Nominal": "N", "Moderate": "M", "Heavy": "H", "Trapping": "T", "Sleeping": "S"}

def med(xs): return st.median(xs) if xs else float("nan")

def window(a, b):
    """Samples whose interval [t - el, t] overlaps [a, b]."""
    return [s for s in pm if s["t"] > a and s["t"] - s["el"] < b]

def clock(samples):
    """Active-time-weighted clock summary of a list of samples."""
    act = sum((1 - s["idle"]) * s["el"] for s in samples)
    if act <= 0: return None
    steps = {}
    for s in samples:
        for f, share in s["steps"].items():
            steps[int(f)] = steps.get(int(f), 0) + share * s["el"]
    tot = sum(steps.values()) or 1
    steps = {f: v / tot for f, v in steps.items()}
    return {
        "mhz": sum(f * v for f, v in steps.items()),
        "top": steps.get(1470, 0.0),
        "steps": steps,
        "busy": act / sum(s["el"] for s in samples),
        "gpu_w": st.mean(s["gpu_w"] for s in samples),
        "cpu_w": st.mean(s["cpu_w"] for s in samples),
        "press": "".join(sorted({LVL[s["pressure"]] for s in samples}, key="NMHTS".index)),
        "pshare": {k: sum(1 for s in samples if LVL[s["pressure"]] == k) / len(samples) for k in "NMH"},
        "req": med([s["req"] for s in samples if s.get("req") is not None]),
        "app": med([s["app"] for s in samples if s.get("app") is not None]),
        "p_mhz": st.mean(s["p_mhz"] for s in samples),
    }

for r in rows:
    c = r["chunk_ms"]
    r["ttft"] = c[0] if c else None
    r["tpot"] = (c[-1] - c[0]) / (r["completion"] - 1) if len(c) > 1 and r["completion"] and r["completion"] > 1 else None
    gaps = [b - a for a, b in zip(c, c[1:])]
    r["maxgap"] = max(gaps) if gaps else None
    r["stalls"] = sum(1 for g in gaps if g > 100)
    r["t_end"] = r["t_send"] + (r["end_ms"] or 0) / 1000

chunks = {}
for r in rows:
    if r["phase"] == "warmup": continue
    chunks.setdefault((r["block"], r["pos"]), []).append(r)

def chunk_row(key):
    rs = chunks[key]
    a, b = rs[0]["t_send"], rs[-1]["t_end"]
    ck = clock(window(a, b))
    return {
        "block": key[0], "pos": key[1], "engine": rs[0]["engine"], "phase": rs[0]["phase"],
        "at": a - T0, "dur": b - a, "n": len(rs),
        "ttft": med([r["ttft"] for r in rs]), "tpot": med([r["tpot"] for r in rs]),
        "ttft_tok": med([r["ttft"] / r["prompt_tokens"] for r in rs]),
        "toks": sum(r["completion"] for r in rs) / (b - a),
        "stalls": sum(r["stalls"] for r in rs), "maxgap": max(r["maxgap"] for r in rs),
        **{k: ck[k] for k in ("mhz", "top", "press", "pshare", "gpu_w", "cpu_w", "req", "app", "busy", "steps", "p_mhz")},
    }

CH = [chunk_row(k) for k in sorted(chunks)]
(R / "chunks.json").write_text(json.dumps(CH))

if "chunks" in SECTIONS:
    print("   at  phase    blk eng  pos  dur  tok/s  TTFT  TTFT/ptok TPOT  stalls maxgap |  MHz  top  busy press  N/M/H share   gpuW cpuW  req  app  P-MHz")
    for c in CH:
        ps = c["pshare"]
        print(f'{c["at"]:5.0f} {c["phase"]:8s} {c["block"]:3d} {c["engine"]:5s} {c["pos"]}  {c["dur"]:4.0f} {c["toks"]:6.1f} {c["ttft"]:6.0f}  {c["ttft_tok"]:6.3f} {c["tpot"]:6.2f}  {c["stalls"]:4d} {c["maxgap"]:6.0f} | {c["mhz"]:5.0f} {c["top"]:.2f} {c["busy"]:.2f} {c["press"]:4s} {ps["N"]:.2f}/{ps["M"]:.2f}/{ps["H"]:.2f}  {c["gpu_w"]:5.2f} {c["cpu_w"]:4.2f} {c["req"]:4.1f} {c["app"]:4.1f} {c["p_mhz"]:5.0f}')

if "timeline" in SECTIONS:
    print("\nTIMELINE, 1-minute bins from driver start (settle shown as negative minutes)")
    print(" min  press(N/M/H)   busy  MHz(active) top   gpuW  cpuW")
    t = T0 - 240
    while t < pm[-1]["t"]:
        w = window(t, t + 60)
        if w:
            ck = clock(w)
            ps = {k: sum(1 for s in w if LVL[s["pressure"]] == k) / len(w) for k in "NMH"}
            mhz = f'{ck["mhz"]:6.0f}  {ck["top"]:.2f}' if ck and ck["busy"] > 0.05 else "   idle      "
            print(f'{(t - T0) / 60:4.0f}  {ps["N"]:.2f}/{ps["M"]:.2f}/{ps["H"]:.2f}  {1 - st.mean(s["idle"] for s in w):.2f}  {mhz}  {st.mean(s["gpu_w"] for s in w):5.2f} {st.mean(s["cpu_w"] for s in w):5.2f}')
        t += 60

if "sensors" in SECTIONS:
    lines = (R / "sensors.jsonl").open()
    names = json.loads(next(lines))["names"]
    S = [json.loads(l) for l in lines]
    groups = {
        "die max": [i for i, n in enumerate(names) if "tdie" in n],
        "die med": [i for i, n in enumerate(names) if "tdie" in n],
        "tdev": [i for i, n in enumerate(names) if "tdev" in n],
        "NAND": [i for i, n in enumerate(names) if "NAND" in n],
        "battery": [i for i, n in enumerate(names) if "gas gauge" in n],
    }
    def val(s, g):
        v = [s["v"][i] for i in groups[g] if s["v"][i] is not None and s["v"][i] > 0]
        if not v: return float("nan")
        return max(v) if g in ("die max",) else st.median(v)
    print("\nSENSORS, 1-minute means (deg C)")
    print(" min   die max  die med   tdev   NAND  battery")
    t = S[0]["t"]
    while t < S[-1]["t"]:
        w = [s for s in S if t <= s["t"] < t + 60]
        if w:
            print(f'{(t - T0) / 60:4.0f}   ' + "  ".join(f'{st.mean(val(s, g) for s in w):6.1f}' for g in groups))
        t += 60
