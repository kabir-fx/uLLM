"""Throwaway: would each twin-gate variant pass the two spikes? Capped rule, margin 0.3, 'each'."""
import importlib.util, sys, statistics as st, random
sys.argv = ["x", "~/phoenix/perplexity/uLLM/bench/.scratch/thermal-spike", "0.3", "0"]
import io, contextlib
spec = importlib.util.spec_from_file_location("rc", "regimes_check.py")
rc = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rc)
LIMIT = {"P": 0.005, "S": 0.03}
for run in rc.RUNS:
    rows, pm = rc.load(run)
    meas = [r for r in rows if r["phase"] != "warmup"]
    blocks = sorted({r["block"] for r in meas})
    chunk = {}
    for b in blocks:
        for e in ("ullm", "mlxA", "mlxB"):
            rs = [r for r in meas if r["block"] == b and r["engine"] == e]
            a, z = min(r["t_send"] for r in rs), max(r["t_end"] for r in rs)
            w = rc.window(pm, a, z)
            chunk[(b, e)] = dict(phase=rs[0]["phase"], share=rc.capped_share(w)[0], any=any(map(rc.is_capped, w)),
                                 toks=sum(r["completion"] for r in rs) / (z - a))
    cls = {}
    for b in blocks:
        if chunk[(b, "mlxA")]["phase"] == "lpm": continue
        anyc = any(chunk[(b, e)]["any"] for e in ("ullm", "mlxA", "mlxB"))
        cls[b] = "P" if not anyc else ("S" if min(chunk[(b, e)]["share"] for e in ("mlxA", "mlxB")) >= 0.5 else "T")
    by = {(r["block"], r["engine"], r["id"]): r for r in meas}
    print(f"=== {run}")
    for reg in "PS":
        bs = [b for b in cls if cls[b] == reg]
        for m in ("ttft", "tpot", "e2e", "tok/s"):
            if m == "tok/s":
                per = {b: [chunk[(b, "mlxA")]["toks"] / chunk[(b, "mlxB")]["toks"]] for b in bs}
            else:
                per = {b: [by[(b, "mlxA", k[2])][m] / by[(b, "mlxB", k[2])][m] for k in by if k[0] == b and k[1] == "mlxA"] for b in bs}
            med, lo, hi, n = rc.boot(per, n=10000)
            contains = lo <= 1 <= hi
            within = abs(med - 1) <= LIMIT[reg]
            print(f"  {reg} {m:5s} median {med:.4f}  CI [{lo:.4f}, {hi:.4f}]  n={n:3d} blocks={len(bs)}  "
                  f"CI has 1: {'yes' if contains else 'NO '}  |med-1|<={LIMIT[reg]}: {'yes' if within else 'NO'}")
