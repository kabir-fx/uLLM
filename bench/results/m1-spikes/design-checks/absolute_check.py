"""Throwaway: mlx-lm absolute TPOT and TTFT p50 per regime and spike, capped rule (margin 0.3, each)."""
import importlib.util, io, contextlib, sys, statistics as st
sys.argv = ["x", "~/phoenix/perplexity/uLLM/bench/.scratch/thermal-spike", "0.3", "0"]
spec = importlib.util.spec_from_file_location("rc", "regimes_check.py")
rc = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(rc)
for run in rc.RUNS:
    rows, pm = rc.load(run)
    meas = [r for r in rows if r["phase"] not in ("warmup", "lpm")]
    blocks = sorted({r["block"] for r in meas})
    cls = {}
    for b in blocks:
        share, anyc = {}, False
        for e in ("ullm", "mlxA", "mlxB"):
            rs = [r for r in meas if r["block"] == b and r["engine"] == e]
            w = rc.window(pm, min(r["t_send"] for r in rs), max(r["t_end"] for r in rs))
            share[e] = rc.capped_share(w)[0]
            anyc |= any(map(rc.is_capped, w))
        cls[b] = "Peak" if not anyc else ("Sustained" if min(share["mlxA"], share["mlxB"]) >= 0.5 else "Transition")
    for reg in ("Peak", "Sustained"):
        for e in ("mlxA", "mlxB", "ullm"):
            rs = [r for r in meas if cls[r["block"]] == reg and r["engine"] == e]
            print(f"{run} {reg:9s} {e:5s} n={len(rs):3d} TPOT p50 {st.median(r['tpot'] for r in rs):5.2f} ms  TTFT p50 {st.median(r['ttft'] for r in rs):6.1f} ms")
