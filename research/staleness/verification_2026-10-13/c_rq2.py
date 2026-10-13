"""POST HOC: RQ2 under ASE's operational oracle.  ASE: violation <=> y' stays essentially identical to
y_base (here: the current/original value); satisfied <=> y' reflects the injected fact OR acknowledges
the conflict; abstention = satisfied.  Real-update analogue: base = old value, mutation = add the new
version; violation <=> answer stays with the OLD value."""
import collections, json, re
import numpy as np
from vcommon import CACHE, P9, jl, page_boot, sample, wilson
_, _, qs = sample()
R12 = jl(CACHE / "p12/records.jsonl"); R11 = jl(CACHE / "p11/records_scale.jsonl")
def rate(rows, f):
    x = [f(r) for r in rows]; k = sum(x); n = len(x)
    return k, n, k / n, page_boot(x, [r["page"] for r in rows])
for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
    cn = [r for r in R12 if r["gen"] == gen]
    real = [r for r in R11 if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"]
    for r in real: r["c"] = P9.commit(qs[r["q"]]["new"], qs[r["q"]]["old"], r["output"])
    abst = lambda o: "unknown" in o.lower()
    print(f"\n{gen}")
    rows = [
      ("p12 as registered: CN injected-only  vs real old-only", lambda r: r["choice"] == "injected", lambda r: r["c"] == "old"),
      ("ASE oracle (strict): CN current-only vs real old-only", lambda r: r["choice"] == "current", lambda r: r["c"] == "old"),
      ("ASE oracle (+ non-abstaining 'neither' as violation)", lambda r: r["choice"] == "current" or (r["choice"] == "neither" and not abst(r["output"])),
                                                              lambda r: r["c"] == "old" or (r["c"] == "neither" and not abst(r["output"]))),
      ("satisfaction (injected/both vs new/both)", lambda r: r["choice"] in ("injected", "both"), lambda r: r["c"] in ("new", "both")),
    ]
    for name, fc, fr in rows:
        k1, n1, m1, ci1 = rate(cn, fc); k2, n2, m2, ci2 = rate(real, fr)
        print(f"  {name:<56} CN {m1:.3f} [{ci1[0]:.3f},{ci1[1]:.3f}]  real {m2:.3f} [{ci2[0]:.3f},{ci2[1]:.3f}]  diff {m1-m2:+.3f}  -> {'|d|>=0.10' if abs(m1-m2) >= 0.10 else '|d|<0.10'}")
    # paired (same retriever, q)
    rc = {(r["retriever"], r["q"]): r for r in cn}; rr = {(r["retriever"], r["q"]): r for r in real}
    ks = sorted(rc.keys() & rr.keys())
    a = np.mean([rc[k]["choice"] == "current" for k in ks]); b = np.mean([rr[k]["c"] == "old" for k in ks])
    print(f"  paired n={len(ks)}: ASE-oracle CN current-only {a:.3f} vs real old-only {b:.3f} (diff {a-b:+.3f})")
