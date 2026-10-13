"""POST HOC: position effect and cue strata in the DATED variant (explicit 'as of' labels on every passage)."""
import json, collections
import numpy as np
from vcommon import CACHE, P9, jl, sample
_, _, qs = sample()
S = {(c["retriever"], c["q"]): c for c in json.load(open("stress_cases.json")) if c["gen"] == "phi3:mini"}
R = jl(CACHE / "p11/records_scale.jsonl")
for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
    D = [r for r in R if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "dated"]
    for r in D:
        q = qs[r["q"]]; r["lab"] = P9.commit(q["new"], q["old"], r["output"]); s = S[(r["retriever"], r["q"])]
        r["po"], r["pn"], r["cue"] = s["po"], s["pn"], s["cue"]
    c = collections.Counter(r["lab"] for r in D); n = len(D)
    of = [r for r in D if r["po"] < r["pn"]]; nf = [r for r in D if r["pn"] < r["po"]]
    f = lambda rs, v: np.mean([r["lab"] == v for r in rs])
    single = [r for r in D if r["lab"] in ("old", "new")]
    print(f"{gen:<20} DATED n={n}: new {c['new']/n:.2f} old {c['old']/n:.2f} both {c['both']/n:.2f} neither {c['neither']/n:.2f} | "
          f"old-only when old listed first {f(of,'old'):.2f} (n={len(of)}) vs new first {f(nf,'old'):.2f} (n={len(nf)}) | "
          f"single commits choosing first-listed {np.mean([(r['lab']=='old') == (r['po'] < r['pn']) for r in single]):.2f}")
