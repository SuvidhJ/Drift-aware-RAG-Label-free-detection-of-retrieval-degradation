"""Order effect WITHOUT ASE's parser: share of answers committing to the ORIGINAL value only (four-way
P9.commit), by whether the mutation passage came before or after the original in the retrieved context.
Observational (retrieval order, not randomized)."""
import json, re, sys
import numpy as np
exec(open("g_aseoracle.py").read().split("R12 = jl(")[0])          # reuse the context/order reconstruction
from vcommon import mcnemar_exact
R12 = jl(CACHE / "p12/records.jsonl"); R11 = jl(CACHE / "p11/records_scale.jsonl")
S = {(c["retriever"], c["q"]): c for c in json.load(open("stress_cases.json")) if c["gen"] == "phi3:mini"}
for gen in ("phi3:mini", "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"):
    for src in ("CN", "REAL"):
        rows = []
        if src == "CN":
            for r in R12:
                if r["gen"] == gen:
                    lab = {"current": "orig", "injected": "mut"}.get(r["choice"], r["choice"])
                    rows.append((lab, order_cn[(r["retriever"], r["q"])]))
        else:
            for r in R11:
                if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none":
                    q = qs[r["q"]]; s = S[(r["retriever"], r["q"])]
                    lab = {"old": "orig", "new": "mut"}.get(P9.commit(q["new"], q["old"], r["output"]), P9.commit(q["new"], q["old"], r["output"]))
                    rows.append((lab, "injected_after" if s["pn"] > s["po"] else "injected_before"))
        if not rows: continue
        out = []
        for o in ("injected_before", "injected_after"):
            ro = [l for l, oo in rows if oo == o]
            out.append(f"{o.split('_')[1]:<6} n={len(ro):<3} orig-only {np.mean([l=='orig' for l in ro]):.3f} mut-only {np.mean([l=='mut' for l in ro]):.3f} both {np.mean([l=='both' for l in ro]):.3f}")
        print(f"{gen[:12]:<12} {src:<4} | " + " | ".join(out))
