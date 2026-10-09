"""Headline result: what a fully stale index costs, for every (model, corpus).

    python -m research.staleness.run_headline [--models minilm bge] [--corpora ...]

Embeddings are cached under embeddings/cache/staleness/<model>/, so a second run
only re-scores.
"""
import argparse, json, time

from . import measure as M
from .data import REPO

ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["minilm", "bge"])
ap.add_argument("--corpora", nargs="+", default=["arxiv_big", "python_docs"])
a = ap.parse_args()
out = {}
for model in a.models:
    for corpus in a.corpora:
        t0 = time.time()
        res = M.per_pool(model, corpus)
        row = {"weights": res["weights"]}
        for mt in ("hit@1", "hit@10", "mrr"):
            row[mt] = M.net_cost(res, mt)
        out[f"{model}/{corpus}"] = row
        w = res["weights"]
        print(f"\n{model} / {corpus}   ({time.time()-t0:.0f}s)   weights new {w['new']:.3f} "
              f"retained {w['retained']:.3f} unchanged {w['unchanged']:.3f}", flush=True)
        for mt in ("hit@1", "hit@10", "mrr"):
            r = row[mt]; p = r["pools"]
            print(f"  {mt:<7} NET {r['net']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}]"
                  f"   new {p['new'][0]:+.4f}  retained {p['retained'][0]:+.4f}"
                  f"  unchanged {p['unchanged'][0]:+.4f}", flush=True)
path = REPO / "research" / "staleness" / "results_headline.json"
path.write_text(json.dumps(out, indent=1))
print(f"\nwrote {path}")
