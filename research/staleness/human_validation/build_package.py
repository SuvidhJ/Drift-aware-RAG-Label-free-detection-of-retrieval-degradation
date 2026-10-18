"""Build the human-validation package for the four-way committed-value label (P9.commit).

Pre-registered sampling (2026-10-17): 200 cases from p15 records, stratified by model x source x auto-label
(new/old/both/neither), equal allocation per stratum (capped by availability; shortfall redistributed to
the largest strata), seed 2026.  Annotators see the context the model saw, the statement, the model's
answer, and two candidate values A/B in random order (old/new hidden).  Labels: A, B, BOTH, NEITHER.

Outputs (this folder):
  annotator_1.csv, annotator_2.csv  same 200 cases, independently shuffled order, blank label columns
  KEY_DO_NOT_SHARE.csv              case_id -> which of A/B is the original / mutated value, auto label
  INSTRUCTIONS.md                   one-page guide
"""
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pretests"))
sys.argv = sys.argv[:1] + ["--dry-run"]
import p15_provenance_swap as P15  # noqa: E402

N, SEED = 200, 2026
cases, _ = P15.build()
C = {c["q"]: c for c in cases}
recs = [json.loads(l) for l in open(P15.OUT / "records.jsonl") if l.strip()]
recs = [r for r in recs if r["src"] in ("REAL", "CN")]
strata = defaultdict(list)
for r in recs:
    strata[(r["gen"], r["src"], r["label"])].append(r)
rng = random.Random(SEED)
keys = sorted(strata)
alloc = {k: 0 for k in keys}
left = N
while left > 0:                                   # round-robin equal allocation, capped by availability
    progressed = False
    for k in keys:
        if left and alloc[k] < len(strata[k]):
            alloc[k] += 1; left -= 1; progressed = True
    if not progressed:
        break
sample = []
for k in keys:
    sample += rng.sample(strata[k], alloc[k])

rows, key = [], []
for i, r in enumerate(sample):
    c = C[r["q"]]
    pr = P15.prompts(c)[(r["src"], r["dates"], r["order"])]
    context = pr.split("Context:\n", 1)[1].split("\n\nFill in", 1)[0]
    orig, mut = (c["old"], c["new"]) if r["src"] == "REAL" else (c["new"], c["inj"])
    a_is_orig = rng.random() < 0.5
    A, B = (orig, mut) if a_is_orig else (mut, orig)
    cid = f"HV{i+1:03d}"
    rows.append({"case_id": cid, "statement": c["statement"], "context": context, "model_answer": r["output"],
                 "value_A": ", ".join(A), "value_B": ", ".join(B), "label (A/B/BOTH/NEITHER)": "", "comment": ""})
    auto = {"old": "orig", "new": "mut"}.get(r["label"], r["label"])
    key.append({"case_id": cid, "A_is": "orig" if a_is_orig else "mut", "auto_label": r["label"],
                "auto_as_AB": {"orig": "A" if a_is_orig else "B", "mut": "B" if a_is_orig else "A"}.get(auto, auto.upper()),
                "gen": r["gen"], "src": r["src"], "dates": r["dates"], "order": r["order"], "q": r["q"]})

def write(path, data, order_seed=None):
    data = list(data)
    if order_seed is not None:
        random.Random(order_seed).shuffle(data)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(data[0].keys()))
        w.writeheader(); w.writerows(data)

write(HERE / "annotator_1.csv", rows, order_seed=1)
write(HERE / "annotator_2.csv", rows, order_seed=2)
write(HERE / "KEY_DO_NOT_SHARE.csv", key)
from collections import Counter
print("sampled", len(sample), "cases;", "per model:", dict(Counter(r["gen"] for r in sample)))
print("per source:", dict(Counter(r["src"] for r in sample)), "| per auto label:", dict(Counter(r["label"] for r in sample)))
