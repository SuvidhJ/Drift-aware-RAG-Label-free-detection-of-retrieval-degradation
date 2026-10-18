"""POST HOC: refusal counts per cell with the v2 pattern, plus precision/recall reading samples (seed 5)."""
import random
from refusal_common import REF2, DATASETS, cn_mutants
allcf = []
for ds in DATASETS:
    for sc in ("pre_chunk", "post_chunk"):
        rows = [(cf, bool(REF2.search(cf))) for _, cf in cn_mutants(ds, sc)]
        k = sum(h for _, h in rows); allcf += [(ds, sc, cf, h) for cf, h in rows]
        print(f"{ds:<10}{sc:<11} refusals {k:>3}/{len(rows)} = {k/len(rows):.3f}")
K = sum(x[3] for x in allcf); print(f"TOTAL refusals (v2 pattern): {K}/{len(allcf)} = {K/len(allcf):.3f}")
rng = random.Random(5); hits = [x for x in allcf if x[3]]; miss = [x for x in allcf if not x[3]]
print("\nPRECISION sample (30 random matches):"); [print(f"  [{d[:5]}] {c[:110]!r}") for d, _, c, _ in rng.sample(hits, 30)]
print("\nRECALL sample (60 random non-matches):"); [print(f"  [{d[:5]}] {c[:110]!r}") for d, _, c, _ in rng.sample(miss, 60)]
