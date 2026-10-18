"""POST HOC: published CN rates with v2 refusals excluded; verdicts and preservation labels of refusal mutants."""
from collections import Counter
from refusal_common import REF2, DATASETS, cn_mutants, verdicts
tot = Counter()
for ds in DATASETS:
    for sc in ("pre_chunk", "post_chunk"):
        V = verdicts(ds, sc); n = vio = n_ok = vio_ok = 0; lab = Counter(); ver = Counter()
        for m, cf in cn_mutants(ds, sc):
            v = V.get((m["question_id"], m["mutation_operator"]))
            if v is None: continue
            r = bool(REF2.search(cf)); n += 1; vio += v == "VIOLATED"
            if r: lab["altering" if m.get("is_semantics_preserving") is False else "preserving"] += 1; ver[v] += 1
            else: n_ok += 1; vio_ok += v == "VIOLATED"
        tot.update({"n": n, "ref": n - n_ok, **{f"lab_{k}": c for k, c in lab.items()}, **{f"ver_{k}": c for k, c in ver.items()}})
        print(f"{ds:<10}{sc:<11} published {vio}/{n} = {vio/n:.3f} | excl. refusals {vio_ok}/{n_ok} = {vio_ok/n_ok:.3f} | refusals {dict(lab)} {dict(ver)}")
print(f"\nTOTAL refusals {tot['ref']}/{tot['n']}; preserving {tot['lab_preserving']}, altering {tot['lab_altering']}; SATISFIED {tot['ver_SATISFIED']}, VIOLATED {tot['ver_VIOLATED']}")
