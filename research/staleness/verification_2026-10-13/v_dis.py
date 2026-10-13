import collections
from vcommon import CACHE, P9, jl, mine_commit, sample
_, _, qs = sample()
R = jl(CACHE / "p11/records_scale.jsonl")
rows = []
for r in R:
    q = qs[r["q"]]
    m, p = mine_commit(q["new"], q["old"], r["output"]), P9.commit(q["new"], q["old"], r["output"])
    if m != p:
        rows.append((r["gen"], r["policy"], m, p, r["output"], q["new"], q["old"], q["key"]))
print(collections.Counter((g, pol, m, p) for g, pol, m, p, *_ in rows))
seen = set()
for g, pol, m, p, out, new, old, key in rows:
    if (out, tuple(new), tuple(old)) in seen: continue
    seen.add((out, tuple(new), tuple(old)))
    print(f"{g[:10]:<10} {pol:<7} mine={m:<7} p9={p:<7} out={out!r:<28} new={new} old={old} [{key}]")
