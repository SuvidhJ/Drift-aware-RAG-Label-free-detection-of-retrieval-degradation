import collections, json, urllib.request
from vcommon import CACHE, P9, jl, sample
_, _, qs = sample()
R = jl(CACHE / "p11/records_scale.jsonl")
# token counts via the local Ollama tokenizer is not exposed; use a character proxy and inspect.
for gen in ("phi3:mini", "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"):
    ap = [r for r in R if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"]
    L = [len(r["output"]) for r in ap]
    long_ = [r for r in ap if len(r["output"]) >= 60]
    c = collections.Counter(P9.commit(qs[r["q"]]["new"], qs[r["q"]]["old"], r["output"]) for r in long_)
    print(f"{gen}: append none n={len(ap)}; median len {sorted(L)[len(L)//2]}; >=60 chars: {len(long_)} {dict(c)}")
    for r in long_[:6]:
        q = qs[r["q"]]
        print(f"    {P9.commit(q['new'], q['old'], r['output']):<7} new={q['new']} old={q['old']} | {r['output']!r}")
