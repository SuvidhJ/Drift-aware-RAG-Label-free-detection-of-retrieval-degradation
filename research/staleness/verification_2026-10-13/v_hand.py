import json, random
from vcommon import CACHE, P9, jl, sample
_, _, qs = sample()
ctx = json.load(open("ctx.json"))
R = jl(CACHE / "p11/records_scale.jsonl")
rng = random.Random(20261013)
for gen in ("phi3:mini", "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"):
    ap = [r for r in R if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"]
    print(f"\n######## {gen}")
    for i, r in enumerate(rng.sample(ap, 30)):
        q = qs[r["q"]]
        key = q["key"].replace("_", " ")
        texts = ctx[f"{r['retriever']}|append|{r['q']}"]
        print(f"[{i+1}] {r['retriever']} q{r['q']} | {q['statement']} | old={q['old']} new={q['new']} | P9.commit={P9.commit(q['new'], q['old'], r['output'])}")
        for t in texts:
            s = t if len(t) < 150 else t[:150] + "…"
            print("     ·", s.replace("\n", " "))
        print(f"     OUT: {r['output']!r}")
