import collections, json, re
import numpy as np
from vcommon import CACHE, P9, YEAR, C, jl, load_emb, pages, sample
norm, NUMRE, W = P9.facts.norm, P9.facts.NUM, P9.W
pgs = pages(); _, _, qs = sample()
cfs = json.load(open("cfs.json"))
cur = [[f"{p['title']} -- {k.replace('_', ' ')}: {v}." for k, v in P9.fields(p["text1"]).items()] + C.chunks(W.prose(p["text1"])) for p in pgs]
p11u = set()
for p in pgs:
    for t in ("text0", "text1"):
        p11u |= {f"{p['title']} -- {k.replace('_', ' ')}: {v}." for k, v in P9.fields(p[t]).items()} | set(C.chunks(W.prose(p[t])))
p11u = sorted(p11u)
R12 = jl(CACHE / "p12/records.jsonl")
keys = {(r["retriever"], r["q"]) for r in R12}
def has(vs, out):
    return any(re.search(r"(?<![\d.])" + re.escape(f) + r"(?!\d)", out) for v in vs for f in {v, v.replace(",", "")})
stats = collections.Counter(); changed = []
incl = {}
for retr in ("minilm", "bge"):
    E = np.vstack([load_emb(retr, "p11_scale", "passages", p11u), load_emb(retr, "p12", "counterfactuals", cfs)])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    Q = load_emb(retr, "p11_scale", "questions", [q["question"] for q in qs])
    idx = {}
    for i, x in enumerate(p11u + cfs): idx.setdefault(x, i)
    base = [x for ps in cur for x in ps]; rows = np.array([idx[x] for x in base])
    for n, q in enumerate(qs):
        cf = cfs[n]
        allrows = np.append(rows, idx[cf]); texts_all = base + [cf]
        top = np.argsort(-(E[allrows] @ Q[n]))[:3]
        texts = [texts_all[t] for t in top]
        cfv_all = {norm(x) for x in NUMRE.findall(cf) if len(x) >= 3}
        cfv_noyear = {norm(x) for x in NUMRE.findall(cf.split(":", 1)[-1]) if len(x) >= 3 and not YEAR.fullmatch(x)}
        vals = set().union(*(P9.facts.values(t) for t in texts))
        hn = bool(vals & {norm(a) for a in q["new"]})
        sel_rec = (retr, n) in keys
        sel_noyear = hn and bool(vals & cfv_noyear) and (cf in texts)
        stats[(retr, "selected_in_records")] += sel_rec
        if sel_rec:
            stats[(retr, "cf_passage_in_top3")] += cf in texts
            stats[(retr, "cf_number_in_ctx_excluding_years")] += bool(vals & cfv_noyear)
        stats[(retr, "would_be_selected_without_years_and_cf_in_top3")] += sel_noyear
        incl[(retr, n)] = (sel_rec, sel_noyear)
print(dict(stats))
# label change if years dropped from the injected values
for r in R12:
    cf = cfs[r["q"]]
    inj_y = [x for x in NUMRE.findall(cf.split(":", 1)[-1]) if len(x) >= 3]
    inj_n = [x for x in inj_y if not YEAR.fullmatch(x)]
    new = qs[r["q"]]["new"]
    c2 = {(1, 0): "current", (0, 1): "injected", (1, 1): "both", (0, 0): "neither"}[(has(new, r["output"]), has(inj_n, r["output"]))]
    if c2 != r["choice"]:
        changed.append((r["gen"], r["choice"], c2, r["output"], new, inj_n, cf))
print("labels that change when years are dropped from injected values:", len(changed))
print(__import__("collections").Counter((c[0], c[1], c[2]) for c in changed))
# rates restricted to cases selected without year help
for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
    rows = [r for r in R12 if r["gen"] == gen and incl[(r["retriever"], r["q"])][1]]
    print(gen, "cases valid without year:", len(rows), "injected-only", round(np.mean([r["choice"] == "injected" for r in rows]), 3))
