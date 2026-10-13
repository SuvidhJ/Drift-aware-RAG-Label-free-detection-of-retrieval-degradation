"""EXPLORATORY (2026-10-13): apply ASE's *released* numeric MR oracle (pipeline_config.parse_numeric_answer /
numeric_answers_match, verbatim) to our CN (p12) and real-update (p11) records, and test order dependence.
Semantics-altering CN: SATISFIED iff the parsed answer != original (current) value.  Real-update analogue:
original = old value."""
import collections, json, re, sys
import numpy as np
sys.path.insert(0, "ase")
from vcommon import CACHE, P9, C, jl, load_emb, pages, sample, page_boot
_re = re
NUMERIC_TOLERANCE = 0.01
def parse_numeric_answer(text):                       # verbatim from ASE pipeline_config.py L752-771
    if not text or text.strip().upper() in ("N/A", "NONE", "NULL", ""):
        return None
    cleaned = text.strip().replace(",", "").replace("%", "").replace("$", "")
    cleaned = cleaned.replace("million", "").replace("billion", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        pass
    matches = _re.findall(r"-?\d+\.?\d*", cleaned)
    if matches:
        return float(matches[-1])
    return None
def numeric_answers_match(pred, gt, tol=NUMERIC_TOLERANCE):   # verbatim L774-789
    p = parse_numeric_answer(pred)
    if p is None:
        return False
    if gt == 0.0:
        return abs(p - gt) < 1e-6
    return abs(p - gt) / abs(gt) <= tol
def ase_verdict(out, orig):                           # _check_mr_numeric, semantics-altering branch
    if out.strip().upper() == "N/A":
        return "SATISFIED"
    return "VIOLATED" if numeric_answers_match(out, orig) else "SATISFIED"
norm = P9.facts.norm; NUMRE = P9.facts.NUM
pgs = pages(); _, _, qs = sample()
cfs = json.load(open("cfs.json"))
ctx = json.load(open("ctx.json"))
fv = lambda v: float(norm(v))
# ---- CN contexts: rebuild top-3 (as v_p12b) to get the order of current vs injected passage
W = P9.W
cur = [[f"{p['title']} -- {k.replace('_', ' ')}: {v}." for k, v in P9.fields(p["text1"]).items()] + C.chunks(W.prose(p["text1"])) for p in pgs]
p11u = set()
for p in pgs:
    for t in ("text0", "text1"):
        p11u |= {f"{p['title']} -- {k.replace('_', ' ')}: {v}." for k, v in P9.fields(p[t]).items()} | set(C.chunks(W.prose(p[t])))
p11u = sorted(p11u)
order_cn = {}
for retr in ("minilm", "bge"):
    E = np.vstack([load_emb(retr, "p11_scale", "passages", p11u), load_emb(retr, "p12", "counterfactuals", cfs)])
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    Q = load_emb(retr, "p11_scale", "questions", [q["question"] for q in qs])
    idx = {}
    for i, x in enumerate(p11u + cfs): idx.setdefault(x, i)
    base = [x for ps in cur for x in ps]; rows = np.array([idx[x] for x in base])
    for n, q in enumerate(qs):
        allrows = np.append(rows, idx[cfs[n]]); texts = (base + [cfs[n]])
        top = [texts[t] for t in np.argsort(-(E[allrows] @ Q[n]))[:3]]
        key = " -- " + q["key"].replace("_", " ") + ":"
        pc = [i for i, t in enumerate(top) if t != cfs[n] and key in t and any(norm(x) in {norm(a) for a in q["new"]} for x in NUMRE.findall(t.split(":", 1)[1]))]
        pi = [i for i, t in enumerate(top) if t == cfs[n]]
        order_cn[(retr, n)] = ("injected_after" if pi and pc and pi[0] > pc[0] else "injected_before" if pi and pc else "na")
R12 = jl(CACHE / "p12/records.jsonl"); R11 = jl(CACHE / "p11/records_scale.jsonl")
S = {(c["retriever"], c["q"]): c for c in json.load(open("stress_cases.json")) if c["gen"] == "phi3:mini"}
print("== ASE numeric oracle applied to our records (semantics-altering branch)")
for gen in ("phi3:mini", "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"):
    rows = []
    for r in R12:
        if r["gen"] != gen: continue
        q = qs[r["q"]]
        rows.append(("CN", ase_verdict(r["output"], fv(q["new"][0])), r["choice"].replace("current", "orig").replace("injected", "mut"), order_cn[(r["retriever"], r["q"])], r["page"]))
    for r in R11:
        if r["gen"] != gen or r["policy"] != "append" or r["variant"] != "none": continue
        q = qs[r["q"]]; s = S[(r["retriever"], r["q"])]
        c = P9.commit(q["new"], q["old"], r["output"]).replace("old", "orig").replace("new", "mut")
        o = "injected_after" if s["pn"] > s["po"] else "injected_before"     # 'mutation' = the new version
        rows.append(("REAL", ase_verdict(r["output"], fv(q["old"][0])), c, o, r["page"]))
    for src in ("CN", "REAL"):
        rs = [x for x in rows if x[0] == src]
        if not rs: continue
        v = np.mean([x[1] == "VIOLATED" for x in rs])
        sat = [x for x in rs if x[1] == "SATISFIED"]; vio = [x for x in rs if x[1] == "VIOLATED"]
        cs = collections.Counter(x[2] for x in sat); cv = collections.Counter(x[2] for x in vio)
        print(f"\n{gen} {src}: n={len(rs)}  ASE violation rate {v:.3f}")
        print(f"   SATISFIED broken down by what the answer actually was: {dict(cs)}  -> 'neither' (wrong/hallucinated) counted as satisfied: {cs['neither']}/{len(sat)}")
        print(f"   VIOLATED  broken down: {dict(cv)}  -> 'both' answers judged by LAST number: {cv['both']}")
        for o in ("injected_before", "injected_after"):
            ro = [x for x in rs if x[3] == o]
            if ro:
                print(f"   mutation passage {o:<16} n={len(ro):<4} ASE violation {np.mean([x[1]=='VIOLATED' for x in ro]):.3f}")
