"""Pre-test 11: the p7/p8 effect at scale (per-field passages) and on real human questions.

Pre-registered 2026-10-11, before any p11 retrieval or generation.  p9 was void twice
(year-token bug; then low retrieval coverage and Qwen adequacy 0.51).  This redesign after
a void run must be disclosed as such.

SCALE   pages fetched by p9 (UN members + Nasdaq/NYSE samples), frozen when this stage
        starts.  Passages: one per infobox field, "<page> -- <field>: <value>.", plus the
        page's prose chunked as in p5.  Questions as p9 (year fix), seeded sample of 400.
FRESH   the 23 usable FreshQA questions of p10 (real human questions); passages are p5
        chunks of the 2024 and current page text of all 117 fetched FreshQA pages.
        Answers are names or numbers, so scoring is by normalised phrase: correct if the
        new answer appears, stale if only the old one does.  n is small, so this set is
        descriptive only (no kill rule).

Both    `replace` vs `append` index; minilm + bge; k = 3; generators phi3:mini and
        qwen2.5:7b-instruct (pass `--14b` to add qwen2.5:14b-instruct); labels none vs dated.
Rules (SCALE)
        adequacy: replace, new value in context, correct >= 0.60 for both generators,
                  otherwise VOID
        replicates if stale-when-both (no labels) >= 0.10 for both generators; a page-
                  clustered bootstrap CI is reported alongside Wilson
        secondary: dated vs none, paired McNemar

    .venv/bin/python research/staleness/pretests/p11_field_passages.py scale [--14b]
    .venv/bin/python research/staleness/pretests/p11_field_passages.py fresh [--14b]
"""
import json
import random
import re
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
STAGE = sys.argv[1] if len(sys.argv) > 1 else "scale"
GENS = ["phi3:mini", "qwen2.5:7b-instruct"] + (["qwen2.5:14b-instruct"] if "--14b" in sys.argv else [])
sys.argv = sys.argv[:1] + ["stats"]
import p5_chunked as C  # noqa: E402
import p9_wiki_infobox as P9  # noqa: E402
from research.staleness import answer_level, data, facts, generation  # noqa: E402

W = P9.W
OUT = data.CACHE / "p11"
QA_TEMPLATE = ("Context:\n{context}\n\nAnswer the question using only the context. Reply with only "
               "the answer. If the context does not contain it, reply: unknown\n\nQuestion: {question}")


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def has(phrase: str, text: str) -> bool:
    p = norm(phrase)
    return bool(p) and f" {p} " in f" {norm(text)} "


def classify_text(out: str, new: str, old: str) -> str:
    if has(new, out):
        return "correct"
    if has(old, out):
        return "stale"
    return "unknown" if "unknown" in out.lower() else "other"


def run_eval(docs, versions, qs, stmt_of, has_new_old, classify, tag):
    """docs: list of page dicts with title, rev0, rev1. versions: per doc (old passages, new passages)."""
    uniq = sorted({p for v0, v1 in versions for p in v0 + v1})
    idx = {p: i for i, p in enumerate(uniq)}
    recs, jobs = [], []
    for retr in ("minilm", "bge"):
        E = data.cached(retr, f"p11_{tag}", "passages", lambda: uniq)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        Q = data.cached(retr, f"p11_{tag}", "questions", lambda: [q["question"] for q in qs])
        for pol in ("replace", "append"):
            items = []
            for d, (v0, v1) in enumerate(versions):
                s0, s1 = set(v0), set(v1)
                if pol == "append":
                    items += [(p, d, "both" if p in s1 else "old") for p in v0]
                    items += [(p, d, "new") for p in v1 if p not in s0]
                else:
                    items += [(p, d, "new") for p in v1]
            top = np.argsort(-(Q @ E[[idx[p] for p, _, _ in items]].T), axis=1)[:, :3]
            for n, q in enumerate(qs):
                texts = [items[t][0] for t in top[n]]
                hn, ho = has_new_old(q, texts)
                if (pol == "replace" and not hn) or (pol == "append" and not (hn and ho)):
                    continue
                for variant in (("none", "dated") if pol == "append" else ("none",)):
                    def lab(t):
                        doc, v = docs[items[t][1]], items[t][2]
                        when = {"old": doc["rev0"], "new": doc["rev1"], "both": f"{doc['rev0']} and {doc['rev1']}"}[v]
                        return f"({doc['title']}, as of {when}) " if variant == "dated" else ""
                    ctx = "\n\n".join(f"[{j+1}] {lab(t)}{items[t][0]}" for j, t in enumerate(top[n]))
                    for gen in GENS:
                        # 14B scope (decided 2026-10-12 before any 14B result, for runtime: ~20 s
                        # per answer): adequacy + no-label stale only, not the dated variant
                        if gen == "qwen2.5:14b-instruct" and variant == "dated":
                            continue
                        jobs.append(({"retriever": retr, "policy": pol, "variant": variant, "gen": gen, "q": n,
                                      "page": q["doc"], "qtext": q.get("statement", q["question"])},
                                     stmt_of(q, ctx), q))
        print(f"[{tag}/{retr}] {len(jobs)} prompts", flush=True)
    # one generator at a time, so only one model is held in memory (three together exceed 16 GB)
    for gen in GENS:
        for r, prompt, q in jobs:
            if r["gen"] == gen:
                out = P9.generate(gen, prompt)
                recs.append({**r, "class": classify(out, q), "output": out})
        subprocess.run(["ollama", "stop", gen], check=False)
        print(f"[{tag}] {gen} done", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"records_{tag}.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    return recs


def report(recs, kill_rules: bool):
    from scipy.stats import binomtest
    rng = np.random.default_rng(0)
    for gen in GENS:
        ad = [r for r in recs if r["gen"] == gen and r["policy"] == "replace"]
        k = sum(r["class"] == "correct" for r in ad)
        verdict = ("ok" if k / max(len(ad), 1) >= 0.60 else "VOID") if kill_rules else "descriptive"
        print(f"{gen:<22} ADEQUACY replace, new in context: correct {k}/{len(ad)} = {k/max(len(ad),1):.2f} "
              f"{np.round(answer_level.wilson(k, len(ad)), 3)} -> {verdict}")
        base = {(r["retriever"], r["q"]): r["class"] for r in recs
                if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"}
        for variant in ("none", "dated"):
            rows = [r for r in recs if r["gen"] == gen and r["policy"] == "append" and r["variant"] == variant]
            if not rows:
                print(f"{gen:<22} append labels={variant}: no both-in-context cases"); continue
            st = np.array([r["class"] == "stale" for r in rows]); pages = np.array([r["page"] for r in rows])
            up = np.unique(pages)
            boots = []
            for _ in range(2000):                 # page-clustered bootstrap
                pick = rng.choice(up, len(up))
                m = np.concatenate([st[pages == p] for p in pick])
                boots.append(m.mean())
            lo, hi = np.percentile(boots, [2.5, 97.5])
            line = (f"{gen:<22} append k=3 both-in-context labels={variant:<5}: stale {st.sum()}/{len(st)} = {st.mean():.2f} "
                    f"Wilson {np.round(answer_level.wilson(int(st.sum()), len(st)), 3)} page-boot [{lo:.3f}, {hi:.3f}] "
                    f"({len(up)} pages)  correct {np.mean([r['class'] == 'correct' for r in rows]):.2f}")
            if variant == "none" and kill_rules:
                line += "  -> " + ("REPLICATES" if st.mean() >= 0.10 else "does NOT replicate (< 0.10)")
            if variant == "dated":
                b = sum(base[(r["retriever"], r["q"])] == "stale" and r["class"] != "stale" for r in rows)
                c = sum(base[(r["retriever"], r["q"])] != "stale" and r["class"] == "stale" for r in rows)
                line += f"  McNemar vs none: fixed {b}, broke {c}, p={binomtest(b, b + c, 0.5).pvalue if b + c else 1:.3g}"
            print(line, flush=True)


def scale():
    frozen = data.CACHE / "p11_wiki_frozen"
    if not frozen.exists():
        shutil.copytree(P9.RAW, frozen)
    P9.RAW = frozen
    pages, qs = [], []
    for path in sorted(frozen.glob("*.json")):
        if path.name == "pages.json":
            continue
        p = json.loads(path.read_text())
        if p.get("skip") or not p.get("text0") or not p.get("text1"):
            continue
        pages.append(p)
    docs, versions = [], []
    for p in pages:
        f0, f1 = P9.fields(p["text0"]), P9.fields(p["text1"])
        fp = lambda f: [f"{p['title']} -- {k.replace('_', ' ')}: {v}." for k, v in f.items()]  # noqa: E731
        versions.append((fp(f0) + C.chunks(W.prose(p["text0"])), fp(f1) + C.chunks(W.prose(p["text1"]))))
        docs.append({"title": p["title"], "rev0": p["ts0"][:10], "rev1": p["ts1"][:10]})
    _, allq = P9.build()                     # p9 questions (year fix) on the same frozen pages
    qs = random.Random(0).sample(allq, min(P9.SAMPLE, len(allq)))
    print(f"SCALE: {len(docs)} pages, {len(allq)} questions, sampled {len(qs)}", flush=True)

    def hno(q, texts):
        vals = set().union(*(facts.values(t) for t in texts))
        return bool(vals & {facts.norm(a) for a in q["new"]}), bool(vals & {facts.norm(a) for a in q["old"]})
    recs = run_eval(docs, versions, qs, lambda q, ctx: P9.TEMPLATE.format(context=ctx, statement=q["statement"]),
                    hno, lambda out, q: generation.classify(out, q["new"], q["old"]), "scale")
    report(recs, kill_rules=True)


def fresh():
    import p10_freshqa_feasibility as F
    feas = json.loads((F.RAW / "feasibility.json").read_text())
    titles = sorted({c["title"] for c in feas})
    docs, versions, tix = [], [], {}
    for t in titles:
        path = F.RAW / f"page_{F.norm(t).replace(' ', '_')[:80]}.json"
        pt = json.loads(path.read_text())
        if not pt["t0"] or not pt["t1"]:
            continue
        tix[t] = len(docs)
        docs.append({"title": t, "rev0": "2024-02-26", "rev1": "2026-10-11"})
        versions.append((C.chunks(pt["t0"]), C.chunks(pt["t1"])))
    qs = [{**c, "doc": tix[c["title"]]} for c in feas if c["usable"] and c["title"] in tix]
    print(f"FRESH: {len(docs)} pages, {len(qs)} real questions", flush=True)
    recs = run_eval(docs, versions, qs, lambda q, ctx: QA_TEMPLATE.format(context=ctx, question=q["question"]),
                    lambda q, texts: (any(has(q["new"], t) for t in texts), any(has(q["old"], t) for t in texts)),
                    lambda out, q: classify_text(out, q["new"], q["old"]), "fresh")
    report(recs, kill_rules=False)


if __name__ == "__main__":
    {"scale": scale, "fresh": fresh}[STAGE]()
