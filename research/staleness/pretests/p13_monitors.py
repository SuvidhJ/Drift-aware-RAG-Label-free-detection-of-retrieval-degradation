"""Pre-test 13 (RQ3): which monitors can see update-induced faults?

Pre-registered 2026-10-12, before any p13 judge call or detector run.
Faults   p11 SCALE `append` (no labels) both-in-context cases.  The label comes from the
         deterministic diff oracle: stale (positive) vs correct (negative); 'other' and
         'unknown' answers are excluded.  Generators phi3 and qwen2.5:7b, as in p11.
Monitors
  M1 retrieval     gold-page hit@3 for all 400 questions under stale-only, append and replace
                   indexes; per case the score is constant (1) by construction
  M2 faithfulness  RAGAS-style claim support, judged by a local LLM (qwen2.5:7b-instruct,
                   temperature 0): "Is the answer supported by the context? yes/no"; it
                   flags when the answer is unsupported.  This is RAGAS's *metric definition*
                   run with a local judge, not the RAGAS library.
  M3 drift         (a) batch level: permutation-calibrated sliced-W1 and MMD between old- and
                   new-version embeddings of the updated pages' passages (drift/ modules);
                   (b) case level: displacement of the queried field's passage,
                   1 - cos(old, new), as a stale-answer score
  M4 judge oracle  ASE-style LLM judge given the question, context, the perturbation
                   ("<field> changed from <old> to <new>") and the answer; it flags when the
                   judge says the answer does not reflect the update
Amended 2026-10-13, before any p13 run: labels come from P9.commit (digit-boundary
         matching); stale = old value only (positive), correct = new value only (negative).
         Answers giving both values are excluded and counted.  The p11 classifier misread
         comma-joined answers ("36.77,29.86") and counted "both" as correct.
Reading  per monitor and generator: AUROC (M3b) or TPR on stale with FPR on correct
         (M2, M4); 95% bootstrap CIs.  A monitor "sees" the fault if AUROC >= 0.75, or
         TPR >= 0.5 at FPR <= 0.10.  M4 is also scored for agreement with the
         deterministic oracle.

    .venv/bin/python research/staleness/pretests/p13_monitors.py
"""
import json
import random
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
sys.argv = sys.argv[:1] + ["stats"]
import p5_chunked as C  # noqa: E402
import p9_wiki_infobox as P9  # noqa: E402
from drift.mmd import mmd_rbf_fast, median_heuristic_sigma  # noqa: E402
from drift.null import permutation_null  # noqa: E402
from drift.wasserstein import sliced_w1_fast  # noqa: E402
from research.staleness import data, facts  # noqa: E402

W = P9.W
JUDGE = "qwen2.5:7b-instruct"
FAITH = ("Context:\n{context}\n\nQuestion: {question}\nAnswer: {answer}\n\nIs the answer fully supported by "
         "the context? Reply with only yes or no.")
ASE = ("The knowledge base was updated: the {field} of {title} changed from {old} to {new}.\n\nContext given "
       "to the system:\n{context}\n\nQuestion: {question}\nSystem answer: {answer}\n\nDoes the system answer "
       "correctly reflect the updated knowledge base? Reply with only yes or no.")


def yes(out: str) -> bool:
    return out.strip().lower().startswith("yes")


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return float("nan")
    return float((np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])))


def main():
    frozen = data.CACHE / "p11_wiki_frozen"
    P9.RAW = frozen
    pages = [json.loads(p.read_text()) for p in sorted(frozen.glob("*.json")) if p.name != "pages.json"]
    pages = [p for p in pages if not p.get("skip") and p.get("text0") and p.get("text1")]
    _, allq = P9.build()
    qs = random.Random(0).sample(allq, min(P9.SAMPLE, len(allq)))
    versions, fields = [], []
    for p in pages:
        f0, f1 = P9.fields(p["text0"]), P9.fields(p["text1"])
        fp = lambda f: [f"{p['title']} -- {k.replace('_', ' ')}: {v}." for k, v in f.items()]  # noqa: E731
        versions.append((fp(f0) + C.chunks(W.prose(p["text0"])), fp(f1) + C.chunks(W.prose(p["text1"]))))
        fields.append((f0, f1))
    uniq = sorted({x for v0, v1 in versions for x in v0 + v1})
    idx = {x: i for i, x in enumerate(uniq)}
    p11 = [json.loads(l) for l in open(data.CACHE / "p11" / "records_scale.jsonl")]
    out = {}

    for retr in ("minilm", "bge"):
        E = data.cached(retr, "p11_scale", "passages", lambda: uniq)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        Q = data.cached(retr, "p11_scale", "questions", lambda: [q["question"] for q in qs])
        # ---- M1: gold-page hit@3 under three indexes
        for pol in ("stale", "append", "replace"):
            items = []
            for d, (v0, v1) in enumerate(versions):
                if pol == "stale":
                    items += [(x, d) for x in v0]
                elif pol == "replace":
                    items += [(x, d) for x in v1]
                else:
                    items += [(x, d) for x in v0] + [(x, d) for x in v1 if x not in set(v0)]
            top = np.argsort(-(Q @ E[[idx[x] for x, _ in items]].T), axis=1)[:, :3]
            hit = np.mean([any(items[t][1] == q["doc"] for t in tt) for q, tt in zip(qs, top)])
            out[f"M1/{retr}/{pol}"] = float(hit)
            print(f"M1 [{retr}] gold-page hit@3, {pol:<7} index: {hit:.3f}", flush=True)
            if pol == "append":
                ctxs = {n: [items[t][0] for t in tt] for n, tt in enumerate(top)}
        # ---- cases with contexts (append, no labels)
        cases, n_both = [], 0
        for r in p11:
            if (r["retriever"] == retr and r["policy"] == "append" and r["variant"] == "none"
                    and r["gen"] in ("phi3:mini", "qwen2.5:7b-instruct")):
                assert r["qtext"] == qs[r["q"]]["statement"], "p11 records come from a different question sample"
                r["commit"] = P9.commit(qs[r["q"]]["new"], qs[r["q"]]["old"], r["output"])
                if r["commit"] in ("old", "new"):        # 'both' is reported, not labelled
                    cases.append(r)
                n_both += r["commit"] == "both"
        print(f"[{retr}] answers giving both values, excluded: {n_both}", flush=True)
        for r in cases:
            q = qs[r["q"]]
            ctx = "\n\n".join(f"[{j+1}] {t}" for j, t in enumerate(ctxs[r["q"]]))
            r["M2_flag"] = not yes(P9.generate(JUDGE, FAITH.format(context=ctx, question=q["question"], answer=r["output"])))
            r["M4_flag"] = not yes(P9.generate(JUDGE, ASE.format(
                field=q["key"].replace("_", " "), title=pages[q["doc"]]["title"], old=", ".join(q["old"]),
                new=", ".join(q["new"]), context=ctx, question=q["question"], answer=r["output"])))
            f0, f1 = fields[q["doc"]]
            a = f"{pages[q['doc']]['title']} -- {q['key'].replace('_', ' ')}: {f0[q['key']]}."
            b = f"{pages[q['doc']]['title']} -- {q['key'].replace('_', ' ')}: {f1[q['key']]}."
            r["M3b_score"] = float(1 - E[idx[a]] @ E[idx[b]])
        out.setdefault("cases", []).extend(cases)
        print(f"[{retr}] {len(cases)} labelled cases judged", flush=True)
        # ---- M3a: batch-level drift on the updated pages' changed passages
        changed = [(x, y) for v0, v1 in versions for x, y in zip(sorted(set(v0) - set(v1)), sorted(set(v1) - set(v0)))]
        X = E[[idx[x] for x, _ in changed]]
        Y = E[[idx[y] for _, y in changed]]
        sub = np.random.default_rng(1)
        if len(X) > 1000:                      # MMD is O(n^2) per permutation: subsample both sides
            X = X[sub.choice(len(X), 1000, replace=False)]
            Y = Y[sub.choice(len(Y), 1000, replace=False)]
        for name, fn in (("sliced-W1", lambda a, b: sliced_w1_fast(a, b)),
                         ("MMD", lambda a, b, s=median_heuristic_sigma(X, Y): mmd_rbf_fast(a, b, s))):
            obs = fn(X, Y)
            null = permutation_null(X, Y, fn, n_perm=199, seed=0)
            p = (1 + np.sum(null >= obs)) / (1 + len(null))
            out[f"M3a/{retr}/{name}"] = {"obs": float(obs), "p": float(p), "n": len(changed)}
            print(f"M3a [{retr}] {name} old vs new changed passages (n={len(changed)}): p = {p:.3f}", flush=True)

    rng = np.random.default_rng(0)
    print()
    for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
        cs = [r for r in out["cases"] if r["gen"] == gen]
        pos = [r for r in cs if r["commit"] == "old"]; neg = [r for r in cs if r["commit"] == "new"]
        for m in ("M2", "M4"):
            tpr = np.mean([r[f"{m}_flag"] for r in pos]); fpr = np.mean([r[f"{m}_flag"] for r in neg])
            bt = [np.mean([pos[i][f"{m}_flag"] for i in rng.integers(0, len(pos), len(pos))]) for _ in range(1000)]
            sees = tpr >= 0.5 and fpr <= 0.10
            print(f"{gen:<20} {m}: TPR {tpr:.2f} [{np.percentile(bt, 2.5):.2f},{np.percentile(bt, 97.5):.2f}] "
                  f"FPR {fpr:.2f}  (stale {len(pos)}, correct {len(neg)}) -> {'SEES' if sees else 'does not see'}")
        a = auroc([r["M3b_score"] for r in pos], [r["M3b_score"] for r in neg])
        print(f"{gen:<20} M3b displacement AUROC {a:.3f} -> {'SEES' if a >= 0.75 else 'does not see'}")
        agree = np.mean([r["M4_flag"] == (r["commit"] == "old") for r in cs])
        print(f"{gen:<20} M4 agreement with the deterministic diff oracle: {agree:.3f}")
    (data.CACHE / "p13_results.json").write_text(json.dumps(
        {k: v for k, v in out.items() if k != "cases"}, indent=1))
    with open(data.CACHE / "p13_cases.jsonl", "w") as f:
        for r in out["cases"]:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
