"""Shared loaders for the independent verification (2026-10-13 chat).

Imports ONLY the data-construction code (p9 build/fields, p5 chunks, e0 prose) needed to
rebuild the question sample and passages.  Scoring, retrieval and statistics are written here
from scratch and do not call P9.commit / generation.classify / any report() function.
"""
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path.home() / "SemProject/Drift-aware-RAG-Label-free-detection-of-retrieval-degradation"
PRE = REPO / "research/staleness/pretests"
sys.path.insert(0, str(PRE))
sys.path.insert(0, str(REPO))
_argv = sys.argv
sys.argv = [sys.argv[0], "build"]
import p9_wiki_infobox as P9  # noqa: E402  (data construction only)
import p5_chunked as C  # noqa: E402
sys.argv = _argv
from research.staleness import data  # noqa: E402  (cache paths / digest only)

CACHE = data.CACHE
FROZEN = CACHE / "p11_wiki_frozen"
P9.RAW = FROZEN
YEAR = re.compile(r"(19|20)\d\d")

# ---------------------------------------------------------------- independent number reader
# A number token: either comma-grouped thousands (1,234,567.8) or a plain decimal.  A comma
# NOT followed by exactly three digits is a separator, so "36.77,29.86" -> 36.77, 29.86.
TOK = re.compile(r"(?<![\d.])(\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)(?![\d])")


def numbers(text: str) -> set[float]:
    out = set()
    for m in TOK.finditer(text):
        try:
            out.add(round(float(m.group(1).replace(",", "")), 9))
        except ValueError:
            pass
    return out


def val(v: str) -> float | None:
    """Value of an answer string from the records ('36.77', '1,234', '12%', '450K')."""
    t = v.strip().rstrip("%").strip()
    mult = 1
    if t and t[-1] in "KMBk":
        mult = {"K": 1e3, "k": 1e3, "M": 1e6, "B": 1e9}[t[-1]]
        t = t[:-1].strip()
    try:
        return round(float(t.replace(",", "")) * mult, 9)
    except ValueError:
        return None


def raw_variants(v: str) -> set[float]:
    """Values a reader would accept for answer string v: the number as written (and, for a
    K/M/B suffix, also the bare mantissa, which is how such values appear in text)."""
    out = {val(v)}
    t = v.strip().rstrip("%").strip()
    if t and t[-1] in "KMBk":
        out.add(val(t[:-1]))
    return {x for x in out if x is not None}


def mine_commit(new, old, out: str) -> str:
    said = numbers(out)
    hn = any(said & raw_variants(v) for v in new)
    ho = any(said & raw_variants(v) for v in old)
    return {(1, 0): "new", (0, 1): "old", (1, 1): "both", (0, 0): "neither"}[(hn, ho)]


# ---------------------------------------------------------------- stats, from scratch
def wilson(k, n, z=1.959963984540054):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def page_boot(x, pages, reps=4000, seed=12345):
    x, pages = np.asarray(x, float), np.asarray(pages)
    groups = {p: x[pages == p] for p in np.unique(pages)}
    keys = list(groups)
    rng = np.random.default_rng(seed)
    sums = np.array([groups[k].sum() for k in keys]); ns = np.array([len(groups[k]) for k in keys])
    bs = []
    for _ in range(reps):
        pick = rng.integers(0, len(keys), len(keys))
        bs.append(sums[pick].sum() / ns[pick].sum())
    return tuple(np.percentile(bs, [2.5, 97.5]))


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, p)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    # average ties
    for v in np.unique(allv):
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    u = ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2
    return u / (len(pos) * len(neg))


# ---------------------------------------------------------------- data rebuild
def pages():
    out = []
    for path in sorted(FROZEN.glob("*.json")):
        if path.name == "pages.json":
            continue
        p = json.loads(path.read_text())
        if p.get("skip") or not p.get("text0") or not p.get("text1"):
            continue
        out.append(p)
    return out


def sample():
    docs, allq = P9.build()
    return docs, allq, random.Random(0).sample(allq, min(P9.SAMPLE, len(allq)))


def versions(pgs):
    W = P9.W
    vs, fs = [], []
    for p in pgs:
        f0, f1 = P9.fields(p["text0"]), P9.fields(p["text1"])
        fp = lambda f: [f"{p['title']} -- {k.replace('_', ' ')}: {v}." for k, v in f.items()]  # noqa: E731
        vs.append((fp(f0) + C.chunks(W.prose(p["text0"])), fp(f1) + C.chunks(W.prose(p["text1"]))))
        fs.append((f0, f1))
    return vs, fs


def load_emb(model, corpus, what, texts):
    path = CACHE / model / f"{corpus}__{what}__{data.digest(model, list(texts))}.npy"
    assert path.exists(), f"missing cached embeddings {path.name} (would need recomputation)"
    E = np.load(path)
    assert len(E) == len(texts)
    return E


def gen_cache(model):
    store = CACHE / "p9_generation" / f"{model.replace(':', '_')}.jsonl"
    d = {}
    for line in open(store):
        r = json.loads(line)
        d[r["key"]] = r["output"]
    return d


def gkey(model, prompt):
    return hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]
