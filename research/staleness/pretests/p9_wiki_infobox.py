"""Pre-test 9: replicate p7/p8 at scale on clean, natural questions from real Wikipedia updates.

Pre-registered 2026-10-11, before any data was fetched.

Data   UN member states + companies from Category:Companies listed on the Nasdaq / NYSE
       (seeded sample).  For each page, the lead section + infobox (rvsection=0) as it
       stood at T0 = 2025-10-01 and as it stands now (T1).
Facts  infobox fields present at both times whose numeric values changed; year/date
       fields excluded; at most 2 numbers per side; >= 1 distinctive (>= 3 chars) new
       value and >= 1 replaced value.  Values come from the rendered text the index holds.
Q      "The <field> of <page> is ___."  (natural, not a blanked revised sentence)
Run    p5 chunks (128); minilm + bge; `replace` vs `append`; k = 3; generators phi3:mini and
       qwen2.5:7b-instruct with p7's prompt; a seeded sample of 400 questions; labels A =
       none, D = "(<page>, as of <date>)".
Primary run on the pages fetched by 2026-10-11 21:15 (countries + Nasdaq sample); a rerun on the
       full fetch is a robustness check, reported whichever way it goes.
Adequacy (added 2026-10-11 before the corrected rerun; p7's rule, wrongly omitted at first)
       replace, new value in context: correct >= 0.60 for both generators, otherwise VOID.
Reading
       replicates if, with no labels, the pooled stale-when-both rate is >= 0.10 for both
       generators; otherwise p7 does not generalise to clean natural questions.
       Secondary: D vs none, paired McNemar.

    .venv/bin/python research/staleness/pretests/p9_wiki_infobox.py fetch
    .venv/bin/python research/staleness/pretests/p9_wiki_infobox.py build
    .venv/bin/python research/staleness/pretests/p9_wiki_infobox.py run
"""
import datetime as dt
import json
import random
import re
import sys
import time

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
STAGE = sys.argv[1] if len(sys.argv) > 1 else "build"
sys.argv = sys.argv[:1] + ["stats"]
import e0_fact_rates as W  # noqa: E402  (API client and wikitext helpers; guarded main)
from research.staleness import data, facts  # noqa: E402

RAW = data.CACHE / "p9_wiki"
T0 = dt.datetime(2025, 10, 1, tzinfo=dt.timezone.utc)
CATS = {"Category:Member states of the United Nations": None,
        "Category:Companies listed on the Nasdaq": 600,
        "Category:Companies listed on the New York Stock Exchange": 600}
SAMPLE = 400
YEAR = re.compile(r"(19|20)\d\d")
# footnotes / traded_as / parent added 2026-10-11 before any generation: not facts (6 questions)
SKIP_KEY = re.compile(r"year|date|ref|as_of|asof|founded|established|coordinates|footnote|traded_as|parent", re.I)


# ------------------------------------------------------------------ fetch
def members(cat: str) -> list[dict]:
    out, cont = [], {}
    while True:
        r = W.get({"action": "query", "list": "categorymembers", "cmtitle": cat, "cmtype": "page",
                   "cmnamespace": 0, "cmlimit": "max", **cont})
        out += r["query"]["categorymembers"]
        if "continue" not in r:
            return out
        cont = r["continue"]


def fetch() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    pages_path = RAW / "pages.json"
    if pages_path.exists():
        pages = json.loads(pages_path.read_text())
    else:
        rng, pages = random.Random(0), []
        for cat, n in CATS.items():
            m = members(cat)
            m = m if n is None else rng.sample(m, min(n, len(m)))
            pages += [{"pageid": x["pageid"], "title": x["title"], "cat": cat} for x in m]
        pages = list({p["pageid"]: p for p in pages}.values())
        pages_path.write_text(json.dumps(pages))
    print(f"{len(pages)} pages", flush=True)
    t0 = time.time()
    for k, p in enumerate(pages):
        path = RAW / f"{p['pageid']}.json"
        if path.exists():
            continue
        try:
            old = W.get({"action": "query", "prop": "revisions", "pageids": p["pageid"], "rvlimit": 1,
                         "rvdir": "older", "rvstart": W.iso(T0), "rvprop": "ids|timestamp"})
            new = W.get({"action": "query", "prop": "revisions", "pageids": p["pageid"], "rvlimit": 1,
                         "rvprop": "ids|timestamp"})
            r0 = old["query"]["pages"][0].get("revisions", [])
            r1 = new["query"]["pages"][0].get("revisions", [])
            if not r0 or not r1:
                path.write_text(json.dumps({**p, "skip": "no revision at T0"}))
                continue
            ids = [r0[0]["revid"], r1[0]["revid"]]
            c = W.get({"action": "query", "prop": "revisions", "revids": "|".join(map(str, ids)),
                       "rvprop": "ids|content", "rvslots": "main", "rvsection": "0"})
            text = {x["revid"]: x.get("slots", {}).get("main", {}).get("content")
                    for pg in c["query"]["pages"] for x in pg.get("revisions", [])}
            path.write_text(json.dumps({**p, "ts0": r0[0]["timestamp"], "ts1": r1[0]["timestamp"],
                                        "text0": text.get(ids[0]), "text1": text.get(ids[1])}))
        except RuntimeError:
            continue                       # rerun `fetch` to retry; pages are cached one by one
        if k % 50 == 0:
            print(f"{k}/{len(pages)}  {time.time() - t0:.0f}s", flush=True)
    print("fetch done", flush=True)


# ------------------------------------------------------------------ build
LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")


def render_value(v: str) -> str:
    """Infobox value as a reader would see it: refs/comments/links/templates flattened."""
    v = W.strip_noise(v)
    for _ in range(4):                     # flatten nested templates to their arguments
        v = re.sub(r"\{\{([^{}]*)\}\}", lambda m: " ".join(
            a.split("=", 1)[-1] for a in m.group(1).split("|")[1:]), v)
    v = LINK.sub(r"\1", v)
    v = re.sub(r"<[^>]+>|'''|''", "", v)
    return re.sub(r"\s+", " ", v).strip(" ,;")


def fields(wikitext: str) -> dict[str, str]:
    t = W.strip_noise(wikitext)
    for a, b in W.templates(t):
        if re.match(r"\{\{\s*Infobox", t[a:b], re.I):
            body, parts, depth, cur = t[a + 2:b - 2], [], 0, []
            for i, ch in enumerate(body):
                if body.startswith("{{", i) or body.startswith("[[", i):
                    depth += 1
                elif body.startswith("}}", i) or body.startswith("]]", i):
                    depth -= 1
                if ch == "|" and depth == 0:
                    parts.append("".join(cur)); cur = []
                else:
                    cur.append(ch)
            parts.append("".join(cur))
            out = {}
            for p in parts[1:]:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                k = k.strip().lower()
                if not re.fullmatch(r"[a-z_ ]{1,40}\d*", k) or W.LAYOUT.search(k):
                    continue
                v = render_value(v)
                if v:
                    out[k] = v
            return out
    return {}


def doc_text(title: str, f: dict[str, str], wikitext: str) -> str:
    lines = [f"{k.replace('_', ' ')}: {v}." for k, v in f.items()]
    return f"{title}. " + " ".join(lines) + " " + W.prose(wikitext)


def build() -> list[dict]:
    docs, qs = [], []
    for path in sorted(RAW.glob("*.json")):
        if path.name == "pages.json":
            continue
        p = json.loads(path.read_text())
        if p.get("skip") or not p.get("text0") or not p.get("text1"):
            continue
        f0, f1 = fields(p["text0"]), fields(p["text1"])
        d = len(docs)
        docs.append({"pageid": p["pageid"], "title": p["title"], "cat": p["cat"],
                     "rev0": p["ts0"][:10], "rev1": p["ts1"][:10],
                     "text0": doc_text(p["title"], f0, p["text0"]),
                     "text1": doc_text(p["title"], f1, p["text1"])})
        # sorted: set order follows PYTHONHASHSEED, so unsorted keys gave each process a
        # different question order and hence a different seeded sample (fixed 2026-10-13)
        for k in sorted(f0.keys() & f1.keys()):
            if SKIP_KEY.search(k):
                continue
            # years ("US$2.53 billion (2025)") are as-of annotations, not the fact; counting them
            # made year-only changes into questions and faked "answer in context" (fixed
            # 2026-10-11 after the first run failed adequacy; that run is kept as *_VOID_yearbug.log)
            v0 = {facts.norm(x): x for x in facts.NUM.findall(f0[k]) if len(x) >= facts.MIN_ANSWER and not YEAR.fullmatch(x)}
            v1 = {facts.norm(x): x for x in facts.NUM.findall(f1[k]) if len(x) >= facts.MIN_ANSWER and not YEAR.fullmatch(x)}
            new, old = set(v1) - set(v0), set(v0) - set(v1)
            if not new or not old or len(v0) > 2 or len(v1) > 2:
                continue
            stmt = f"The {k.replace('_', ' ')} of {p['title']} is ___."
            qs.append({"doc": d, "key": k, "statement": stmt, "question": stmt.replace(" ___", ""),
                       "new": sorted(v1[x] for x in new), "old": sorted(v0[x] for x in old)})
    return docs, qs


# ------------------------------------------------------------------ run
SYSTEM = "You answer strictly from the provided context and never from memory."
TEMPLATE = (                     # p7's template, unchanged
    "Context:\n{context}\n\n"
    "Fill in each blank (___) in the statement below with the exact number taken from "
    "the context. Reply with only the number(s), separated by commas. If the context "
    "does not contain it, reply: unknown\n\nStatement: {statement}"
)
GEN_OUT = data.CACHE / "p9_generation"


def value_in(vals, out: str) -> bool:
    """Any of `vals` in `out` at digit boundaries.  facts.NUM reads "36.77,29.86" (two values
    joined by the comma the prompt asks for) as one number; this does not (added 2026-10-13)."""
    for v in vals:
        for f in {v, v.replace(",", "")}:
            if re.search(r"(?<![\d.])" + re.escape(f) + r"(?![\d])", out):
                return True
    return False


def commit(new, old, out: str) -> str:
    """Which value the answer commits to: 'new', 'old', 'both' or 'neither'."""
    return {(1, 0): "new", (0, 1): "old", (1, 1): "both", (0, 0): "neither"}[
        (value_in(new, out), value_in(old, out))]


def generate(model: str, prompt: str, _cache: dict = {}) -> str:
    import hashlib
    import urllib.request
    GEN_OUT.mkdir(parents=True, exist_ok=True)
    store = GEN_OUT / f"{model.replace(':', '_')}.jsonl"
    if model not in _cache:
        _cache[model] = {}
        if store.exists():
            for line in open(store):
                r = json.loads(line)
                _cache[model][r["key"]] = r["output"]
    key = hashlib.sha256(f"{model}|{prompt}".encode()).hexdigest()
    if key not in _cache[model]:
        body = {"model": model, "stream": False, "options": {"temperature": 0, "seed": 0, "num_predict": 24},
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]}
        req = urllib.request.Request("http://localhost:11434/api/chat", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            _cache[model][key] = json.loads(r.read())["message"]["content"].strip()
        with open(store, "a") as f:
            f.write(json.dumps({"key": key, "output": _cache[model][key]}) + "\n")
    return _cache[model][key]


def run() -> None:
    import numpy as np
    from scipy.stats import binomtest

    import p5_chunked as P
    from research.staleness import answer_level, generation

    docs, qs = build()
    qs = random.Random(0).sample(qs, min(SAMPLE, len(qs)))
    cl = P.classify(docs)
    uniq = sorted({c for c0, c1, _ in cl for c in c0 + c1})
    idx = {c: i for i, c in enumerate(uniq)}
    recs = []
    for retr in ("minilm", "bge"):
        E = data.cached(retr, "wiki_infobox", f"chunks{P.CHUNK}", lambda: uniq)
        E = E / np.linalg.norm(E, axis=1, keepdims=True)
        Q = data.cached(retr, "wiki_infobox", "q_p9", lambda: [q["question"] for q in qs])
        for pol in ("replace", "append"):
            items = []
            for d, (c0, c1, lab) in enumerate(cl):
                s1 = set(c1)
                if pol == "append":
                    items += [(c, d, "both" if c in s1 else "old") for c in c0]
                    items += [(c, d, "new") for c, (k, _, _) in zip(c1, lab) if k != "identical"]
                else:
                    items += [(c, d, "new") for c in c1]
            top = np.argsort(-(Q @ E[[idx[c] for c, _, _ in items]].T), axis=1)[:, :3]
            for n, q in enumerate(qs):
                texts = [items[t][0] for t in top[n]]
                vals = set().union(*(facts.values(t) for t in texts))
                has_new = bool(vals & {facts.norm(a) for a in q["new"]})
                has_old = bool(vals & {facts.norm(a) for a in q["old"]})
                if pol == "replace" and not has_new:
                    continue                                   # adequacy uses new-in-context only
                if pol == "append" and not (has_new and has_old):
                    continue                                   # primary: both values in context
                for variant in (("none", "dated") if pol == "append" else ("none",)):
                    def lab_(t):
                        doc, v = docs[items[t][1]], items[t][2]
                        when = {"old": doc["rev0"], "new": doc["rev1"], "both": f"{doc['rev0']} and {doc['rev1']}"}[v]
                        return f"({doc['title']}, as of {when}) " if variant == "dated" else ""
                    ctx = "\n\n".join(f"[{j+1}] {lab_(t)}{items[t][0]}" for j, t in enumerate(top[n]))
                    for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
                        out = generate(gen, TEMPLATE.format(context=ctx, statement=q["statement"]))
                        recs.append({"retriever": retr, "policy": pol, "variant": variant, "gen": gen, "q": n,
                                     "class": generation.classify(out, q["new"], q["old"]), "output": out})
        print(f"[{retr}] done, {len(recs)} records", flush=True)
    (GEN_OUT / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    print(f"\n{len(qs)} sampled questions")
    for gen in ("phi3:mini", "qwen2.5:7b-instruct"):
        ad = [r for r in recs if r["gen"] == gen and r["policy"] == "replace"]
        k = sum(r["class"] == "correct" for r in ad)
        print(f"{gen:<20} ADEQUACY replace, new in context: correct {k}/{len(ad)} = {k/len(ad):.2f} "
              f"{np.round(answer_level.wilson(k, len(ad)), 3)}  -> {'ok' if k/len(ad) >= 0.60 else 'VOID'}")
        base = {(r["retriever"], r["q"]): r["class"] for r in recs
                if r["gen"] == gen and r["policy"] == "append" and r["variant"] == "none"}
        for variant in ("none", "dated"):
            rows = [r for r in recs if r["gen"] == gen and r["policy"] == "append" and r["variant"] == variant]
            k = sum(r["class"] == "stale" for r in rows)
            line = (f"{gen:<20} append k=3 both-in-context, labels={variant:<5}: stale {k}/{len(rows)} = {k/len(rows):.2f} "
                    f"{np.round(answer_level.wilson(k, len(rows)), 3)}  correct {np.mean([r['class'] == 'correct' for r in rows]):.2f}")
            if variant == "none":
                line += "   -> " + ("REPLICATES" if k / len(rows) >= 0.10 else "does NOT replicate (< 0.10)")
            else:
                b = sum(base[(r["retriever"], r["q"])] == "stale" and r["class"] != "stale" for r in rows)
                c = sum(base[(r["retriever"], r["q"])] != "stale" and r["class"] == "stale" for r in rows)
                line += f"   McNemar vs none: fixed {b}, broke {c}, p={binomtest(b, b + c, 0.5).pvalue if b + c else 1:.3g}"
            print(line, flush=True)


if __name__ == "__main__":
    if STAGE == "fetch":
        fetch()
    elif STAGE == "run":
        run()
    elif STAGE == "build":
        docs, qs = build()
        rev = sum(d["text0"] != d["text1"] for d in docs)
        print(f"{len(docs)} pages with both snapshots; {rev} changed; {len(qs)} changed-field questions "
              f"from {len({q['doc'] for q in qs})} pages")
        import collections
        print(collections.Counter(q["key"] for q in qs).most_common(20))
        for q in random.Random(1).sample(qs, min(12, len(qs))):
            print(" ", q["statement"], "| old", q["old"], "-> new", q["new"])
