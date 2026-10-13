"""Pre-test E0: do per-page FACT-change rates diverge from PAGE-change rates?

Go/no-go for G6 (harm-aware index maintenance).  Pre-registered before any data was
fetched (2026-10-09):

  K1 divergence   Spearman rho(fact-change rate, page-change rate) over pages.
                  rho >= 0.8 -> the two rates do not diverge -> no-go.
  K2 scheduling   rank pages on year-1 rates, refresh the top 20%, count the share of
                  year-2 fact changes caught.  Fact-rate ranking must beat page-rate
                  ranking with a paired bootstrap 95% CI over pages excluding 0;
                  otherwise no-go.  Primary fact ranking: year-1 fact count, ties
                  (many pages have none) broken by year-1 page count -- what a
                  harm-aware scheduler would do.  Pure fact ranking is also printed.
  control         predicting year-2 PAGE changes, the fact-rate ranking must not beat
                  the page-rate ranking.

Sample: the 344 pages of wp_corpus.json (chosen before this test, not hand-picked).
Observation: a crawler's view -- the lead section (rvsection=0: lead + infobox) as it
stood at every Monday 00:00 UTC, 2024-09-30 .. 2026-09-28 (104 weekly intervals,
52 per year).  Page change = the canonical lead text differs between consecutive
snapshots, the strongest page-change baseline available here, since it watches the
same text the facts come from.  Fact change = facts.aligned_edits (the paper's
definition) on the cleaned prose, OR a numeric infobox field whose value set changes.

    .venv/bin/python research/staleness/pretests/e0_fact_rates.py fetch
    .venv/bin/python research/staleness/pretests/e0_fact_rates.py analyse
"""
import datetime as dt
import gzip
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from drift.freshness import canonical  # noqa: E402
from research.staleness import data, facts  # noqa: E402

API = "https://en.wikipedia.org/w/api.php"
UA = "drift-rag-staleness-pretest/0.1 (academic research; github.com/SuvidhJ)"
RAW = data.CACHE / "e0_wiki"
START = dt.datetime(2024, 9, 30, tzinfo=dt.timezone.utc)      # a Monday
WEEKS = 104
SNAPS = [START + dt.timedelta(weeks=w) for w in range(WEEKS + 1)]
BUDGET = 0.20


def get(params: dict) -> dict:
    params = {**params, "format": "json", "formatversion": "2", "maxlag": "5"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params),
                                 headers={"User-Agent": UA, "Accept-Encoding": "gzip"})
    for attempt in range(6):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                body = r.read()
                if r.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
            out = json.loads(body)
            if out.get("error", {}).get("code") == "maxlag":
                time.sleep(5 * (attempt + 1))
                continue
            return out
        except Exception:
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"failed: {params}")


def iso(t: dt.datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch_page(pageid: int) -> dict:
    path = RAW / f"{pageid}.json"
    if path.exists():
        return json.loads(path.read_text())
    # every revision in the window (metadata only), oldest first
    revs, cont = [], {}
    while True:
        r = get({"action": "query", "prop": "revisions", "pageids": pageid,
                 "rvprop": "ids|timestamp", "rvlimit": "max", "rvdir": "newer",
                 "rvstart": iso(SNAPS[0]), "rvend": iso(SNAPS[-1]), **cont})
        revs += r["query"]["pages"][0].get("revisions", [])
        if "continue" not in r:
            break
        cont = r["continue"]
    # the revision live at the first snapshot
    r = get({"action": "query", "prop": "revisions", "pageids": pageid,
             "rvprop": "ids|timestamp", "rvlimit": 1, "rvdir": "older", "rvstart": iso(SNAPS[0])})
    page = r["query"]["pages"][0]
    prior = page.get("revisions", [])
    allrevs = prior + revs
    ts = [dt.datetime.strptime(x["timestamp"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
          for x in allrevs]
    snap_rev = []
    for s in SNAPS:
        live = [x["revid"] for x, t in zip(allrevs, ts) if t <= s]
        snap_rev.append(live[-1] if live else None)
    want = sorted({x for x in snap_rev if x is not None})
    text = {}
    for b in range(0, len(want), 50):
        r = get({"action": "query", "prop": "revisions", "revids": "|".join(map(str, want[b:b + 50])),
                 "rvprop": "ids|content", "rvslots": "main", "rvsection": "0"})
        for p in r["query"]["pages"]:
            for x in p.get("revisions", []):
                text[x["revid"]] = x.get("slots", {}).get("main", {}).get("content")
    out = {"pageid": pageid, "title": page.get("title"), "n_revisions_window": len(revs),
           "snap_rev": snap_rev, "text": {str(k): v for k, v in text.items()}}
    RAW.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------------ wikitext
COMMENT = re.compile(r"<!--.*?-->", re.S)
REF = re.compile(r"<ref[^>/]*/>|<ref[^>]*>.*?</ref>", re.S | re.I)
TAG = re.compile(r"<[^>]+>")
URL = re.compile(r"https?://\S+")
LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
FILELINK = re.compile(r"\[\[(?:File|Image):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]", re.I)
EXTLINK = re.compile(r"\[\S+ ([^\]]*)\]")
LAYOUT = re.compile(r"image|logo|caption|alt|size|width|upright|map|coord|pushpin|module|signature|seal|flag", re.I)


def strip_noise(t: str) -> str:
    return URL.sub("", REF.sub("", COMMENT.sub("", t)))


def templates(t: str) -> list[tuple[int, int]]:
    """Spans of top-level {{...}} blocks (balanced)."""
    spans, depth, start, i = [], 0, 0, 0
    while i < len(t) - 1:
        if t[i:i + 2] == "{{":
            if depth == 0:
                start = i
            depth += 1
            i += 2
        elif t[i:i + 2] == "}}" and depth:
            depth -= 1
            i += 2
            if depth == 0:
                spans.append((start, i))
        else:
            i += 1
    return spans


def infobox(t: str) -> dict[str, frozenset]:
    """Numeric infobox fields: key -> set of distinctive values (by facts.norm)."""
    t = strip_noise(t)
    for a, b in templates(t):
        if re.match(r"\{\{\s*Infobox", t[a:b], re.I):
            body = t[a + 2:b - 2]
            out = {}
            # split on '|' at nesting depth 0
            parts, depth, cur = [], 0, []
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
            for p in parts[1:]:
                if "=" not in p:
                    continue
                k, v = p.split("=", 1)
                k = k.strip().lower()
                # a key that is not an identifier is a mis-split, not a field
                if LAYOUT.search(k) or not re.fullmatch(r"[a-z_ ]{1,40}?\d*", k):
                    continue
                # numbered fields (established_date1..10) are one list: inserting an
                # entry renumbers the rest, which is not a change of any fact
                k = re.sub(r"\d+$", "", k)
                vals = frozenset(facts.norm(x) for x in facts.NUM.findall(v) if len(x) >= facts.MIN_ANSWER)
                if vals:
                    out[k] = out.get(k, frozenset()) | vals
            return out
    return {}


def prose(t: str) -> str:
    t = strip_noise(t)
    t = FILELINK.sub("", t)
    for a, b in reversed(templates(t)):
        t = t[:a] + t[b:]
    t = LINK.sub(r"\1", t)
    t = EXTLINK.sub(r"\1", t)
    t = TAG.sub("", t).replace("'''", "").replace("''", "")
    t = re.sub(r"^[=*#:;|!{].*$", "", t, flags=re.M)      # headings, lists, table rows
    return re.sub(r"\s+", " ", t).strip()


def events(page: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Per weekly interval: page changed, fact changed (prose OR infobox), prose-fact, infobox-fact."""
    texts = [page["text"].get(str(r)) if r is not None else None for r in page["snap_rev"]]
    pc, pf, ib = (np.zeros(WEEKS, bool) for _ in range(3))
    cache = {}
    for w in range(WEEKS):
        a, b = texts[w], texts[w + 1]
        if a is None or b is None or a == b:
            continue
        for x in (a, b):
            if x not in cache:
                cache[x] = (prose(x), infobox(x))
        (pa, ia), (pb, ibx) = cache[a], cache[b]
        if canonical(pa) == canonical(pb) and ia == ibx:
            continue
        pc[w] = True
        pf[w] = bool(facts.fact_questions([{"text0": pa, "text1": pb}])[0])
        ib[w] = any(k in ibx and ia[k] != ibx[k] and (ibx[k] - ia[k]) for k in ia)
    return pc, pf | ib, pf, ib


# ------------------------------------------------------------------ analysis
def cho_rate(x: np.ndarray, n: int) -> np.ndarray:
    """Cho & Garcia-Molina (2003) change-rate estimator from n periodic visits."""
    return -np.log((n - x + 0.5) / (n + 0.5))


def capture(score: np.ndarray, target: np.ndarray, rng, reps: int = 200) -> np.ndarray:
    """Per-page indicator of being refreshed (top BUDGET by score), averaged over random tie-breaks."""
    k = int(round(BUDGET * len(score)))
    sel = np.zeros(len(score))
    for _ in range(reps):
        order = np.lexsort((rng.random(len(score)), -score))
        sel[order[:k]] += 1
    return sel / reps


def analyse() -> None:
    rng = np.random.default_rng(0)
    pages = [json.loads(p.read_text()) for p in sorted(RAW.glob("*.json"))]
    rows = []
    for p in pages:
        if not p["text"]:
            continue
        pc, fc, pf, ib = events(p)
        rows.append((p["title"], pc, fc, pf, ib, p["n_revisions_window"]))
    n = len(rows)
    PC = np.array([r[1] for r in rows]); FC = np.array([r[2] for r in rows])
    PF = np.array([r[3] for r in rows]); IB = np.array([r[4] for r in rows])
    nrev = np.array([r[5] for r in rows])
    print(f"pages {n}; weekly intervals {WEEKS}; revisions in window median {np.median(nrev):.0f}")
    print(f"intervals with a page change {PC.sum()} ({PC.mean():.3f}); with a fact change {FC.sum()} "
          f"({FC.mean():.3f}): prose {PF.sum()}, infobox {IB.sum()}, both {(PF & IB).sum()}")
    print(f"fact changes / page changes overall {FC.sum() / max(PC.sum(), 1):.3f}")
    print(f"pages with >=1 page change {np.mean(PC.any(1)):.2f}; with >=1 fact change {np.mean(FC.any(1)):.2f}")

    criteria(PC, FC, "PRIMARY: fact = prose aligned edit OR infobox value change", rng)
    criteria(PC, IB, "SENSITIVITY: fact = infobox value change only", rng)


def criteria(PC: np.ndarray, FC: np.ndarray, label: str, rng) -> None:
    n, h = len(PC), WEEKS // 2
    print(f"\n===== {label}  ({int(FC.sum())} fact-change intervals)")
    # K1 -- divergence over the whole window, among pages that changed at all
    act = PC.any(1)
    rp, rf = cho_rate(PC.sum(1), WEEKS), cho_rate(FC.sum(1), WEEKS)
    rho = spearmanr(rp[act], rf[act]).statistic
    boots = []
    idx = np.flatnonzero(act)
    for _ in range(2000):
        j = rng.choice(idx, len(idx))
        boots.append(spearmanr(rp[j], rf[j]).statistic)
    lo, hi = np.nanpercentile(boots, [2.5, 97.5])
    share = FC.sum(1)[act] / PC.sum(1)[act]
    print(f"K1  Spearman rho(fact rate, page rate), {act.sum()} changing pages: {rho:.3f} [{lo:.3f}, {hi:.3f}]"
          f"   -> {'NO-GO' if rho >= 0.8 else 'pass'} (kill if >= 0.8)")
    print(f"    per-page fact/page share: median {np.median(share):.2f}, IQR {np.percentile(share, 25):.2f}-"
          f"{np.percentile(share, 75):.2f}, zero for {np.mean(share == 0):.2f}")

    # K2 -- year-1 ranking, year-2 fact changes caught at a 20% refresh budget
    p1, f1 = PC[:, :h].sum(1), FC[:, :h].sum(1)
    p2, f2 = PC[:, h:].sum(1), FC[:, h:].sum(1)
    sel_p, sel_f = capture(p1.astype(float), f2, rng), capture(f1.astype(float), f2, rng)
    sel_c = capture(f1 + 1e-3 * p1, f2, rng)     # fact rate, ties broken by page rate
    for lab, tgt, ctl in (("K2 ", f2, False), ("CTL", p2, True)):
        cp, cf, cc = (float((s * tgt).sum() / max(tgt.sum(), 1)) for s in (sel_p, sel_f, sel_c))
        d = []
        for _ in range(4000):
            j = rng.integers(0, n, n)
            t = tgt[j].sum()
            d.append(((sel_c[j] - sel_p[j]) * tgt[j]).sum() / t if t else 0.0)
        lo, hi = np.percentile(d, [2.5, 97.5])
        what = "year-2 PAGE changes (control)" if ctl else "year-2 FACT changes"
        print(f"{lab} share of {what} ({int(tgt.sum())}) caught at {BUDGET:.0%} budget:")
        print(f"    rank by year-1 page rate {cp:.3f} | fact rate {cf:.3f} | fact rate, page tie-break {cc:.3f}"
              f" | oracle {np.sort(tgt)[::-1][:int(round(BUDGET * n))].sum() / max(tgt.sum(), 1):.3f}")
        if ctl:
            verdict = "pass" if hi > 0 or cc <= cp else "FAIL (fact ranking wins where it should not)"
        else:
            verdict = "pass" if lo > 0 else "NO-GO (CI includes 0)"
        print(f"    (fact, page tie-break) - page: {cc - cp:+.3f} [{lo:+.3f}, {hi:+.3f}]   -> {verdict}")
    print(f"year-1 vs year-2 stability, Spearman: page {spearmanr(p1, p2).statistic:.3f}, "
          f"fact {spearmanr(f1, f2).statistic:.3f}")


if __name__ == "__main__":
    if sys.argv[1] == "fetch":
        import json as _j
        wp = _j.loads((data.RESULTS / data.CORPORA["wikipedia"]).read_text())
        ids = list(dict.fromkeys(d["pageid"] for d in wp))
        t0 = time.time()
        failed = []
        for k, pid in enumerate(ids):
            try:
                fetch_page(pid)
            except RuntimeError:
                failed.append(pid)        # rerun 'fetch' to retry; pages are cached one by one
            if k % 20 == 0:
                print(f"{k}/{len(ids)}  {time.time() - t0:.0f}s", flush=True)
        print(f"done; failed {len(failed)}: {failed}")
    else:
        analyse()
