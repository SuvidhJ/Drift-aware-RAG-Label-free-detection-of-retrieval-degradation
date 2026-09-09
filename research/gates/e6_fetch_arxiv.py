"""Collect real arXiv abstract revisions (v1 vs latest) across topical categories.

Why arXiv rather than Wikipedia: a corpus of paper abstracts is a canonical RAG use
case, arXiv categories give strong topical structure, and the API serves
version-specific abstracts via id_list in batches of 100 - so the whole corpus costs
~30 requests instead of ~1000.

Politeness: strictly serialized, 3.0s between requests, descriptive User-Agent,
resumable checkpoint.  arXiv asks for 3s; we honour it.
"""
import json, os, re, sys, time, urllib.parse, urllib.request, urllib.error

API = "http://export.arxiv.org/api/query"
UA = {"User-Agent": "drift-aware-rag-research/1.0 (academic study of embedding "
                    "staleness in retrieval indexes; serialized, 1 req / 3s)"}
DELAY = 3.0
CKPT = "ax_ckpt.json"

CATS = [
    "cs.CL", "cs.CV", "cs.LG", "cs.CR", "cs.DB",
    "math.NT", "math.AG", "math.PR",
    "cond-mat.stat-mech", "astro-ph.GA", "q-bio.NC", "stat.ME", "econ.EM", "eess.SP",
]
PER_CAT = 130
WINDOW = "submittedDate:[202301010000 TO 202306302359]"

_last = [0.0]


def get(url, tries=4):
    for a in range(tries):
        dt = time.time() - _last[0]
        if dt < DELAY:
            time.sleep(DELAY - dt)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = int(e.headers.get("Retry-After", "60") or 60)
                print("    [429] sleeping %ds" % (ra + 5), flush=True)
                time.sleep(ra + 5)
                continue
            if a == tries - 1:
                return None
            time.sleep(5 * (a + 1))
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(5 * (a + 1))
    return None


ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)


def parse_entries(xml):
    out = []
    for e in ENTRY.findall(xml or ""):
        i = re.search(r"<id>http://arxiv\.org/abs/([^<]+)</id>", e)
        s = re.search(r"<summary>(.*?)</summary>", e, re.S)
        t = re.search(r"<title>(.*?)</title>", e, re.S)
        p = re.search(r"<published>([^<]+)</published>", e)
        u = re.search(r"<updated>([^<]+)</updated>", e)
        if not (i and s):
            continue
        out.append({"vid": i.group(1),
                    "abstract": " ".join(s.group(1).split()),
                    "title": " ".join(t.group(1).split()) if t else "",
                    "published": p.group(1) if p else "",
                    "updated": u.group(1) if u else ""})
    return out


def list_category(cat, want):
    """Latest version of each paper in a category (oldest-first for a stable sample)."""
    got, start = [], 0
    while len(got) < want:
        url = (API + "?" + urllib.parse.urlencode({
            "search_query": "cat:%s AND %s" % (cat, WINDOW), "start": start, "max_results": 100,
            "sortBy": "submittedDate", "sortOrder": "descending"}))
        xml = get(url)
        ent = parse_entries(xml)
        if not ent:
            break
        got += ent
        start += 100
        print("    %-20s %d" % (cat, len(got)), flush=True)
    return got[:want]


def v1_abstracts(base_ids):
    """Batch: 100 versioned ids per request -> {base_id: v1 abstract}"""
    out = {}
    for i in range(0, len(base_ids), 100):
        chunk = base_ids[i:i + 100]
        url = API + "?" + urllib.parse.urlencode({
            "id_list": ",".join(b + "v1" for b in chunk), "max_results": 100})
        for e in parse_entries(get(url)):
            out[re.sub(r"v\d+$", "", e["vid"])] = e["abstract"]
        print("    v1 %d/%d" % (min(i + 100, len(base_ids)), len(base_ids)), flush=True)
    return out


if __name__ == "__main__":
    ck = json.load(open(CKPT, encoding="utf-8")) if os.path.exists(CKPT) else {}

    ck.setdefault("cats", {})
    print("[1/3] listing categories...", flush=True)
    for c in CATS:
        if c in ck["cats"]:
            continue
        ck["cats"][c] = list_category(c, PER_CAT)
        json.dump(ck, open(CKPT, "w", encoding="utf-8"))
        print("    %-20s done (%d)" % (c, len(ck["cats"][c])), flush=True)

    # keep only papers that actually have a later version
    allp, multi = [], []
    for c, ents in ck["cats"].items():
        for e in ents:
            mm = re.search(r"v(\d+)$", e["vid"])
            nv = int(mm.group(1)) if mm else 1
            rec = dict(e, cat=c, base=re.sub(r"v\d+$", "", e["vid"]), nver=nv)
            allp.append(rec)
            if nv >= 2:
                multi.append(rec)
    print("    total papers: %d, with >=2 versions: %d" % (len(allp), len(multi)), flush=True)

    if "v1" not in ck:
        print("[2/3] fetching v1 abstracts (batched 100/request)...", flush=True)
        ck["v1"] = v1_abstracts([m["base"] for m in multi])
        json.dump(ck, open(CKPT, "w", encoding="utf-8"))
    v1 = ck["v1"]

    print("[3/3] assembling...", flush=True)
    out, ident = [], 0
    for m in allp:
        # single-version papers were never revised: v1 IS the current abstract, m = 0
        a0 = v1.get(m["base"]) if m["nver"] >= 2 else m["abstract"]
        a1 = m["abstract"]
        if not a0 or len(a0) < 250 or len(a1) < 250:
            continue
        if a0 == a1:
            ident += 1
        out.append({"hub": m["cat"], "title": m["title"], "pageid": m["base"],
                    "rev0": m["base"] + "v1", "ts0": m["published"],
                    "rev1": m["vid"], "ts1": m["updated"], "nver": m["nver"],
                    "text0": a0[:4000], "text1": a1[:4000]})
    json.dump(out, open("ax_corpus.json", "w", encoding="utf-8"))
    from collections import Counter
    print("DONE n=%d  identical_abstract=%d (%.1f%%)"
          % (len(out), ident, 100.0 * ident / max(len(out), 1)))
    print("  per category:", dict(Counter(d["hub"] for d in out)))
