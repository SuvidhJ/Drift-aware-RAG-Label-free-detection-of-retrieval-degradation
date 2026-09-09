"""Collect real Wikipedia lead sections at two timestamps - polite, serialized, resumable.

v2 after a 429: single-threaded, adaptive pacing, honours Retry-After, and batches
content fetches 50 revids/titles per call so the whole corpus costs ~950 requests.
"""
import json, os, re, sys, time, urllib.parse, urllib.request, urllib.error, random

API = "https://en.wikipedia.org/w/api.php"
UA = ("drift-aware-rag-research/2.0 (academic study of embedding staleness; "
      "single-threaded, <=1.2 req/s)")
T0 = "2023-01-01T00:00:00Z"
CKPT = "wp_ckpt.json"

HUBS = [
    ("ai", "Artificial intelligence"), ("ml", "Machine learning"),
    ("crypto", "Cryptocurrency"), ("covid", "COVID-19 pandemic"),
    ("climate", "Climate change"), ("elections", "Elections in India"),
    ("spaceflight", "Spaceflight"), ("smartphone", "Smartphone"),
    ("cellbio", "Cell biology"), ("orgchem", "Organic chemistry"),
    ("football", "Association football"), ("econ", "Macroeconomics"),
    ("geometry", "Euclidean geometry"), ("rome", "Ancient Rome"),
    ("mechanics", "Classical mechanics"), ("baroque", "Baroque music"),
    ("mythology", "Greek mythology"), ("mineral", "Mineralogy"),
    ("latin", "Latin"), ("numbertheory", "Number theory"),
]
BAD = re.compile(r"^(List of|Index of|Outline of|Timeline of|Glossary of|Comparison of|"
                 r"Category:|Template:|Portal:|Wikipedia:|Help:|File:)", re.I)


class Pacer:
    def __init__(self, delay=2.0):
        self.delay, self.last = delay, 0.0

    def wait(self):
        dt = time.time() - self.last
        if dt < self.delay:
            time.sleep(self.delay - dt)
        self.last = time.time()

    def penalise(self):
        self.delay = min(self.delay * 1.5, 8.0)

    def relax(self):
        self.delay = max(self.delay * 0.98, 2.0)


P = Pacer()


def api(params, tries=6):
    params = dict(params)
    params["format"] = "json"
    params["maxlag"] = 5
    url = API + "?" + urllib.parse.urlencode(params)
    for a in range(tries):
        P.wait()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode("utf-8"))
            if "error" in d and d["error"].get("code") == "maxlag":
                time.sleep(6)
                P.penalise()
                continue
            P.relax()
            return d
        except urllib.error.HTTPError as e:
            if e.code == 429:
                ra = int(e.headers.get("Retry-After", "60") or 60)
                waited = 0
                while waited < 1800:
                    time.sleep(ra + 5)
                    waited += ra + 5
                    P.penalise()
                    try:
                        req2 = urllib.request.Request(
                            "https://en.wikipedia.org/w/api.php?action=query&format=json&meta=siteinfo",
                            headers={"User-Agent": UA})
                        urllib.request.urlopen(req2, timeout=30).read()
                        print("    [429] cleared after %ds" % waited, flush=True)
                        break
                    except urllib.error.HTTPError as e2:
                        if e2.code != 429:
                            break
                        ra = int(e2.headers.get("Retry-After", "60") or 60)
                    except Exception:
                        break
                continue
            if a == tries - 1:
                return None
            time.sleep(2 ** a)
        except Exception:
            if a == tries - 1:
                return None
            time.sleep(2 ** a)
    return None


def links_from(hub):
    out, cont = [], {}
    while len(out) < 500:
        d = api({"action": "query", "prop": "links", "titles": hub,
                 "plnamespace": 0, "pllimit": 500, **cont})
        if not d or "query" not in d:
            break
        for p in d["query"]["pages"].values():
            out += [l["title"] for l in p.get("links", [])]
        if "continue" in d:
            cont = d["continue"]
        else:
            break
    return out


def revid_at(title, ts):
    d = api({"action": "query", "prop": "revisions", "titles": title,
             "rvprop": "ids|timestamp", "rvstart": ts, "rvdir": "older", "rvlimit": 1})
    if not d or "query" not in d:
        return None
    for p in d["query"]["pages"].values():
        if "missing" in p or "invalid" in p:
            return None
        rv = p.get("revisions") or []
        if rv:
            return {"pageid": p["pageid"], "title": p["title"],
                    "revid": rv[0]["revid"], "ts": rv[0]["timestamp"]}
    return None


def content_by_revids(revids):
    """Batch 50 revids per call -> {revid: wikitext}"""
    out = {}
    for i in range(0, len(revids), 50):
        chunk = revids[i:i + 50]
        d = api({"action": "query", "prop": "revisions",
                 "revids": "|".join(map(str, chunk)),
                 "rvprop": "content|ids", "rvslots": "main"})
        if d and "query" in d:
            for p in d["query"]["pages"].values():
                for rv in p.get("revisions", []):
                    try:
                        out[rv["revid"]] = rv["slots"]["main"]["*"]
                    except KeyError:
                        pass
        print("    old-content %d/%d" % (min(i + 50, len(revids)), len(revids)), flush=True)
    return out


def content_by_titles(titles):
    """Batch 50 titles per call -> {title: (revid, ts, wikitext)}"""
    out = {}
    for i in range(0, len(titles), 50):
        chunk = titles[i:i + 50]
        d = api({"action": "query", "prop": "revisions", "titles": "|".join(chunk),
                 "rvprop": "content|ids|timestamp", "rvslots": "main"})
        if d and "query" in d:
            norm = {n["to"]: n["from"] for n in d["query"].get("normalized", [])}
            for p in d["query"]["pages"].values():
                for rv in p.get("revisions", []):
                    try:
                        key = norm.get(p["title"], p["title"])
                        out[key] = (rv["revid"], rv["timestamp"], rv["slots"]["main"]["*"])
                    except KeyError:
                        pass
        print("    cur-content %d/%d" % (min(i + 50, len(titles)), len(titles)), flush=True)
    return out


def strip_pairs(s, op, cl):
    out, depth, i = [], 0, 0
    while i < len(s):
        if s.startswith(op, i):
            depth += 1
            i += len(op)
            continue
        if s.startswith(cl, i) and depth:
            depth -= 1
            i += len(cl)
            continue
        if not depth:
            out.append(s[i])
        i += 1
    return "".join(out)


def lead_of(wt):
    """First section only: everything before the first '==' heading."""
    if not wt:
        return ""
    return re.split(r"\n==[^=]", wt, maxsplit=1)[0]


APOS3 = chr(39) * 3
APOS2 = chr(39) * 2


def clean(wt):
    if not wt:
        return ""
    s = re.sub(r"<!--.*?-->", " ", wt, flags=re.S)
    s = re.sub(r"<ref[^>]*/>", " ", s)
    s = re.sub(r"<ref.*?</ref>", " ", s, flags=re.S)
    s = re.sub(r"<[^>]+>", " ", s)
    s = strip_pairs(s, "{{", "}}")
    s = strip_pairs(s, "{|", "|}")
    s = re.sub(r"\[\[(?:File|Image|Category):[^\]]*\]\]", " ", s, flags=re.I)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[\[([^\]]*)\]\]", r"\1", s)
    s = re.sub(r"\[https?://\S+\s([^\]]*)\]", r"\1", s)
    s = re.sub(r"https?://\S+", " ", s)
    s = s.replace(APOS3, "").replace(APOS2, "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def wait_until_clear(max_wait=7200, step=120):
    """Block until the API stops returning 429. Long blocks outlast Retry-After."""
    waited = 0
    while waited <= max_wait:
        try:
            req = urllib.request.Request(
                API + "?action=query&format=json&meta=siteinfo",
                headers={"User-Agent": UA})
            urllib.request.urlopen(req, timeout=30).read()
            print("    API clear after %ds of waiting" % waited, flush=True)
            return True
        except urllib.error.HTTPError as e:
            if e.code != 429:
                return True
        except Exception:
            pass
        print("    still rate-limited, waited %ds" % waited, flush=True)
        time.sleep(step)
        waited += step
    return False


if __name__ == "__main__":
    per_hub = int(sys.argv[1]) if len(sys.argv) > 1 else 55
    random.seed(20260906)
    print("[0/4] waiting for API rate limit to clear...", flush=True)
    if not wait_until_clear():
        print("GIVING UP: still rate-limited after 2h"); sys.exit(2)
    ck = json.load(open(CKPT, encoding="utf-8")) if os.path.exists(CKPT) else {}

    if "cand" not in ck:
        print("[1/4] hub links (serialized)...", flush=True)
        cand = ck.get("cand_partial", [])
        done = set(ck.get("hubs_done", []))
        seen = set(c["title"] for c in cand)
        for lab, hub in HUBS:
            if lab in done:
                continue
            ls = [t for t in links_from(hub) if not BAD.match(t) and t not in seen]
            random.shuffle(ls)
            take = ls[:per_hub]
            seen.update(take)
            cand += [{"hub": lab, "title": t} for t in take]
            done.add(lab)
            print("    %-14s -> %3d   (total %d)" % (lab, len(take), len(cand)), flush=True)
            ck["cand_partial"], ck["hubs_done"] = cand, sorted(done)
            json.dump(ck, open(CKPT, "w", encoding="utf-8"))
        ck["cand"] = cand
        ck.pop("cand_partial", None)
        json.dump(ck, open(CKPT, "w", encoding="utf-8"))
    cand = ck["cand"]
    print("    candidates: %d" % len(cand), flush=True)

    if "r0" not in ck:
        print("[2/4] revid at %s (%d calls)..." % (T0, len(cand)), flush=True)
        r0 = ck.get("r0_partial", [])
        for i in range(len(r0), len(cand)):
            r0.append(revid_at(cand[i]["title"], T0))
            if (i + 1) % 10 == 0:
                print("    %d/%d  ok=%d  delay=%.2fs"
                      % (i + 1, len(cand), sum(1 for x in r0 if x), P.delay), flush=True)
                ck["r0_partial"] = r0
                json.dump(ck, open(CKPT, "w", encoding="utf-8"))
        ck["r0"] = r0
        ck.pop("r0_partial", None)
        json.dump(ck, open(CKPT, "w", encoding="utf-8"))
    r0 = ck["r0"]
    pairs = [(c, a) for c, a in zip(cand, r0) if a]
    print("    existed at T0: %d" % len(pairs), flush=True)

    print("[3/4] batch content fetch...", flush=True)
    old = content_by_revids([a["revid"] for _, a in pairs])
    cur = content_by_titles([a["title"] for _, a in pairs])

    print("[4/4] clean + save...", flush=True)
    out = []
    for c, a in pairs:
        w0 = old.get(a["revid"])
        got = cur.get(a["title"])
        if not w0 or not got:
            continue
        rev1, ts1, w1 = got
        if rev1 == a["revid"]:
            continue
        t0, t1 = clean(lead_of(w0)), clean(lead_of(w1))
        if len(t0) < 250 or len(t1) < 250:
            continue
        out.append({"hub": c["hub"], "title": a["title"], "pageid": a["pageid"],
                    "rev0": a["revid"], "ts0": a["ts"], "rev1": rev1, "ts1": ts1,
                    "text0": t0[:4000], "text1": t1[:4000]})
    json.dump(out, open("wp_corpus.json", "w", encoding="utf-8"))
    ident = sum(1 for d in out if d["text0"] == d["text1"])
    from collections import Counter
    print("DONE n=%d  identical_lead=%d (%.1f%%)"
          % (len(out), ident, 100.0 * ident / max(len(out), 1)))
    print("  per hub:", dict(Counter(d["hub"] for d in out)))
