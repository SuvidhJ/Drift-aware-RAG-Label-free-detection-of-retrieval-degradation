"""Collect a REAL documentation corpus with real revisions: Python docs 3.8 vs 3.13.

This is the case the selective-reindex claim actually targets - RAG over technical
documentation, where a whole module is rewritten at once, so revision should be far
more regionally concentrated than arXiv's independent per-paper peer-review edits.

Documents are SECTIONS (matched across versions by their stable HTML id), which is
also a realistic RAG chunking.  Static CDN pages: no API, no rate limit, no auth.
"""
import html as _html
import json, os, re, sys, time, urllib.request, urllib.error

OLD, NEW = "3.8", "3.13"
BASE = "https://docs.python.org/%s/"
UA = {"User-Agent": "drift-aware-rag-research/1.0 (academic study of embedding staleness)"}
CKPT = "pd_ckpt.json"
DELAY = 0.35

_last = [0.0]


def get(url, tries=3):
    for a in range(tries):
        dt = time.time() - _last[0]
        if dt < DELAY:
            time.sleep(DELAY - dt)
        _last[0] = time.time()
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            time.sleep(2 * (a + 1))
        except Exception:
            time.sleep(2 * (a + 1))
    return None


def page_list(ver):
    """Library + reference + howto pages, from the per-version index pages."""
    pages = set()
    for idx in ["library/index.html", "reference/index.html", "howto/index.html",
                "tutorial/index.html", "using/index.html", "c-api/index.html"]:
        h = get(BASE % ver + idx)
        if not h:
            continue
        sub = idx.rsplit("/", 1)[0]
        for m in re.finditer(r'href="([a-zA-Z0-9_.\-]+\.html)"', h):
            pages.add("%s/%s" % (sub, m.group(1)))
    return sorted(pages)


TAG = re.compile(r"<[^>]+>")
SECT = re.compile(r'<section id="([^"]+)"[^>]*>', re.I)
DIV = re.compile(r'<div class="section" id="([^"]+)"[^>]*>', re.I)


def strip(h):
    h = re.sub(r"<(script|style)\b.*?</\1>", " ", h, flags=re.S | re.I)
    h = re.sub(r'<a class="headerlink".*?</a>', " ", h, flags=re.S | re.I)
    t = TAG.sub(" ", h)
    t = _html.unescape(t)
    return re.sub(r"\s+", " ", t).strip()


def sections(page_html):
    """{section_id: text} - works for both the <section> (3.10+) and <div class=section> forms."""
    if not page_html:
        return {}
    marks = [(m.start(), m.group(1)) for m in SECT.finditer(page_html)]
    if not marks:
        marks = [(m.start(), m.group(1)) for m in DIV.finditer(page_html)]
    out = {}
    for i, (pos, sid) in enumerate(marks):
        end = marks[i + 1][0] if i + 1 < len(marks) else len(page_html)
        txt = strip(page_html[pos:end])
        if len(txt) >= 300:
            out[sid] = txt[:4000]
    return out


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 100000
    ck = json.load(open(CKPT, encoding="utf-8")) if os.path.exists(CKPT) else {}

    if "pages" not in ck:
        print("[1/3] listing pages...", flush=True)
        a, b = set(page_list(OLD)), set(page_list(NEW))
        ck["pages"] = sorted(a & b)          # only pages present in BOTH versions
        json.dump(ck, open(CKPT, "w", encoding="utf-8"))
    pages = ck["pages"][:limit]
    print("    pages in both %s and %s: %d" % (OLD, NEW, len(pages)), flush=True)

    ck.setdefault("docs", {})
    print("[2/3] fetching section text for each page (both versions)...", flush=True)
    for i, p in enumerate(pages):
        if p in ck["docs"]:
            continue
        s0 = sections(get(BASE % OLD + p))
        s1 = sections(get(BASE % NEW + p))
        common = sorted(set(s0) & set(s1))
        ck["docs"][p] = [{"sid": s, "t0": s0[s], "t1": s1[s]} for s in common]
        if (i + 1) % 10 == 0:
            json.dump(ck, open(CKPT, "w", encoding="utf-8"))
            tot = sum(len(v) for v in ck["docs"].values())
            print("    %d/%d pages, %d sections" % (i + 1, len(pages), tot), flush=True)
    json.dump(ck, open(CKPT, "w", encoding="utf-8"))

    print("[3/3] assembling...", flush=True)
    out, ident = [], 0
    for p, secs in ck["docs"].items():
        hub = p.split("/")[0] + "/" + p.split("/")[-1].replace(".html", "")
        for s in secs:
            if len(s["t0"]) < 300 or len(s["t1"]) < 300:
                continue
            if s["t0"] == s["t1"]:
                ident += 1
            out.append({"hub": hub, "title": "%s#%s" % (p, s["sid"]),
                        "pageid": "%s#%s" % (p, s["sid"]),
                        "rev0": OLD, "ts0": OLD, "rev1": NEW, "ts1": NEW,
                        "text0": s["t0"], "text1": s["t1"]})
    json.dump(out, open("pd_corpus.json", "w", encoding="utf-8"))
    print("DONE n=%d sections  identical=%d (%.1f%%)  pages=%d"
          % (len(out), ident, 100.0 * ident / max(len(out), 1), len(ck["docs"])))
