import json, random, urllib.error, re, time, urllib.parse, urllib.request
from vcommon import FROZEN, P9, pages, sample
API = "https://en.wikipedia.org/w/api.php"
UA = "SemProject-verification/0.1 (academic replication check; python-urllib)"
def get(params):
    params = {**params, "format": "json", "formatversion": "2"}
    req = urllib.request.Request(API + "?" + urllib.parse.urlencode(params), headers={"User-Agent": UA})
    for i in range(8):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                time.sleep(3)
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 429: raise
            time.sleep(15 * (i + 1))
    raise RuntimeError("rate limited")
pgs = pages(); _, _, qs = sample()
docs_with_q = sorted({q["doc"] for q in qs})
pick = random.Random(5).sample(docs_with_q, 5)
import sys
for d in pick[int(sys.argv[1]):]:
    p = pgs[d]
    print(f"\n=== {p['title']} (pageid {p['pageid']}) ts0={p['ts0']} ts1={p['ts1']}")
    revs = {}
    for tag, ts in (("old", p["ts0"]), ("new", p["ts1"])):
        r = get({"action": "query", "prop": "revisions", "pageids": p["pageid"], "rvlimit": 1, "rvdir": "older",
                 "rvstart": ts, "rvprop": "ids|timestamp"})
        rv = r["query"]["pages"][0]["revisions"][0]
        c = get({"action": "query", "prop": "revisions", "revids": rv["revid"], "rvprop": "ids|content",
                 "rvslots": "main", "rvsection": "0"})
        text = c["query"]["pages"][0]["revisions"][0]["slots"]["main"]["content"]
        stored = p["text0" if tag == "old" else "text1"]
        revs[tag] = text
        print(f"  {tag}: revid {rv['revid']} ts {rv['timestamp']} (stored ts match {rv['timestamp'] == ts}); "
              f"live section-0 == stored text: {text == stored} (len {len(text)} vs {len(stored)})")
        time.sleep(1)
    # latest revision now (has the page changed since the fetch?)
    r = get({"action": "query", "prop": "revisions", "pageids": p["pageid"], "rvlimit": 1, "rvprop": "ids|timestamp"})
    print(f"  current latest revision: {r['query']['pages'][0]['revisions'][0]['timestamp']}")
    for q in [q for q in qs if q["doc"] == d]:
        f0, f1 = P9.fields(revs["old"]), P9.fields(revs["new"])
        k = q["key"]
        o_in_old = all(re.search(re.escape(v) + "|" + re.escape(v.replace(",", "")), revs["old"]) for v in q["old"])
        n_in_new = all(re.search(re.escape(v) + "|" + re.escape(v.replace(",", "")), revs["new"]) for v in q["new"])
        n_in_old_field = any(v in f0.get(k, "") for v in q["new"])
        print(f"  Q: {q['statement']}  old={q['old']} new={q['new']}")
        print(f"     old value in live old wikitext: {o_in_old}; new value in live new wikitext: {n_in_new}")
        print(f"     field '{k}' old: {f0.get(k)!r}\n     field '{k}' new: {f1.get(k)!r}\n     field changed: {f0.get(k) != f1.get(k)}; new value already in old field: {n_in_old_field}")
        raw0 = re.search(r"\|\s*" + re.escape(k).replace("_", "[_ ]") + r"\s*=([^\n]*)", revs["old"], re.I)
        raw1 = re.search(r"\|\s*" + re.escape(k).replace("_", "[_ ]") + r"\s*=([^\n]*)", revs["new"], re.I)
        print(f"     raw wikitext old: {raw0.group(1)[:110] if raw0 else None!r}\n     raw wikitext new: {raw1.group(1)[:110] if raw1 else None!r}")
