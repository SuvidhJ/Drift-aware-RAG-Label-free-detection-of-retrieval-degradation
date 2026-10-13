"""Pre-test 10: can FreshQA give real, human-written questions with real document revisions?

FreshQA (Vu et al., arXiv 2310.03214; Apache-2.0) is a set of 600 questions whose answers
change, re-released in dated versions (Feb 2024 .. Apr 2026).  Versions are matched on the
question text (the `id` column is not stable across versions).

Usable question (pre-registered 2026-10-11, before any Wikipedia text was fetched):
  - true premise; fact_type slow- or fast-changing
  - answer_0 differs between the Feb 26 2024 and Apr 21 2026 versions, and not just in
    formatting: neither normalised answer contains the other
  - the source is an English Wikipedia page
  - in that page, the old answer appears in the revision live on 2024-02-26, the new
    answer appears in the current revision, and the new answer is absent from the 2024
    revision (otherwise the old document was never stale for this question)

    .venv/bin/python research/staleness/pretests/p10_freshqa_feasibility.py
"""
import csv
import io
import json
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[3]))
sys.argv = sys.argv[:1] + ["none"]
import e0_fact_rates as W  # noqa: E402
from research.staleness import data  # noqa: E402

SHEETS = {"2024-02-26": "1V6nIxVTI9tqZ-wfgK-uFuUPiGEa1Zmnz53OeGbaNtO0",
          "2026-04-21": "1_8mi-yuK30mvoDJu1KQXD6ODem7MKMcIgVAwDSzJkjM"}
RAW = data.CACHE / "p10_freshqa"


def norm(s: str) -> str:
    return re.sub(r"\W+", " ", s.lower()).strip()


def sheet(date: str) -> dict:
    path = RAW / f"freshqa_{date}.csv"
    if not path.exists():
        url = f"https://docs.google.com/spreadsheets/d/{SHEETS[date]}/export?format=csv"
        RAW.mkdir(parents=True, exist_ok=True)
        path.write_bytes(urllib.request.urlopen(url, timeout=60).read())
    rows = list(csv.reader(io.StringIO(path.read_text(encoding="utf-8"))))
    h = rows[2]
    return {norm(d["question"]): d for d in (dict(zip(h, r)) for r in rows[3:] if len(r) > 3 and r[2].strip())}


def page_text(title: str, when: str | None) -> str | None:
    """Plain text of the page as of `when` (ISO date), or now."""
    params = {"action": "query", "prop": "revisions", "titles": title, "rvlimit": 1, "redirects": 1,
              "rvprop": "ids|timestamp|content", "rvslots": "main"}
    if when:
        params |= {"rvdir": "older", "rvstart": f"{when}T23:59:59Z"}
    r = W.get(params)
    pg = r["query"]["pages"][0]
    revs = pg.get("revisions", [])
    if not revs:
        return None
    return W.prose(revs[0]["slots"]["main"].get("content", "") or "")


def present(answer: str, text: str) -> bool:
    a = norm(answer)
    return bool(a) and f" {a} " in f" {norm(text)} "


if __name__ == "__main__":
    old, new = sheet("2024-02-26"), sheet("2026-04-21")
    cands = []
    for q, r in new.items():
        o = old.get(q)
        if not o or r["false_premise"].strip().upper() == "TRUE" or r["fact_type"] not in ("fast-changing", "slow-changing"):
            continue
        a0, a1 = norm(o["answer_0"]), norm(r["answer_0"])
        if not a0 or not a1 or a0 in a1 or a1 in a0 or "wikipedia.org/wiki/" not in r["source"]:
            continue
        title = urllib.parse.unquote(r["source"].split("wikipedia.org/wiki/")[1].split("#")[0]).replace("_", " ")
        cands.append({"question": r["question"], "old": o["answer_0"], "new": r["answer_0"],
                      "title": title, "fact_type": r["fact_type"]})
    print(f"{len(cands)} candidates after the formatting filter", flush=True)
    out = []
    for c in cands:
        path = RAW / f"page_{norm(c['title']).replace(' ', '_')[:80]}.json"
        if path.exists():
            t = json.loads(path.read_text())
        else:
            t = {"t0": page_text(c["title"], "2024-02-26"), "t1": page_text(c["title"], None)}
            path.write_text(json.dumps(t))
        if not t["t0"] or not t["t1"]:
            continue
        ok_old, ok_new, new_in_old = present(c["old"], t["t0"]), present(c["new"], t["t1"]), present(c["new"], t["t0"])
        c |= {"old_in_t0": ok_old, "new_in_t1": ok_new, "new_in_t0": new_in_old,
              "usable": ok_old and ok_new and not new_in_old}
        out.append(c)
    (RAW / "feasibility.json").write_text(json.dumps(out, indent=1))
    u = [c for c in out if c["usable"]]
    print(f"pages fetched {len(out)}; old answer in 2024 page {sum(c['old_in_t0'] for c in out)}; "
          f"new answer in current page {sum(c['new_in_t1'] for c in out)}; new already in 2024 page "
          f"{sum(c['new_in_t0'] for c in out)}  ->  USABLE {len(u)}")
    for c in u[:20]:
        print(f"  {c['question'][:60]:<60} | {c['old'][:22]} -> {c['new'][:22]}")
