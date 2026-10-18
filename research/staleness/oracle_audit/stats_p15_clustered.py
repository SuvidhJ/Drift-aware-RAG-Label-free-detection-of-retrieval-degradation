"""Page-clustered bootstrap 95% CIs for the p15 rates in Table 2 (published relation, MR-PS violation), and
for the differences MR-PS minus published.  10,000 resamples of pages (145), seed 0.  Descriptive companion to
the pre-registered Wilson intervals (units are not independent: 2 orders x up to several questions per page)."""
import json, random
from collections import defaultdict
from pathlib import Path
REC = Path("embeddings/cache/staleness/p15/records.jsonl")
recs = [json.loads(l) for l in open(REC) if l.strip()]
def ps_label(a, b):
    if a in ("both", "neither") or b in ("both", "neither"): return "ACK"
    if a == "new" and b == "old": return "PASS"
    return "INSENSITIVE" if a == b else "INVERTED"
rng = random.Random(0)
print(f"{'model':<22}{'src':<5}{'published [page-boot]':>28}{'MR-PS viol [page-boot]':>28}{'diff [page-boot]':>26}")
for g in ("phi3:mini", "qwen2.5:7b-instruct", "qwen2.5:14b-instruct"):
    R = {(r["q"], r["src"], r["dates"], r["order"]): r["label"] for r in recs if r["gen"] == g}
    page = {r["q"]: r["page"] for r in recs if r["gen"] == g}
    for src in ("REAL", "CN"):
        by = defaultdict(lambda: [0, 0, 0, 0])   # page -> [pub_k, pub_n, ps_k, ps_n]
        for q in sorted(page):
            b = by[page[q]]
            b[0] += R[(q, src, "NONE", "M-LAST")] == "old"; b[1] += 1
            for o in ("M-LAST", "M-FIRST"):
                l = ps_label(R[(q, src, "CORRECT", o)], R[(q, src, "SWAPPED", o)])
                b[2] += l in ("INSENSITIVE", "INVERTED"); b[3] += 1
        pages = list(by)
        tot = [sum(by[p][i] for p in pages) for i in range(4)]
        est = (tot[0] / tot[1], tot[2] / tot[3])
        bs = []
        for _ in range(10000):
            pick = [rng.choice(pages) for _ in pages]
            s = [sum(by[p][i] for p in pick) for i in range(4)]
            bs.append((s[0] / s[1], s[2] / s[3], s[2] / s[3] - s[0] / s[1]))
        ci = lambda j: (sorted(x[j] for x in bs)[249], sorted(x[j] for x in bs)[9749])
        c0, c1, c2 = ci(0), ci(1), ci(2)
        print(f"{g:<22}{src:<5}{est[0]:>9.3f} [{c0[0]:.3f}, {c0[1]:.3f}]{est[1]:>11.3f} [{c1[0]:.3f}, {c1[1]:.3f}]{est[1]-est[0]:>+9.3f} [{c2[0]:+.3f}, {c2[1]:+.3f}]")
