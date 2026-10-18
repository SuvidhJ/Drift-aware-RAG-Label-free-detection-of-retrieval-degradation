"""Score the returned human labels against each other and against the automatic label (P9.commit).

    .venv/bin/python research/staleness/human_validation/score_agreement.py annotator_1.csv annotator_2.csv

Reports: Cohen's kappa (annotator 1 vs 2; each vs auto; adjudicated consensus vs auto), raw agreement,
confusion tables, and every disagreement with the auto label.  Consensus = cases where both annotators agree.
"""
import csv
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAB = "label (A/B/BOTH/NEITHER)"
CATS = ["A", "B", "BOTH", "NEITHER"]


def read(p):
    with open(p, newline="") as f:
        return {r["case_id"]: r for r in csv.DictReader(f)}


def kappa(x, y):
    n = len(x)
    po = sum(a == b for a, b in zip(x, y)) / n
    px, py = Counter(x), Counter(y)
    pe = sum(px[c] * py[c] for c in CATS) / (n * n)
    return (po - pe) / (1 - pe) if pe < 1 else float("nan"), po


def norm(v):
    v = (v or "").strip().upper()
    return {"BOTH": "BOTH", "NEITHER": "NEITHER", "A": "A", "B": "B"}.get(v, v)


def main(f1, f2):
    a1, a2 = read(f1), read(f2)
    key = read(HERE / "KEY_DO_NOT_SHARE.csv")
    ids = sorted(set(a1) & set(a2) & set(key))
    missing = [i for i in ids if norm(a1[i][LAB]) not in CATS or norm(a2[i][LAB]) not in CATS]
    if missing:
        print(f"WARNING: {len(missing)} cases with missing/invalid labels: {missing[:10]}")
    ids = [i for i in ids if i not in missing]
    h1 = [norm(a1[i][LAB]) for i in ids]; h2 = [norm(a2[i][LAB]) for i in ids]
    auto = [key[i]["auto_as_AB"] for i in ids]
    for name, x, y in (("annotator1 vs annotator2", h1, h2), ("annotator1 vs auto", h1, auto), ("annotator2 vs auto", h2, auto)):
        k, po = kappa(x, y)
        print(f"{name:<26} n={len(x)}  agreement {po:.3f}  Cohen's kappa {k:.3f}")
    cons = [(i, a, b) for i, a, b in zip(ids, h1, auto) if a == norm(a2[i][LAB])]
    k, po = kappa([a for _, a, _ in cons], [b for _, _, b in cons])
    print(f"{'consensus vs auto':<26} n={len(cons)}  agreement {po:.3f}  Cohen's kappa {k:.3f}")
    print("\nconfusion (rows = consensus human, cols = auto):")
    print("        " + "".join(f"{c:>9}" for c in CATS))
    for c in CATS:
        print(f"{c:<8}" + "".join(f"{sum(1 for _, a, b in cons if a == c and b == d):>9}" for d in CATS))
    print("\nper stratum (consensus agreement with auto):")
    for g in sorted({key[i]['gen'] for i, _, _ in cons}):
        sub = [(a, b) for i, a, b in cons if key[i]["gen"] == g]
        print(f"  {g:<22} {sum(a == b for a, b in sub)}/{len(sub)}")
    print("\ndisagreements between consensus and auto:")
    for i, a, b in cons:
        if a != b:
            print(f"  {i}: human {a:<8} auto {b:<8} answer={a1[i]['model_answer']!r}  A={a1[i]['value_A']} B={a1[i]['value_B']}")


if __name__ == "__main__":
    main(*(sys.argv[1:3] if len(sys.argv) >= 3 else (HERE / "annotator_1.csv", HERE / "annotator_2.csv")))
