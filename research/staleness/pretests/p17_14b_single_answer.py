"""Pre-test 17: is Qwen2.5-14B's listing of both values (p15: ACK 0.860 on real updates) induced by the prompt?

PRE-REGISTERED 2026-10-17, before any p17 generation (sha256 in `p17_prereg.log`).  Motivated by the p16
pilot (qwen2.5:7b, n=20, exploratory), where a single-answer prompt reduced ACK from 0.375 to 0.025.

Design   qwen2.5:14b-instruct; all 392 p15 questions; REAL source only; dates {CORRECT, SWAPPED} x order
         {M-LAST, M-FIRST} = 4 prompts per question (1,568 prompts); contexts identical to p15; prompt = p16
         strategy ONE ("exactly one number"); temperature 0; num_predict 24.
Baseline p15 BASE prompt on the same pairs (ACK 0.860, PASS 0.124, INSENSITIVE 0.014, INVERTED 0.003).
Decision (pre-registered)
  D1 listing is prompt-induced          if ACK(ONE) <= ACK(BASE) - 0.30.
  D2 under forced choice, 14B follows dates if PS-pass(ONE) >= 0.60;
     it ignores dates                        if PS-viol(ONE) >= 0.30;
     otherwise "mixed".
  Paired comparison with p15 BASE on the same (question, order) pairs: McNemar on ACK.
All outcomes reported whichever way they go.

    .venv/bin/python research/staleness/pretests/p17_14b_single_answer.py [--dry-run]
"""
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
DRY = "--dry-run" in sys.argv
sys.argv = sys.argv[:1] + ["--dry-run"]
import p16_strategies as P16  # noqa: E402

P15, P9 = P16.P15, P16.P9
GEN = "qwen2.5:14b-instruct"
OUT = P16.data.CACHE / "p17"


def mcnemar(b, c):
    n = b + c
    return 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def main():
    cases, _ = P15.build()
    keys = [("REAL", d, o) for d in ("CORRECT", "SWAPPED") for o in ("M-LAST", "M-FIRST")]
    print(f"questions {len(cases)}; prompts {len(cases) * len(keys)}", flush=True)
    if DRY:
        print(P16.prompts(cases[0], "ONE")[keys[0]]); return
    print("script sha256", hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), flush=True)
    recs = []
    for c in cases:
        pr = P16.prompts(c, "ONE")
        for k in keys:
            out = P9.generate(GEN, pr[k])
            recs.append({"q": c["q"], "dates": k[1], "order": k[2], "output": out, "label": P9.commit(c["new"], c["old"], out)})
    subprocess.run(["ollama", "stop", GEN], check=False)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "records.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    one = {(r["q"], r["dates"], r["order"]): r["label"] for r in recs}
    base = {(r["q"], r["dates"], r["order"]): r["label"] for r in map(json.loads, open(P15.OUT / "records.jsonl"))
            if r["gen"] == GEN and r["src"] == "REAL" and r["dates"] in ("CORRECT", "SWAPPED")}
    qs = sorted({k[0] for k in one})
    def classes(L):
        return Counter(P15.ps_label(L[(q, "CORRECT", o)], L[(q, "SWAPPED", o)]) for q in qs for o in ("M-LAST", "M-FIRST"))
    co, cb = classes(one), classes(base)
    n = sum(co.values())
    ack_o, ack_b = co["ACK"] / n, cb["ACK"] / n
    pas, vio = co["PASS"] / n, (co["INSENSITIVE"] + co["INVERTED"]) / n
    pairs = [(P15.ps_label(base[(q, "CORRECT", o)], base[(q, "SWAPPED", o)]) == "ACK",
              P15.ps_label(one[(q, "CORRECT", o)], one[(q, "SWAPPED", o)]) == "ACK") for q in qs for o in ("M-LAST", "M-FIRST")]
    b = sum(1 for x, y in pairs if x and not y); c = sum(1 for x, y in pairs if y and not x)
    print(f"\nBASE classes: {dict(cb)}  ACK {ack_b:.3f}")
    print(f"ONE  classes: {dict(co)}  ACK {ack_o:.3f}  PASS {pas:.3f} {P15.wilson(co['PASS'], n)}  PS-viol {vio:.3f} {P15.wilson(co['INSENSITIVE'] + co['INVERTED'], n)}")
    print(f"paired ACK change: BASE-only {b}, ONE-only {c}, McNemar p = {mcnemar(b, c):.3g}")
    print(f"D1 listing prompt-induced: drop {ack_b - ack_o:+.3f} -> {'SUPPORTED' if ack_b - ack_o >= 0.30 else 'NOT supported'}")
    print(f"D2 under forced choice: {'FOLLOWS DATES' if pas >= 0.60 else 'IGNORES DATES' if vio >= 0.30 else 'MIXED'} (PASS {pas:.3f}, PS-viol {vio:.3f})")


if __name__ == "__main__":
    main()
