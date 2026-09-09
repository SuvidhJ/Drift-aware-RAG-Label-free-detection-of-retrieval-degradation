"""Validate the numpy MiniLM by properties that are checkable offline.

No bit-exact torch reference is available in this environment, so this establishes:
(a) the tokenizer matches the checkpoint's own vocab/normaliser,
(b) the forward pass is deterministic and padding-invariant (catches masking bugs),
(c) semantic ordering is correct on a battery of pairs (catches weight/wiring bugs).
"""
import json, pathlib, numpy as np
from minilm_np import MiniLM, ROOT

m = MiniLM()

print("=== (a) tokenizer vs checkpoint tokenizer.json ===")
tj = json.load(open(pathlib.Path(ROOT)/"tokenizer.json", encoding="utf-8"))
vj = tj["model"]["vocab"]
print("  vocab size match      :", len(vj) == len(m.tok.vocab))
print("  vocab ids identical   :", all(m.tok.vocab.get(k) == v for k, v in vj.items()))
print("  normalizer            :", {k: v for k, v in tj["normalizer"].items() if k != "type"})
print("  unk/cls/sep/pad       :", m.tok.unk, m.tok.cls, m.tok.sep, m.tok.pad)
probe = "The RBI's 2023 policy—unchanged—affected NBFCs' café loans."
ids, _ = m.tok.encode(probe, 64)
inv = {v: k for k, v in m.tok.vocab.items()}
toks = [inv[i] for i in ids if i != m.tok.pad]
print("  sample tokenisation   :", " ".join(toks))
print("  [UNK] count           :", toks.count("[UNK]"), "(should be 0)")

print("\n=== (b) determinism and padding invariance ===")
s = ["This is an example sentence", "Each sentence is converted"]
e1 = m.embed(s)
e2 = m.embed(s)
padded = m.embed(s + ["x " * 200])          # forces a much longer batch window
print("  unit norm             :", np.allclose(np.linalg.norm(e1, axis=1), 1.0))
print("  deterministic         :", np.array_equal(e1, e2))
print("  padding-invariant     : max|diff| = %.2e" % float(np.abs(padded[:2] - e1).max()))
solo = np.vstack([m.embed([t]) for t in s])
print("  batch==solo           : max|diff| = %.2e" % float(np.abs(solo - e1).max()))

print("\n=== (c) semantic ordering battery ===")
pairs = [
 ("A man is playing a guitar.", "A man is playing a musical instrument.", "high"),
 ("A man is playing a guitar.", "A woman is slicing an onion.",            "low"),
 ("The stock market fell sharply.", "Equity indices declined steeply.",    "high"),
 ("The stock market fell sharply.", "The recipe calls for two eggs.",      "low"),
 ("How do I reset my password?", "What is the procedure to change my login credentials?", "high"),
 ("How do I reset my password?", "The Amazon rainforest spans nine countries.", "low"),
 ("Delhi is the capital of India.", "India's capital city is Delhi.",      "high"),
 ("Delhi is the capital of India.", "Photosynthesis converts light to sugar.", "low"),
 ("The patient developed a fever after surgery.", "Post-operative pyrexia was observed.", "high"),
 ("The patient developed a fever after surgery.", "The train departs from platform nine.", "low"),
]
hi, lo, ok = [], [], True
for a, b, exp in pairs:
    E = m.embed([a, b]); c = float(E[0] @ E[1])
    (hi if exp == "high" else lo).append(c)
    hit = (c > 0.45) if exp == "high" else (c < 0.25)
    ok &= hit
    print("  %-6s cos=%6.3f  %s" % (exp, c, "OK" if hit else "FAIL"))
print("\n  mean high = %.3f   mean low = %.3f   separation = %.3f" %
      (np.mean(hi), np.mean(lo), np.mean(hi) - np.mean(lo)))
print("  all pairs correct     :", ok)

print("\n=== (d) mask actually participates ===")
long_pad = m.embed(["hello world"] + ["filler text here that is quite a lot longer indeed"] * 4)
short    = m.embed(["hello world"])
print("  same emb w/ and w/o long neighbours: max|diff| = %.2e"
      % float(np.abs(long_pad[0] - short[0]).max()), "(≈0 proves padding is masked out)")
