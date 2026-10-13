"""Pre-test 4: signed cos(q, d1-d0) -- does a revision move a document toward or away from its queries?"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))
from research.staleness import data, queries  # noqa: E402

rng = np.random.default_rng(3)
l2 = lambda X: X / np.linalg.norm(X, axis=1, keepdims=True)  # noqa: E731
for corpus in ("arxiv_big", "python_docs"):
    docs = data.load_corpus(corpus)
    for model in ("minilm", "bge"):
        D0 = l2(data.cached(model, corpus, "doc0", lambda: [d["text0"] for d in docs])); D1 = l2(data.cached(model, corpus, "doc1", lambda: [d["text1"] for d in docs]))
        out = []
        for pool, what, fn in (("retained", "q_retained", queries.retained_text), ("new", "q_new", queries.new_text)):
            Q = l2(data.cached(model, corpus, what, lambda fn=fn: fn(docs)[0])); g = np.array(fn(docs)[1])
            D = D1[g] - D0[g]; n = np.linalg.norm(D, axis=1); ok = n > 1e-6  # Δ=0: edit beyond truncation
            c = (Q[ok] * D[ok]).sum(1) / n[ok]
            ci = np.percentile([c[rng.integers(0, len(c), len(c))].mean() for _ in range(2000)], [2.5, 97.5])
            out.append(f"{pool}: {c.mean():+.4f} [{ci[0]:+.4f},{ci[1]:+.4f}] (n={ok.sum()}, Δ=0 for {np.mean(~ok):.2f})")
        print(f"{model}/{corpus}  mean signed cos(q, d1-d0):  " + "   ".join(out))
