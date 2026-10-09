"""Corpora, models and an embedding cache keyed by (model, corpus, what).

The cache lives under ``embeddings/cache/staleness/`` -- gitignored, so vectors are
never committed, but outside ``/tmp``, which was wiped once mid-project and took every
embedding with it.  Every array is keyed by the model that produced it, so two models
can never share a file.
"""
from __future__ import annotations

import json
import os
import pathlib

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "research" / "gates" / "results"
CACHE = REPO / "embeddings" / "cache" / "staleness"

CORPORA = {
    "arxiv": "ax_corpus.json",            # 1,786 docs, 28% revised
    "arxiv_big": "ax_big_corpus.json",    # 6,835 docs, 25% revised
    "python_docs": "pd_corpus.json",      # 1,751 docs, 79% revised
    "wikipedia": "wp_corpus.json",        # 344 docs, 89% revised
}

MODELS = {
    # name: (checkpoint dir, env override, pooling) -- both run through the repo's own
    # numpy BERT forward pass, so the comparison isolates the model, not the runtime.
    "minilm": ("~/.cache/minilm-l6-v2", "DRIFT_MINILM_DIR", "mean"),
    "bge": ("~/.cache/bge-small-en-v1.5", "DRIFT_BGE_DIR", "cls"),
}


def load_corpus(name: str) -> list[dict]:
    return json.loads((RESULTS / CORPORA[name]).read_text(encoding="utf-8"))


def model_dir(model: str) -> pathlib.Path:
    default, env, _ = MODELS[model]
    return pathlib.Path(os.environ.get(env, default)).expanduser()


_backends: dict[str, object] = {}


def backend(model: str):
    if model not in _backends:
        from embeddings.backends import make_backend

        _, _, pooling = MODELS[model]
        _backends[model] = make_backend(
            "minilm-numpy", model_dir=model_dir(model), pooling=pooling
        )
    return _backends[model]


def cached(model: str, corpus: str, what: str, texts_fn) -> np.ndarray:
    """Embeddings for ``what`` (e.g. 'doc0', 'q_new'), computed once per model."""
    path = CACHE / model / f"{corpus}__{what}.npy"
    if path.exists():
        return np.load(path)
    texts = texts_fn()
    vecs = np.asarray(backend(model).embed(texts, batch_size=64), dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs
