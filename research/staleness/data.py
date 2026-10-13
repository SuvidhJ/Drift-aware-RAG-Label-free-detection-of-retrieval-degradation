"""Corpora, models and a content-addressed embedding cache.

The cache lives under ``embeddings/cache/staleness/`` -- gitignored, so vectors are
never committed, but outside ``/tmp``, which was wiped once mid-project and took every
embedding with it.  Every array is keyed by the model that produced it AND by a digest
of the exact texts it embeds, so a changed corpus, pool or question definition gets new
vectors instead of silently reusing old ones.  (The cache used to key on
(model, corpus, name) alone; files written that way, without a digest in the name, are
left on disk and never read.)

Corpora are cleaned at load time; the JSON files on disk are the raw collector output
and are never rewritten:

``python_docs``  the Sphinx sidebar and footer ("Table of Contents ... Previous topic
                 ... 3.13.15 Documentation ... (c) Copyright 2001-2026 ...") are cut.
                 They held 187 of the 294 changed-fact questions.
``arxiv_big``    rows are deduplicated by ``pageid``.  A cross-listed paper appears
                 once per category (6,835 rows, 6,247 papers); the copies differ only
                 in ``hub``, and the first is kept.
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import re

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "research" / "gates" / "results"
CACHE = REPO / "embeddings" / "cache" / "staleness"

CORPORA = {
    "arxiv_big": "ax_big_corpus.json",    # 6,247 papers after dedup (6,835 raw rows)
    "python_docs": "pd_corpus.json",      # 1,751 docs; 1,381 revised after the footer cut
    "wikipedia": "wp_corpus.json",        # 344 docs, 89% revised
}

MODELS = {
    # name: (checkpoint dir, env override, pooling) -- both run through the repo's own
    # numpy BERT forward pass, so the comparison isolates the model, not the runtime.
    "minilm": ("~/.cache/minilm-l6-v2", "DRIFT_MINILM_DIR", "mean"),
    "bge": ("~/.cache/bge-small-en-v1.5", "DRIFT_BGE_DIR", "cls"),
}
MAX_LEN = 256   # encoder window in wordpieces, including [CLS] and [SEP]

# Sphinx page chrome.  Each section's text was cut at 4,000 characters, so on some
# pages the sidebar's "Table of Contents" is the last thing kept and no footer follows;
# it is chrome all the same.  "(Table of Contents)" in parentheses is IDLE's TOC button,
# which is body text.
SPHINX_CHROME = re.compile(
    r"(?<!\()Table of Contents(?!\))|Previous topic|Next topic"
    r"|This [Pp]age Report a [Bb]ug|« Navigation|Navigation index"
)


def strip_sphinx_chrome(text: str) -> str:
    m = SPHINX_CHROME.search(text)
    return text[: m.start()].rstrip() if m else text


def dedup(docs: list[dict], key: str = "pageid") -> list[dict]:
    seen, out = set(), []
    for d in docs:
        if d[key] not in seen:
            seen.add(d[key])
            out.append(d)
    return out


def load_corpus(name: str, raw: bool = False) -> list[dict]:
    docs = json.loads((RESULTS / CORPORA[name]).read_text(encoding="utf-8"))
    if raw:
        return docs
    if name == "python_docs":
        docs = [{**d, "text0": strip_sphinx_chrome(d["text0"]),
                 "text1": strip_sphinx_chrome(d["text1"])} for d in docs]
    if name == "arxiv_big":
        docs = dedup(docs)
    return docs


def model_dir(model: str) -> pathlib.Path:
    default, env, _ = MODELS[model]
    return pathlib.Path(os.environ.get(env, default)).expanduser()


_backends: dict[str, object] = {}


def backend(model: str):
    if model not in _backends:
        from embeddings.backends import make_backend

        _, _, pooling = MODELS[model]
        _backends[model] = make_backend(
            "minilm-numpy", model_dir=model_dir(model), pooling=pooling, max_len=MAX_LEN
        )
    return _backends[model]


_tokenizers: dict[str, object] = {}


def window_ids(model: str, text: str) -> list[int]:
    """The token ids the encoder actually sees: the first MAX_LEN - 2 wordpieces."""
    if model not in _tokenizers:
        from embeddings.backends import WordPieceTokenizer

        _tokenizers[model] = WordPieceTokenizer(model_dir(model) / "vocab.txt")
    return _tokenizers[model].encode(text, MAX_LEN)[0]


def visible(model: str, before: str, after: str) -> bool:
    """Can the encoder see this edit at all?

    False when both texts give identical token ids inside the window: the edit lies
    past the truncation point, the embedding cannot move, and the edit must not be
    counted as evidence that the encoder is insensitive to its content.
    """
    return window_ids(model, before) != window_ids(model, after)


def digest(model: str, texts: list[str]) -> str:
    _, _, pooling = MODELS[model]
    h = hashlib.sha256(f"{model}|{model_dir(model).name}|{pooling}|{MAX_LEN}".encode())
    for t in texts:
        b = t.encode("utf-8")
        h.update(len(b).to_bytes(8, "little"))
        h.update(b)
    return h.hexdigest()[:16]


def cached(model: str, corpus: str, what: str, texts_fn) -> np.ndarray:
    """Embeddings of ``texts_fn()`` (e.g. 'doc0', 'q_new'), computed once per content."""
    texts = list(texts_fn())
    path = CACHE / model / f"{corpus}__{what}__{digest(model, texts)}.npy"
    if path.exists():
        vecs = np.load(path)
        if len(vecs) == len(texts):
            return vecs
    vecs = np.asarray(backend(model).embed(texts, batch_size=64), dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, vecs)
    tmp.replace(path)
    return vecs
