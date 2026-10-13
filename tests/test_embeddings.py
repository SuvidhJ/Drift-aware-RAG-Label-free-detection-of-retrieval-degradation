"""Embedding backend.

The numpy MiniLM implementation has no reference output shipped with it -- the model
card prints embeddings, not a similarity -- so it is validated by properties that a
correct sentence encoder must satisfy and an incorrect one would fail: unit norms,
semantic ordering, determinism, invariance to batching, and agreement with the
checkpoint's own tokenizer on special-token handling.
"""
from __future__ import annotations

import numpy as np
import pytest

from embeddings.backends import gelu, layer_norm, resolve_minilm_dir

pytestmark = pytest.mark.needs_model


def test_output_is_unit_norm_float32(model):
    e = model.embed(["a short sentence", "another entirely different sentence"])
    assert e.dtype == np.float32
    assert e.shape == (2, model.dim)
    assert np.allclose(np.linalg.norm(e, axis=1), 1.0, atol=1e-5)


def test_paraphrases_are_closer_than_unrelated_text(model):
    e = model.embed([
        "The cat sits on the mat.",
        "A feline rests upon the rug.",
        "Quantum chromodynamics describes the strong interaction.",
    ])
    sim = e @ e.T
    assert sim[0, 1] > sim[0, 2] + 0.2


def test_encoding_is_deterministic(model):
    text = ["determinism matters for a cache keyed on content"]
    assert np.array_equal(model.embed(text), model.embed(text))


def test_batching_does_not_change_the_result(model):
    texts = [f"sentence number {i} with enough words to tokenise properly" for i in range(9)]
    a = model.embed(texts, batch_size=2)
    b = model.embed(texts, batch_size=16)
    assert np.abs(a - b).max() < 2e-5, "padding is leaking into the pooled representation"


def test_empty_and_blank_inputs(model):
    assert model.embed([]).shape == (0, model.dim)
    blank = model.embed([""])
    assert blank.shape == (1, model.dim)
    assert np.isfinite(blank).all()


def test_long_input_is_truncated_not_crashed(model):
    long_text = "word " * 5000
    e = model.embed([long_text])
    assert e.shape == (1, model.dim)
    assert np.isfinite(e).all()


def test_revision_beyond_the_truncation_window_is_invisible(model):
    """A real and load-bearing consequence of max_len.

    Three of 91 revised arXiv documents in a smoke run changed only after character
    ~1500, past the 256-token window, so their token sequences -- and therefore their
    embeddings -- were identical. Displacement of exactly zero for an edited document
    is expected behaviour, not a bug, and anything measuring churn should know it.
    """
    prefix = "The established preamble sentence repeats unchanged. " * 200
    a = model.embed([prefix + "ORIGINAL TAIL."])
    b = model.embed([prefix + "COMPLETELY DIFFERENT TAIL."])
    assert np.abs(a - b).max() < 1e-6


def test_tokenizer_matches_the_checkpoint_vocabulary(model):
    tok = model.tok
    ids, mask = tok.encode("Hello, world!", max_len=32)
    assert ids[0] == tok.cls
    assert ids[sum(mask) - 1] == tok.sep
    assert all(i == tok.pad for i in ids[sum(mask):])
    assert len(ids) == len(mask) == 32


def test_tokenizer_lowercases_and_strips_accents(model):
    tok = model.tok
    assert tok.basic_tokenize("Café") == tok.basic_tokenize("cafe")
    assert tok.basic_tokenize("Hello, world!") == ["hello", ",", "world", "!"]


def test_unknown_words_fall_back_to_wordpiece(model):
    pieces = model.tok.wordpiece("unbelievabilities")
    assert len(pieces) > 1
    assert all(isinstance(p, int) for p in pieces)


def test_layer_norm_preserves_dtype():
    """A stray float64 scalar here promoted the whole forward pass and halved throughput."""
    x = np.random.default_rng(0).standard_normal((4, 8)).astype(np.float32)
    w = np.ones(8, dtype=np.float32)
    b = np.zeros(8, dtype=np.float32)
    out = layer_norm(x, w, b)
    assert out.dtype == np.float32
    assert np.allclose(out.mean(-1), 0.0, atol=1e-4)
    assert np.allclose(out.std(-1), 1.0, atol=1e-3)


def test_gelu_matches_the_erf_form_and_keeps_dtype():
    from scipy.special import erf

    x = np.linspace(-4, 4, 101).astype(np.float32)
    reference = 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))
    assert np.abs(gelu(x) - reference).max() < 1e-6
    assert gelu(x).dtype == np.float32


def test_explicit_bad_path_errors_instead_of_silently_substituting():
    """An explicit model_dir is a requirement, not a preference.

    Falling through to whatever is in the hub cache would embed with different
    weights than the caller asked for, and the resulting vectors would be
    untraceable to the model that produced them.
    """
    with pytest.raises(FileNotFoundError, match="model_dir"):
        resolve_minilm_dir("/definitely/not/a/model/dir")


def test_bad_env_override_errors(monkeypatch):
    monkeypatch.setenv("DRIFT_MINILM_DIR", "/also/not/a/model/dir")
    with pytest.raises(FileNotFoundError, match="DRIFT_MINILM_DIR"):
        resolve_minilm_dir()
