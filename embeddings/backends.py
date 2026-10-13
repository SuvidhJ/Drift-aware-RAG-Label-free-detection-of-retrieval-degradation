"""Embedding backend.

``MiniLMNumpyBackend``   a BERT-shaped sentence encoder (all-MiniLM-L6-v2,
                        bge-small-en-v1.5, ...) evaluated in pure numpy, straight from
                        the safetensors checkpoint.  No torch, no transformers.

It returns L2-normalised float32 arrays of shape (n, d).
"""
from __future__ import annotations

import json
import os
import pathlib
import struct
import unicodedata
from typing import Protocol

import numpy as np
from scipy.special import ndtr

_DTYPES = {"F32": np.float32, "F16": np.float16, "I64": np.int64}


class EmbeddingBackend(Protocol):
    """Anything that turns text into unit-norm row vectors."""

    dim: int
    name: str

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray: ...


# --------------------------------------------------------------------------- weights


def load_safetensors(path: str | pathlib.Path) -> dict[str, np.ndarray]:
    """Minimal safetensors reader - avoids a dependency on the safetensors package."""
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        blob = fh.read()

    out: dict[str, np.ndarray] = {}
    for key, spec in header.items():
        if key == "__metadata__":
            continue
        dtype = _DTYPES.get(spec["dtype"])
        if dtype is None:
            raise ValueError(
                f"unsupported safetensors dtype {spec['dtype']!r} for tensor {key!r}"
            )
        start, end = spec["data_offsets"]
        out[key] = np.frombuffer(blob[start:end], dtype=dtype).reshape(spec["shape"])
    return out


def resolve_minilm_dir(explicit: str | pathlib.Path | None = None) -> pathlib.Path:
    """Locate the all-MiniLM-L6-v2 snapshot.

    Order: explicit argument, then ``DRIFT_MINILM_DIR``, then the HuggingFace hub
    cache.  Raises with a fix-it message rather than falling back to something that
    silently differs.
    """
    def _usable(p: pathlib.Path) -> bool:
        return (p / "model.safetensors").is_file() and (p / "vocab.txt").is_file()

    # An explicitly named directory is a hard requirement, not a preference. Falling
    # through to the hub cache here would silently embed with a different model than
    # the caller asked for, which is precisely the kind of substitution that makes a
    # cached vector untraceable to the weights that produced it.
    for source, value in (("model_dir", explicit), ("DRIFT_MINILM_DIR", os.environ.get("DRIFT_MINILM_DIR"))):
        if value:
            path = pathlib.Path(value)
            if _usable(path):
                return path
            raise FileNotFoundError(
                f"{source} points at {path}, which does not contain both "
                "model.safetensors and vocab.txt.\n"
                "Fix the path, or unset it to fall back to the HuggingFace cache."
            )

    candidates: list[pathlib.Path] = []

    hub = pathlib.Path(
        os.environ.get("HF_HOME", pathlib.Path.home() / ".cache" / "huggingface")
    )
    if hub.name != "hub":
        hub = hub / "hub"
    snap_root = hub / "models--sentence-transformers--all-MiniLM-L6-v2" / "snapshots"
    if snap_root.is_dir():
        candidates.extend(sorted(snap_root.iterdir()))

    for cand in candidates:
        if _usable(cand):
            return cand

    listed = "\n  ".join(str(c) for c in candidates) if candidates else "<no candidates>"
    raise FileNotFoundError(
        "all-MiniLM-L6-v2 checkpoint not found. Expected model.safetensors and "
        "vocab.txt in one of:\n  " + listed + "\n"
        "Fix: set DRIFT_MINILM_DIR, or fetch the model once with\n"
        "  huggingface-cli download sentence-transformers/all-MiniLM-L6-v2"
    )


def _detect_pooling(root: pathlib.Path) -> str:
    """Pooling mode from a sentence-transformers snapshot, defaulting to mean.

    The default matches all-MiniLM-L6-v2, whose snapshot this module was first
    written against and which is often fetched without its 1_Pooling directory.
    """
    cfg = root / "1_Pooling" / "config.json"
    if cfg.is_file():
        c = json.loads(cfg.read_text())
        if c.get("pooling_mode_cls_token"):
            return "cls"
    return "mean"


# --------------------------------------------------------------------------- tokenizer


class WordPieceTokenizer:
    """BertTokenizer(do_lower_case=True) - basic tokenisation then greedy WordPiece."""

    def __init__(self, vocab_file: str | pathlib.Path) -> None:
        self.vocab: dict[str, int] = {}
        with open(vocab_file, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                self.vocab[line.rstrip("\n")] = i
        self.unk = self.vocab["[UNK]"]
        self.cls = self.vocab["[CLS]"]
        self.sep = self.vocab["[SEP]"]
        self.pad = self.vocab["[PAD]"]

    @staticmethod
    def basic_tokenize(text: str) -> list[str]:
        text = unicodedata.normalize("NFD", text.lower())
        text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
        out: list[str] = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or unicodedata.category(ch) == "Cc":
                out.append(" ")
            elif (
                (33 <= cp <= 47)
                or (58 <= cp <= 64)
                or (91 <= cp <= 96)
                or (123 <= cp <= 126)
                or unicodedata.category(ch).startswith("P")
            ):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        return "".join(out).split()

    def wordpiece(self, word: str) -> list[int]:
        if len(word) > 100:
            return [self.unk]
        ids: list[int] = []
        start = 0
        while start < len(word):
            end = len(word)
            cur = None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = self.vocab[sub]
                    break
                end -= 1
            if cur is None:
                return [self.unk]
            ids.append(cur)
            start = end
        return ids

    def encode(self, text: str, max_len: int = 256) -> tuple[list[int], list[int]]:
        ids = [self.cls]
        for word in self.basic_tokenize(text):
            ids += self.wordpiece(word)
            if len(ids) >= max_len - 1:
                break
        ids = ids[: max_len - 1] + [self.sep]
        mask = [1] * len(ids) + [0] * (max_len - len(ids))
        ids = ids + [self.pad] * (max_len - len(ids))
        return ids, mask


# --------------------------------------------------------------------------- model


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """LayerNorm that stays in the input dtype.

    ``eps`` is cast to x's dtype deliberately: a bare Python float promotes the whole
    expression to float64, and because every later op in the layer inherits that, one
    stray scalar doubled the cost of the entire forward pass.
    """
    eps_t = np.asarray(eps, dtype=x.dtype)
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps_t) * w + b


def gelu(x: np.ndarray) -> np.ndarray:
    """Exact gelu, matching the checkpoint's hidden_act.

    gelu(x) = x * Phi(x), and ``scipy.special.ndtr`` is Phi directly.  The equivalent
    ``0.5 * x * (1 + erf(x / sqrt(2)))`` agrees to 3.5e-7 but is both slower and
    float64-promoting, and gelu is the single most expensive op in the model -- about
    6x a projection matmul at this shape -- so the difference is worth having.
    """
    return x * ndtr(x)


class MiniLMNumpyBackend:
    """6-layer BERT forward pass, mean pooling over the attention mask, L2 normalise.

    Mirrors modules.json for sentence-transformers/all-MiniLM-L6-v2.
    """

    name = "all-MiniLM-L6-v2 (numpy)"

    def __init__(
        self,
        model_dir: str | pathlib.Path | None = None,
        max_len: int = 256,
        pooling: str | None = None,
    ) -> None:
        root = resolve_minilm_dir(model_dir)
        self.root = root
        self.W = load_safetensors(root / "model.safetensors")
        self.tok = WordPieceTokenizer(root / "vocab.txt")

        cfg_path = root / "config.json"
        cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
        self.dim = int(cfg.get("hidden_size", 384))
        self.n_heads = int(cfg.get("num_attention_heads", 12))
        self.n_layers = int(cfg.get("num_hidden_layers", 6))
        self.max_len = max_len

        # Any BERT-shaped checkpoint runs through this forward pass; what differs is
        # the pooling. sentence-transformers records it in 1_Pooling/config.json
        # (bge-small-en-v1.5 uses the CLS token, all-MiniLM-L6-v2 the mean).
        self.pooling = pooling or _detect_pooling(root)
        if self.pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be 'mean' or 'cls', got {self.pooling!r}")

        # The Encoder keys its on-disk cache on this name. Two models sharing it would
        # silently mix their vectors in one cache file, so only the original MiniLM
        # keeps the legacy name (existing caches stay valid); anything else is named
        # after its own checkpoint.
        if self.pooling == "mean" and self.n_layers == 6:
            self.name = "all-MiniLM-L6-v2 (numpy)"
        else:
            self.name = f"{root.name} (numpy, {self.pooling} pooling)"
        self.head_dim = self.dim // self.n_heads
        # float32 constant: a Python float here would promote every attention
        # score matrix to float64.
        self._attn_scale = np.float32(1.0 / np.sqrt(self.head_dim))

    def _forward(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        W = self.W
        B, T = ids.shape
        x = (
            W["embeddings.word_embeddings.weight"][ids]
            + W["embeddings.position_embeddings.weight"][None, :T, :]
            + W["embeddings.token_type_embeddings.weight"][0][None, None, :]
        )
        x = layer_norm(
            x, W["embeddings.LayerNorm.weight"], W["embeddings.LayerNorm.bias"]
        )
        additive_mask = (1.0 - mask[:, None, None, :].astype(np.float32)) * np.float32(-1e4)

        for layer in range(self.n_layers):
            p = f"encoder.layer.{layer}."

            def lin(h: np.ndarray, name: str, _p: str = p) -> np.ndarray:
                return h @ W[_p + name + ".weight"].T + W[_p + name + ".bias"]

            shape = (B, T, self.n_heads, self.head_dim)
            q = lin(x, "attention.self.query").reshape(shape).transpose(0, 2, 1, 3)
            k = lin(x, "attention.self.key").reshape(shape).transpose(0, 2, 1, 3)
            v = lin(x, "attention.self.value").reshape(shape).transpose(0, 2, 1, 3)

            scores = q @ k.transpose(0, 1, 3, 2) * self._attn_scale + additive_mask
            scores = scores - scores.max(-1, keepdims=True)
            attn = np.exp(scores)
            attn /= attn.sum(-1, keepdims=True)

            ctx = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, self.dim)
            ctx = lin(ctx, "attention.output.dense")
            x = layer_norm(
                x + ctx,
                W[p + "attention.output.LayerNorm.weight"],
                W[p + "attention.output.LayerNorm.bias"],
            )

            h = gelu(lin(x, "intermediate.dense"))
            h = lin(h, "output.dense")
            x = layer_norm(
                x + h, W[p + "output.LayerNorm.weight"], W[p + "output.LayerNorm.bias"]
            )

        if self.pooling == "cls":
            pooled = x[:, 0, :]
        else:
            m = mask[:, :, None].astype(np.float32)
            pooled = (x * m).sum(1) / np.maximum(m.sum(1), np.float32(1e-9))
        return pooled / np.maximum(
            np.linalg.norm(pooled, axis=1, keepdims=True), np.float32(1e-12)
        )

    def embed(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        chunks: list[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            enc = [self.tok.encode(t, self.max_len) for t in batch]
            ids = np.array([e[0] for e in enc], dtype=np.int64)
            mask = np.array([e[1] for e in enc], dtype=np.int64)
            width = max(int(mask.sum(1).max()), 1)
            chunks.append(self._forward(ids[:, :width], mask[:, :width]).astype(np.float32))
        return np.vstack(chunks)


def make_backend(kind: str, **kwargs) -> EmbeddingBackend:
    if kind in ("minilm-numpy", "numpy", "minilm"):
        return MiniLMNumpyBackend(
            kwargs.get("model_dir"),
            max_len=kwargs.get("max_len", 256),
            pooling=kwargs.get("pooling"),
        )
    raise ValueError(f"unknown embedding backend {kind!r}; expected 'minilm-numpy'")
