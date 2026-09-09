"""Caching encoder over a pluggable backend.

The cache key includes the backend name and a hash of the exact texts, so switching
``EMBED_BACKEND`` or rebuilding snapshots can never silently reuse vectors produced
from different inputs.  The previous version keyed only on window index and prefix.

The Lipschitz estimator that used to live here has been removed rather than repaired.
Two independent reasons (FINDINGS.md F15):

  * it estimated the wrong object -- the Kantorovich-Rubinstein bound needs the
    Lipschitz constant of the *retrieval functional* in embedding space, and W1 in
    D(t) is already measured there, so the encoder's constant is not the quantity the
    theory calls for;
  * the estimator was invalid anyway -- its denominator was a normalised character
    edit distance, which scales as 1/document-length, so the ratio it maximised over
    was 350x larger for a 300-word document than a 5-word one and ``L_hat = max`` was
    always won by the longest document. A bootstrap CI on a maximum is inconsistent
    for extreme order statistics besides.

Nothing downstream consumed it.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from configs.settings import settings
from embeddings.backends import EmbeddingBackend, make_backend


def _one_digest(text: str) -> str:
    """Content address for a single text. 128 bits: collisions are not a concern."""
    return hashlib.blake2b(text.encode("utf-8", "replace"), digest_size=16).hexdigest()


def _slug(value: str) -> str:
    return "".join(ch if ch.isalnum() else "-" for ch in value).strip("-").lower()


class Encoder:
    """Encodes text, caching by content so repeated phases are cheap."""

    def __init__(self, backend: EmbeddingBackend | None = None) -> None:
        cfg = settings.embedding
        self._cfg = cfg
        if backend is not None:
            self._backend = backend
        else:
            logger.info(f"loading embedding backend: {cfg.backend}")
            self._backend = make_backend(
                cfg.backend,
                model_dir=cfg.model_dir,
                model_name=cfg.model_name,
                device=cfg.device,
                max_len=cfg.max_len,
            )
        self._tag = _slug(self._backend.name)
        self._keys: dict[str, int] | None = None
        self._vecs: np.ndarray = np.zeros((0, 0), dtype=np.float32)
        cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"backend ready: {self._backend.name} (dim={self._backend.dim})")

    @property
    def backend(self) -> EmbeddingBackend:
        return self._backend

    @property
    def dim(self) -> int:
        return int(self._backend.dim)

    def encode(self, texts: list[str]) -> np.ndarray:
        return self._backend.embed(texts, batch_size=self._cfg.batch_size)

    def _encode_with_progress(self, texts: list[str], chunk: int = 512) -> np.ndarray:
        """Encode in chunks, reporting rate and ETA.

        A corpus-scale encode is tens of minutes of pure numpy with nothing on stdout.
        Silence that long is indistinguishable from a hang, and during this rebuild it
        cost a process kill to find out which it was.
        """
        if len(texts) <= chunk:
            return self.encode(texts)

        started = time.monotonic()
        out: list[np.ndarray] = []
        for start in range(0, len(texts), chunk):
            out.append(self.encode(texts[start : start + chunk]))
            done = min(start + chunk, len(texts))
            rate = done / max(time.monotonic() - started, 1e-9)
            remaining = (len(texts) - done) / rate if rate > 0 else float("inf")
            logger.info(
                f"  {done}/{len(texts)} texts  ({rate:.1f}/s, ~{remaining / 60:.1f} min left)"
            )
        return np.vstack(out)

    # -------------------------------------------------------------- text cache
    #
    # The cache is content-addressed per *text*, not per call.  Slices overlap
    # heavily -- the null slice's documents are exactly the stale revisions already
    # embedded for the real slice -- so a per-call cache would re-embed thousands of
    # texts it had already seen. Keying on the text itself makes that free, and makes
    # the cache correct by construction: identical text always maps to the vector the
    # current backend produced for it, and switching backends switches file.

    @property
    def _cache_path(self) -> Path:
        return self._cfg.cache_dir / f"textcache__{self._tag}.npz"

    def _load_cache(self) -> None:
        if self._keys is not None:
            return
        path = self._cache_path
        if path.exists():
            try:
                blob = np.load(path, allow_pickle=False)
                vecs = blob["vecs"]
                keys = [str(k) for k in blob["keys"]]
                self._vecs = vecs
                self._keys = {k: i for i, k in enumerate(keys)}
                logger.debug(f"text cache: {len(self._keys)} vectors from {path.name}")
                return
            except Exception as exc:
                logger.warning(f"text cache at {path} unreadable ({exc}); starting fresh")
        self._keys = {}
        self._vecs = np.zeros((0, self.dim), dtype=np.float32)

    def _save_cache(self) -> None:
        if not self._keys:
            return
        ordered = sorted(self._keys, key=self._keys.get)
        np.savez(self._cache_path, keys=np.array(ordered), vecs=self._vecs)

    def encode_cached(self, texts: list[str], name: str, force: bool = False) -> np.ndarray:
        """Encode ``texts``, reusing per-text cached vectors."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        self._load_cache()
        digests = [_one_digest(t) for t in texts]

        missing_pos: dict[str, str] = {}
        for d, t in zip(digests, texts):
            if force or d not in self._keys:
                missing_pos.setdefault(d, t)

        if missing_pos:
            todo_keys = list(missing_pos)
            logger.info(
                f"encoding {len(todo_keys)} new texts for {name} "
                f"({len(digests) - len(todo_keys)} of {len(digests)} already cached)"
            )
            fresh = self._encode_with_progress([missing_pos[k] for k in todo_keys])
            start = len(self._vecs)
            self._vecs = np.vstack([self._vecs, fresh.astype(np.float32)]) if start else fresh.astype(np.float32)
            for offset, k in enumerate(todo_keys):
                self._keys[k] = start + offset
            self._save_cache()
        else:
            logger.debug(f"{name}: all {len(digests)} texts cached")

        return self._vecs[np.array([self._keys[d] for d in digests], dtype=np.int64)]

    def encode_window(
        self, window_idx: int, snapshot_path: Path, prefix: str = "corpus", force: bool = False
    ) -> np.ndarray:
        df = pd.read_parquet(snapshot_path)
        if "text" not in df.columns:
            raise KeyError(f"{snapshot_path} has no 'text' column (columns: {list(df.columns)})")
        texts = df["text"].fillna("").astype(str).tolist()
        slice_name = Path(snapshot_path).parent.name
        return self.encode_cached(texts, f"{slice_name}__{prefix}_t{window_idx:03d}", force=force)

    def encode_snapshot_dir(
        self, snapshot_dir: Path, prefix: str = "corpus", force: bool = False
    ) -> list[np.ndarray]:
        """Embed every window of a snapshot directory, encoding each text only once.

        Across T windows a document takes only two distinct values -- its earlier and
        later revision -- so the union of texts over all windows is about 2n, not T*n.
        Encoding the union and gathering per window is therefore a ~T/2 saving on the
        corpus channel, which is the difference between a run that finishes in minutes
        and one that does not finish at all. Correctness is unaffected: identical text
        must produce an identical vector.
        """
        paths = sorted(Path(snapshot_dir).glob(f"{prefix}_t*.parquet"))
        if not paths:
            raise FileNotFoundError(
                f"no {prefix}_t*.parquet under {snapshot_dir}. Run: python main.py build-snapshots"
            )

        unique: dict[str, int] = {}
        per_window: list[np.ndarray] = []
        for path in paths:
            df = pd.read_parquet(path)
            if "text" not in df.columns:
                raise KeyError(f"{path} has no 'text' column (columns: {list(df.columns)})")
            rows = []
            for text in df["text"].fillna("").astype(str):
                pos = unique.get(text)
                if pos is None:
                    pos = len(unique)
                    unique[text] = pos
                rows.append(pos)
            per_window.append(np.asarray(rows, dtype=np.int64))

        total = sum(len(w) for w in per_window)
        logger.info(
            f"{snapshot_dir.name}/{prefix}: {len(unique)} unique texts across "
            f"{len(paths)} windows ({total} slots, {total / max(len(unique), 1):.1f}x reuse)"
        )

        table = self.encode_cached(
            list(unique.keys()), f"{Path(snapshot_dir).name}__{prefix}_unique", force=force
        )
        return [table[rows] for rows in per_window]


_encoder: Encoder | None = None


def get_encoder() -> Encoder:
    global _encoder
    if _encoder is None:
        _encoder = Encoder()
    return _encoder
