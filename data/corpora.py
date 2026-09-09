"""Loaders for the real revision corpora.

Each corpus is a list of documents observed at two real timepoints:

    hub     topic group (arXiv category, docs section, wiki topic)
    title   human-readable name
    pageid  stable identifier
    rev0    revision id at the earlier timepoint
    ts0     timestamp of that revision
    rev1    revision id at the later timepoint
    ts1     timestamp of that revision
    text0   content at the earlier timepoint
    text1   content at the later timepoint

A document is *revised* when text0 != text1.  Nothing here is simulated: every pair
is a real document and a real later revision of that same document.

Why this replaces the previous source
-------------------------------------
The Amazon Reviews 2023 path produced a query stream that was a single constant
string for every row, so query-side drift was identically zero by construction, and
corpus snapshots that were near-identical across windows because product liveness
could not be reconstructed from the two truncated dumps (FINDINGS.md F3, F4).  It is
still reachable via ``DATA_SOURCE=amazon`` but it cannot produce a valid measurement.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from configs.settings import settings

CORPUS_FILES: dict[str, str] = {
    "arxiv": "ax_corpus.json",
    "python_docs": "pd_corpus.json",
    "wikipedia": "wp_corpus.json",
}

# Larger arXiv build, if it was fetched. Preferred when present: the retrieval effect
# is only resolvable around n ~ 7000 (FINDINGS.md 12.4).
CORPUS_ALTERNATES: dict[str, tuple[str, ...]] = {
    "arxiv": ("ax_corpus_big.json", "ax_corpus.json"),
}


@dataclass
class Document:
    doc_id: str
    hub: str
    title: str
    text0: str
    text1: str
    ts0: str
    ts1: str
    rev0: str
    rev1: str

    @property
    def revised(self) -> bool:
        return self.text0 != self.text1


@dataclass
class RevisionCorpus:
    name: str
    source_path: Path
    docs: list[Document]
    real_timestamps: bool

    def __len__(self) -> int:
        return len(self.docs)

    @property
    def n_revised(self) -> int:
        return sum(d.revised for d in self.docs)

    def summary(self) -> dict:
        return {
            "corpus": self.name,
            "source": str(self.source_path),
            "n_docs": len(self.docs),
            "n_revised": self.n_revised,
            "revised_fraction": self.n_revised / max(len(self.docs), 1),
            "n_hubs": len({d.hub for d in self.docs}),
            "real_timestamps": self.real_timestamps,
        }


def parse_timestamp(value: str) -> datetime | None:
    """Parse an ISO-8601 timestamp, returning None when the field is not a time.

    The Python-docs corpus stores version labels ("3.8", "3.13") rather than
    timestamps.  That is reported honestly via ``real_timestamps=False`` instead of
    being coerced into a fake date.
    """
    if not value:
        return None
    text = str(value).strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def resolve_corpus_path(name: str, corpus_dir: Path | None = None) -> Path:
    corpus_dir = Path(corpus_dir or settings.data.corpus_dir)
    if name not in CORPUS_FILES:
        raise ValueError(
            f"unknown corpus {name!r}; expected one of {sorted(CORPUS_FILES)}"
        )

    for candidate in CORPUS_ALTERNATES.get(name, (CORPUS_FILES[name],)):
        path = corpus_dir / candidate
        if path.is_file():
            return path

    raise FileNotFoundError(
        f"corpus {name!r} not found under {corpus_dir}.\n"
        f"Expected one of: {', '.join(CORPUS_ALTERNATES.get(name, (CORPUS_FILES[name],)))}\n"
        "These are collected by the scripts in research/gates/ "
        "(e6_fetch_arxiv.py, e6_pd_fetch.py, e6_fetch_wikipedia.py)."
    )


def load_corpus(
    name: str | None = None,
    corpus_dir: Path | None = None,
    max_docs: int | None = None,
) -> RevisionCorpus:
    """Load a revision corpus into canonical Document records."""
    name = name or settings.data.corpus
    path = resolve_corpus_path(name, corpus_dir)
    max_docs = max_docs if max_docs is not None else settings.data.max_docs

    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{path} should hold a JSON list of documents, got {type(raw).__name__}")

    docs: list[Document] = []
    for i, rec in enumerate(raw):
        text0 = (rec.get("text0") or "").strip()
        text1 = (rec.get("text1") or "").strip()
        if not text0 or not text1:
            continue
        docs.append(
            Document(
                doc_id=str(rec.get("pageid") or rec.get("title") or i),
                hub=str(rec.get("hub") or "unknown"),
                title=str(rec.get("title") or ""),
                text0=text0,
                text1=text1,
                ts0=str(rec.get("ts0") or ""),
                ts1=str(rec.get("ts1") or ""),
                rev0=str(rec.get("rev0") or ""),
                rev1=str(rec.get("rev1") or ""),
            )
        )
        if max_docs and len(docs) >= max_docs:
            break

    if not docs:
        raise ValueError(f"{path} yielded no usable documents (need non-empty text0 and text1)")

    real_ts = sum(parse_timestamp(d.ts1) is not None for d in docs) > 0.5 * len(docs)

    corpus = RevisionCorpus(name=name, source_path=path, docs=docs, real_timestamps=real_ts)
    logger.info(
        f"corpus {name}: {len(docs)} docs, {corpus.n_revised} revised "
        f"({100 * corpus.n_revised / len(docs):.1f}%), "
        f"{'real' if real_ts else 'version-label'} timestamps"
    )
    return corpus
