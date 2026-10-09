"""What are people asking about now that they were not asking about before?

Why this exists
---------------
`drift/statistic.py` answers query drift with a z-score.  A z-score tells an operator
that *something* moved; it does not say what, it does not say whether the index can
serve it, and it cannot be argued with.  The operational question is the named one:
"80% of traffic was about computers, now 80% is about AI" -- and the useful part of
that sentence is the nouns, not a p-value attached to it.

So this module reports drift as composition, not as distance:

  * every query is assigned to its nearest topic anchor, which is a real labelled
    region of the corpus, not an anonymous cluster id
  * the mixture over topics is tracked window by window
  * the report names the risers, the fallers, the window where they crossed over,
    and -- the part that is actionable -- the topics whose share of traffic has
    outrun their share of the index

Nothing here is a hypothesis test.  The inference is compositional: shares, ratios
and rankings that a human can read and dispute.  `coverage_gap` is the output that
matters, because it converts a topic shift into a statement about what to go and
index next.

Anchors
-------
`TopicModel.from_labels` uses labels the corpus already carries (arXiv categories,
documentation sections, wiki hubs).  `TopicModel.from_clusters` falls back to
k-means when there are none, and then the topic "names" are the titles of the
documents nearest each centroid -- still readable, just less authoritative.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np


@dataclass
class TopicModel:
    """Named anchors in embedding space, one per topic.

    Assignment defaults to kNN over the documents rather than nearest centroid.
    That is not a stylistic choice -- measured on the arXiv corpus against the
    documents' own categories, nearest-centroid is 67.1% correct and kNN at k=5 is
    73.7%.  Overlapping topics (cs.CV / cs.CL / cs.LG) have centroids close enough
    that the nearest one is often not the right one, and the error shows up directly
    as mass appearing in topics that received no traffic at all.

    Call ``validate`` whenever labelled queries exist.  The shares this module
    reports are only as trustworthy as the assignment behind them, and that number
    belongs next to them.
    """

    names: list[str]
    centroids: np.ndarray          # (n_topics, d), unit norm
    doc_counts: np.ndarray         # documents backing each topic
    doc_embeddings: np.ndarray | None = None
    doc_labels: np.ndarray | None = None   # index into names, per document
    assign_mode: str = "knn"       # "knn" | "centroid"
    k: int = 5

    @classmethod
    def from_labels(
        cls, embeddings: np.ndarray, labels: list[str], k: int = 5
    ) -> "TopicModel":
        X = np.asarray(embeddings, dtype=np.float64)
        names = sorted(set(labels))
        lab = np.asarray(labels)
        cents, counts = [], []
        for nm in names:
            rows = X[lab == nm]
            c = rows.mean(axis=0)
            n = np.linalg.norm(c)
            cents.append(c / n if n > 0 else c)
            counts.append(len(rows))
        idx = {nm: i for i, nm in enumerate(names)}
        return cls(
            names, np.stack(cents), np.asarray(counts),
            doc_embeddings=X,
            doc_labels=np.array([idx[l] for l in labels]),
            assign_mode="knn", k=k,
        )

    def validate(self, queries: np.ndarray, true_labels: list[str]) -> float:
        """Share of queries assigned to the topic they actually belong to."""
        pred = self.assign(queries)
        truth = np.array([self.names.index(l) for l in true_labels])
        return float((pred == truth).mean())

    def assign_confident(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Topic index plus the share of the kNN vote that backed it.

        A split vote means the query sits between topics.  Reporting it as a
        confident member of whichever side won by one neighbour is how a topic that
        received no traffic ends up with a share, so the confidence is returned and
        ``analyse`` can decline to count it.
        """
        Q = np.asarray(queries, dtype=np.float64)
        if self.assign_mode != "knn" or self.doc_embeddings is None:
            sims = Q @ self.centroids.T
            return np.argmax(sims, axis=1), np.ones(len(Q))

        X, lab = self.doc_embeddings, self.doc_labels
        k = min(self.k, len(X))
        out = np.empty(len(Q), dtype=np.int64)
        conf = np.empty(len(Q))
        for s in range(0, len(Q), 256):
            S = Q[s : s + 256] @ X.T
            nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
            for r in range(nn.shape[0]):
                counts = np.bincount(lab[nn[r]], minlength=len(self.names))
                out[s + r] = int(np.argmax(counts))
                conf[s + r] = counts.max() / k
        return out, conf

    def regroup(self, mapping: dict[str, str]) -> "TopicModel":
        """Coarsen to themes, which is where assignment becomes reliable.

        Measured on arXiv: 14 categories assign at 73.7%, and the seven themes they
        roll up into assign at 84.0%.  39% of the fine-grained errors were already
        inside the right theme -- ``stat.ME`` called ``econ.EM``, ``cs.LG`` called
        ``cs.CV`` -- which are disagreements about labelling, not about meaning.
        Report at the granularity the assignment can actually support.

        Args:
            mapping: fine topic name -> theme name. Topics absent from the mapping
                keep their own name as their theme.
        """
        if self.doc_embeddings is None or self.doc_labels is None:
            raise ValueError("regroup needs the documents; build with from_labels")
        themes = [mapping.get(n, n) for n in self.names]
        labels = [themes[i] for i in self.doc_labels]
        return TopicModel.from_labels(self.doc_embeddings, labels, k=self.k)

    @classmethod
    def from_clusters(
        cls, embeddings: np.ndarray, titles: list[str], k: int = 12, seed: int = 0
    ) -> "TopicModel":
        from sklearn.cluster import KMeans

        X = np.asarray(embeddings, dtype=np.float64)
        km = KMeans(n_clusters=k, random_state=seed, n_init=10).fit(X)
        cents, names, counts = [], [], []
        for i in range(k):
            c = km.cluster_centers_[i]
            n = np.linalg.norm(c)
            c = c / n if n > 0 else c
            cents.append(c)
            members = np.flatnonzero(km.labels_ == i)
            # name the topic after the document closest to its centre
            best = members[np.argmax(X[members] @ c)] if len(members) else 0
            names.append(titles[best][:48])
            counts.append(int(len(members)))
        return cls(names, np.stack(cents), np.asarray(counts))

    def assign(self, queries: np.ndarray) -> np.ndarray:
        """Topic index for each query.

        kNN where documents are available (more accurate on overlapping topics),
        nearest-centroid otherwise.
        """
        Q = np.asarray(queries, dtype=np.float64)
        if self.assign_mode != "knn" or self.doc_embeddings is None:
            return np.argmax(Q @ self.centroids.T, axis=1)

        X, lab = self.doc_embeddings, self.doc_labels
        k = min(self.k, len(X))
        out = np.empty(len(Q), dtype=np.int64)
        for s in range(0, len(Q), 256):
            S = Q[s : s + 256] @ X.T
            nn = np.argpartition(-S, k - 1, axis=1)[:, :k]
            for r in range(nn.shape[0]):
                counts = np.bincount(lab[nn[r]], minlength=len(self.names))
                out[s + r] = int(np.argmax(counts))
        return out

    def mixture(
        self, queries: np.ndarray, min_confidence: float = 0.0
    ) -> tuple[np.ndarray, float]:
        """Share of queries falling to each topic, and the share that were counted.

        Queries whose assignment did not clear ``min_confidence`` are abstained on:
        they are excluded from the shares rather than pushed into whichever topic
        won by one neighbour. The second return value is the coverage.
        """
        a, c = self.assign_confident(queries)
        sel = c >= min_confidence
        if not sel.any():
            return np.zeros(len(self.names)), 0.0
        m = np.bincount(a[sel], minlength=len(self.names)).astype(np.float64)
        return m / m.sum(), float(sel.mean())


@dataclass
class TopicShift:
    name: str
    first: float
    last: float
    peak: float
    delta: float

    @property
    def fold(self) -> float:
        return self.last / self.first if self.first > 1e-9 else float("inf")


@dataclass
class DriftReport:
    topics: list[str]
    mixtures: np.ndarray           # (n_windows, n_topics)
    risers: list[TopicShift]
    fallers: list[TopicShift]
    crossovers: list[tuple[int, str, str]]
    coverage: list[tuple[str, float, float, float]]   # topic, traffic, index, ratio
    assigned_share: list[float]      # per window, fraction of queries confident enough
    min_confidence: float = 0.0

    def render(self, top: int = 5) -> str:
        W = len(self.mixtures)
        out = [
            "QUERY TOPIC DRIFT",
            f"  {W} windows, {len(self.topics)} topics, composition measured directly",
        ]
        if self.min_confidence > 0:
            lo, hi = min(self.assigned_share), max(self.assigned_share)
            out.append(
                f"  abstaining below a {self.min_confidence:.0%} vote; "
                f"{lo:.0%}-{hi:.0%} of queries counted"
            )
        out.append("")
        out.append("  RISING  (share of queries, first window -> last)")
        for s in self.risers[:top]:
            fold = "new" if s.first < 1e-9 else f"{s.fold:.1f}x"
            out.append(f"    {s.name:<24} {s.first:6.1%} -> {s.last:6.1%}   {fold:>6}")
        out.append("")
        out.append("  FALLING")
        for s in self.fallers[:top]:
            out.append(
                f"    {s.name:<24} {s.first:6.1%} -> {s.last:6.1%}   "
                f"{s.delta:+.1%}"
            )
        if self.crossovers:
            out.append("")
            out.append("  CROSSOVERS  (the window where one overtook the other)")
            for w, up, down in self.crossovers[:top]:
                out.append(f"    window {w:<3} {up} overtook {down}")
        out.append("")
        out.append("  COVERAGE  (share of traffic vs share of the index)")
        out.append(f"    {'topic':<24}{'traffic':>9}{'index':>9}{'ratio':>8}")
        for nm, traffic, index, ratio in self.coverage[:top]:
            flag = "  under-served" if ratio > 1.5 else ""
            out.append(f"    {nm:<24}{traffic:>9.1%}{index:>9.1%}{ratio:>8.2f}{flag}")
        return "\n".join(out)


def analyse(
    model: TopicModel,
    query_windows: list[np.ndarray],
    min_share: float = 0.02,
    min_confidence: float = 0.0,
) -> DriftReport:
    """Track topic composition across windows and name what moved.

    Args:
        model: the topic anchors
        query_windows: one (m_t, d) array of query embeddings per window, in order
        min_share: ignore topics that never reach this share in any window
        min_confidence: abstain on queries whose assignment vote is weaker than this.
            On arXiv themes, 0.6 keeps 84% of queries at 87.6% accuracy against
            84.0% at full coverage; 0.8 keeps 50% at 95.2%.
    """
    if len(query_windows) < 2:
        raise ValueError("need at least two windows to describe a shift")

    pairs = [model.mixture(q, min_confidence) for q in query_windows]
    M = np.stack([p[0] for p in pairs])                       # (W, T)
    assigned = [p[1] for p in pairs]
    keep = M.max(axis=0) >= min_share
    shifts = []
    for j, nm in enumerate(model.names):
        if not keep[j]:
            continue
        shifts.append(
            TopicShift(
                name=nm,
                first=float(M[0, j]),
                last=float(M[-1, j]),
                peak=float(M[:, j].max()),
                delta=float(M[-1, j] - M[0, j]),
            )
        )
    risers = sorted([s for s in shifts if s.delta > 0], key=lambda s: -s.delta)
    fallers = sorted([s for s in shifts if s.delta < 0], key=lambda s: s.delta)

    # crossovers: the window where a riser first passes a faller it started below
    crossovers: list[tuple[int, str, str]] = []
    for up in risers[:3]:
        iu = model.names.index(up.name)
        for down in fallers[:3]:
            idn = model.names.index(down.name)
            if M[0, iu] >= M[0, idn]:
                continue
            ahead = np.flatnonzero(M[:, iu] > M[:, idn])
            if len(ahead):
                crossovers.append((int(ahead[0]), up.name, down.name))
    crossovers.sort()

    # coverage: traffic share in the final window against share of the index
    index_share = model.doc_counts / max(model.doc_counts.sum(), 1)
    cov = []
    for j, nm in enumerate(model.names):
        if not keep[j]:
            continue
        traffic = float(M[-1, j])
        idx = float(index_share[j])
        cov.append((nm, traffic, idx, traffic / idx if idx > 1e-9 else float("inf")))
    cov.sort(key=lambda r: -r[3])

    return DriftReport(
        topics=[n for j, n in enumerate(model.names) if keep[j]],
        mixtures=M,
        risers=risers,
        fallers=fallers,
        crossovers=crossovers,
        coverage=cov,
        assigned_share=assigned,
        min_confidence=min_confidence,
    )


def coverage_gap(report: DriftReport, ratio_threshold: float = 1.5) -> list[str]:
    """Topics whose share of traffic has outrun their share of the index.

    This is the actionable output: what to go and index more of.
    """
    return [nm for nm, _, _, r in report.coverage if r >= ratio_threshold]
