"""
Embedding-space world simulator for drift-aware RAG.

Goal: reproduce, in a controllable and cheap form, the *geometry* that the real
pipeline (BGE-M3 embeddings of Amazon products + questions) actually presents to
the drift detector, so that every downstream claim can be tested against known
ground truth.

Design choices, and why each matches the real setting
-----------------------------------------------------
* Unit-sphere embeddings.  Real pipeline L2-normalises BGE-M3 output, so cosine
  geometry on S^{d-1} is the right space.  We use d=256 (vs 1024) purely for
  speed; every phenomenon of interest (projection dilution, sample-size bias of
  W1, centroid-gap being a *level*) is dimension-robust and present at d=256.

* Sub-category cluster structure.  A catalogue is a union of product lines.  Each
  line is a near-vMF blob around a random mean direction.  Cluster sizes follow a
  Zipf-ish law (a few big lines, many small) — matches real catalogues.

* Queries live *near* the corpus, not on it.  A question about a product is that
  product's embedding plus noise, renormalised.  This creates the real, large,
  roughly-constant gap  ||mean(Q) - mean(C)||  that the original D(t) mis-uses as
  a drift term (it is a *level*, not a *change*).

* Two injectable drift channels, matching the README's own taxonomy:
    - corpus drift  : at t_event, delist a fraction f of products, concentrated
                      in one "discontinued" product line (realistic: a line is
                      EOL'd).  Queries about that line keep arriving.
    - query   drift : from t_event on, a fraction r of that window's questions are
                      about a *held-out* line H that is NOT in the corpus (new
                      products asked about before they are indexed).
  Crossing f x r gives the 2x2 factorial the project never ran.

* Seasonality confound.  |Q_t| varies window to window (multiplicative lognormal).
  Real monthly question volume swings by multiples.  This is what makes a raw,
  un-calibrated distance partly a proxy for sample size.

Everything is seeded; `World(seed).run(spec)` is a pure function of (seed, spec).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

Regime = Literal["stable", "query_drift", "corpus_drift", "joint_drift"]


# --------------------------------------------------------------------------- #
# geometry helpers
# --------------------------------------------------------------------------- #
def _normalise(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _blob(mean_dir: np.ndarray, rel_spread: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Points around `mean_dir` on the unit sphere.

    `rel_spread` is the RMS perturbation norm *relative to the unit signal*, so
    per-coordinate std = rel_spread / sqrt(d).  rel_spread ~ 0.3 => tight line,
    ~0.8 => diffuse.  This keeps the geometry sane in high d (a naive
    per-coordinate sigma makes the noise norm ~ sigma*sqrt(d) and drowns the
    signal, turning "clusters" into near-uniform noise).
    """
    d = mean_dir.shape[0]
    per = rel_spread / np.sqrt(d)
    return _normalise(mean_dir[None, :] + per * rng.standard_normal((n, d)))


def _perturb(bases: np.ndarray, rel: float, rng: np.random.Generator) -> np.ndarray:
    """Perturb a stack of unit vectors by relative RMS `rel`, renormalise."""
    d = bases.shape[1]
    return _normalise(bases + (rel / np.sqrt(d)) * rng.standard_normal(bases.shape))


# --------------------------------------------------------------------------- #
# specification of one experimental run
# --------------------------------------------------------------------------- #
@dataclass
class RunSpec:
    d: int = 256
    n_clusters: int = 20                # product lines in the catalogue
    n_products_0: int = 8_000           # catalogue size at t0
    line_spread: float = 0.45           # RMS within-line spread, relative to unit signal
    query_noise: float = 0.55           # RMS question-vs-product perturbation, relative to unit signal
    T: int = 12                         # number of monthly windows
    q_per_window_med: int = 2_000       # median questions / window
    q_volume_logstd: float = 0.55       # seasonality: lognormal sigma on |Q_t|
    organic_churn: float = 0.01         # fraction of catalogue added & removed each window (noise)
    query_stale_frac: float = 0.50      # share of questions about products seen earlier (may now be delisted)

    # --- injected drift ---
    t_event: int = 5                    # window at which injections switch on
    corpus_delist_lines: int = 0        # f : number of WHOLE product lines EOL'd at t_event
    corpus_delist_frac: float = 0.0     # optional extra uniform-random delisting on top
    query_newline_frac: float = 0.0     # r : fraction of post-event questions about held-out line(s)
    query_newline_count: int = 1        # how many held-out lines the new questions target
    query_newline_ramp: bool = False    # if True, r ramps 0 -> query_newline_frac linearly over t_event..T-1

    seed: int = 0

    def regime(self) -> Regime:
        f = (self.corpus_delist_lines > 0) or (self.corpus_delist_frac > 1e-9)
        r = self.query_newline_frac > 1e-9
        if f and r:
            return "joint_drift"
        if r:
            return "query_drift"
        if f:
            return "corpus_drift"
        return "stable"


@dataclass
class WindowData:
    t: int
    corpus: np.ndarray          # (n_c, d)   product embeddings live at window t
    queries: np.ndarray         # (n_q, d)   question embeddings asked in window t
    corpus_ids: np.ndarray      # (n_c,)     stable product id for each corpus row
    query_target_ids: np.ndarray  # (n_q,)   product id each question is "about" (-1 if held-out line)
    drifted: bool               # ground-truth: is this window past t_event with a live injection
    regime: Regime              # ground-truth regime label for this window
    unanswerable_frac: float = 0.0  # share of this window's questions with no live target (oracle harm)


@dataclass
class SimResult:
    spec: RunSpec
    windows: list[WindowData] = field(default_factory=list)
    # bookkeeping useful for the KR-bound / retrieval experiments
    cluster_means: np.ndarray | None = None
    heldout_mean: np.ndarray | None = None
    delisted_ids: np.ndarray | None = None


class World:
    def __init__(self, spec: RunSpec):
        self.spec = spec
        self.rng = np.random.default_rng(spec.seed)

    # ------------------------------------------------------------------ #
    def _build_catalogue(self):
        s = self.spec
        rng = self.rng
        # extra held-out directions so query drift can target multiple new lines
        n_held = max(1, s.query_newline_count)
        cluster_means = _normalise(rng.standard_normal((s.n_clusters, s.d)))
        heldout_means = _normalise(rng.standard_normal((n_held, s.d)))  # never indexed

        w = 1.0 / np.arange(1, s.n_clusters + 1)
        w = w / w.sum()
        counts = rng.multinomial(s.n_products_0, w)

        embs, cl_of = [], []
        for k, c in enumerate(counts):
            if c == 0:
                continue
            embs.append(_blob(cluster_means[k], s.line_spread, int(c), rng))
            cl_of.append(np.full(int(c), k))
        embs = np.vstack(embs)
        cl_of = np.concatenate(cl_of)
        ids = np.arange(embs.shape[0])
        return embs, cl_of, ids, cluster_means, heldout_means

    # ------------------------------------------------------------------ #
    def _make_queries(self, live_embs, live_ids, seen_embs, seen_ids, n_q, newline_frac, heldout_means):
        """Questions come from three sources:
          * new-line   (fraction newline_frac): about a held-out, never-indexed line
          * stale      (fraction query_stale_frac of the rest): about a product seen
            live at some earlier window -- may since have been delisted
          * fresh      (remainder): about a currently-live product
        """
        s = self.spec
        rng = self.rng
        n_new = int(round(n_q * newline_frac))
        rest = n_q - n_new
        n_stale = int(round(rest * s.query_stale_frac))
        n_fresh = rest - n_stale

        pf = rng.integers(0, live_embs.shape[0], size=n_fresh)
        q_fresh = _perturb(live_embs[pf], s.query_noise, rng)
        tf = live_ids[pf]

        if n_stale > 0 and seen_embs.shape[0] > 0:
            ps = rng.integers(0, seen_embs.shape[0], size=n_stale)
            q_stale = _perturb(seen_embs[ps], s.query_noise, rng)
            ts = seen_ids[ps]
        else:
            q_stale = np.zeros((0, s.d)); ts = np.zeros(0, int)

        if n_new > 0:
            which = rng.integers(0, heldout_means.shape[0], size=n_new)
            q_new = _perturb(heldout_means[which], s.query_noise, rng)
        else:
            q_new = np.zeros((0, s.d))
        tn = np.full(n_new, -1)

        q = np.vstack([x for x in (q_fresh, q_stale, q_new) if x.shape[0] > 0])
        tgt = np.concatenate([tf, ts, tn])
        perm = rng.permutation(q.shape[0])
        return q[perm], tgt[perm]

    # ------------------------------------------------------------------ #
    def run(self) -> SimResult:
        s = self.spec
        rng = self.rng
        embs, cl_of, ids, cluster_means, heldout_means = self._build_catalogue()

        alive = np.ones(embs.shape[0], dtype=bool)
        next_id = embs.shape[0]

        # lines EOL'd by the injection: the mid-sized ones (avoid the huge head
        # lines, which would be an implausibly large single event, and the tiny
        # tail, which would be undetectable)
        line_sizes = np.bincount(cl_of, minlength=s.n_clusters)
        order = np.argsort(line_sizes)              # ascending
        mid = order[len(order) // 4 : 3 * len(order) // 4]
        eol_lines = set(int(x) for x in mid[-s.corpus_delist_lines:]) if s.corpus_delist_lines > 0 else set()

        delisted_ids_all: list[int] = []
        res = SimResult(spec=s, cluster_means=cluster_means, heldout_mean=heldout_means[0])

        # rolling memory of every product ever seen live (for "stale" questions)
        seen_e = embs.copy()
        seen_i = ids.copy()

        for t in range(s.T):
            # ---- organic churn (happens every window, both arms) ----
            n_churn = int(round(alive.sum() * s.organic_churn))
            if n_churn > 0:
                live_idx = np.where(alive)[0]
                drop = rng.choice(live_idx, size=min(n_churn, live_idx.size), replace=False)
                alive[drop] = False
                # add n_churn fresh products spread over random existing lines
                add_lines = rng.integers(0, s.n_clusters, size=n_churn)
                new_e = _perturb(cluster_means[add_lines], s.line_spread, rng)
                embs = np.vstack([embs, new_e])
                cl_of = np.concatenate([cl_of, add_lines])
                ids = np.concatenate([ids, np.arange(next_id, next_id + n_churn)])
                alive = np.concatenate([alive, np.ones(n_churn, dtype=bool)])
                next_id += n_churn

            # ---- injected corpus drift: one-shot delist at t_event ----
            if t == s.t_event and (eol_lines or s.corpus_delist_frac > 1e-9):
                live_idx = np.where(alive)[0]
                in_lines = live_idx[np.isin(cl_of[live_idx], list(eol_lines))] if eol_lines else np.array([], int)
                extra_pool = np.setdiff1d(live_idx, in_lines)
                n_extra = int(round(embs.shape[0] * s.corpus_delist_frac))
                pick_extra = (rng.choice(extra_pool, size=min(n_extra, extra_pool.size), replace=False)
                              if n_extra > 0 else np.array([], int))
                delist = np.concatenate([in_lines, pick_extra]).astype(int)
                alive[delist] = False
                delisted_ids_all.extend(ids[delist].tolist())

            # ---- assemble window ----
            live_e = embs[alive]
            live_i = ids[alive]

            # keep the "seen" pool bounded and current-ish: union of live + a
            # decaying tail of previously-seen products
            seen_e = np.vstack([seen_e, live_e])
            seen_i = np.concatenate([seen_i, live_i])
            if seen_i.size > 60_000:
                keep = rng.choice(seen_i.size, 60_000, replace=False)
                seen_e, seen_i = seen_e[keep], seen_i[keep]

            vol = s.q_per_window_med * float(np.exp(s.q_volume_logstd * rng.standard_normal()))
            n_q = max(200, int(round(vol)))

            if t < s.t_event:
                newline_frac = 0.0
            elif s.query_newline_ramp:
                span = max(1, s.T - 1 - s.t_event)
                newline_frac = s.query_newline_frac * min(1.0, (t - s.t_event) / span)
            else:
                newline_frac = s.query_newline_frac
            q_e, q_tgt = self._make_queries(live_e, live_i, seen_e, seen_i, n_q, newline_frac, heldout_means)

            live_id_set = set(live_i.tolist())
            unanswerable = float(np.mean([
                (tid == -1) or (tid not in live_id_set) for tid in q_tgt
            ]))

            injected = (t >= s.t_event) and (s.regime() != "stable")
            res.windows.append(
                WindowData(
                    t=t,
                    corpus=live_e.astype(np.float32),
                    queries=q_e.astype(np.float32),
                    corpus_ids=live_i.copy(),
                    query_target_ids=q_tgt.copy(),
                    drifted=injected,
                    regime=s.regime() if injected else "stable",
                    unanswerable_frac=unanswerable,
                )
            )

        res.delisted_ids = np.array(sorted(set(delisted_ids_all)), dtype=int)
        return res


def simulate(spec: RunSpec) -> SimResult:
    return World(spec).run()
