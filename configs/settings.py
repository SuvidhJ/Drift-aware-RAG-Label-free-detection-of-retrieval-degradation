"""Configuration.

Every field is overridable by environment variable using the section prefix, e.g.
``DRIFT_ALPHA_LEVEL=0.01``, ``EMBED_BACKEND=sentence-transformers``,
``DATA_CORPUS=python_docs``.  A ``.env`` file is read if present.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parent.parent


class DataConfig(BaseSettings):
    """Corpus selection and windowing.

    The default source is a real revision corpus: documents observed at two real
    timepoints, with real edit timestamps.  The previous default (Amazon Reviews
    2023) is still reachable via ``source="amazon"`` but is not recommended -- its
    query stream was a single constant string and its snapshots were near-identical
    across windows (FINDINGS.md F3, F4).
    """

    model_config = SettingsConfigDict(env_prefix="DATA_", env_file=".env", extra="ignore")

    source: Literal["revision", "amazon"] = "revision"
    corpus: Literal["arxiv", "python_docs", "wikipedia"] = "arxiv"
    corpus_dir: Path = REPO_ROOT / "research" / "gates" / "results"

    n_windows: int = Field(default=12, ge=3, description="Number of time windows T")
    min_window_samples: int = Field(
        default=200,
        ge=2,
        description="Windows with fewer live documents than this are dropped, not silently kept",
    )
    max_docs: int | None = Field(default=None, description="Cap corpus size for fast runs")
    queries_per_window: int = Field(default=300, ge=10)
    query_targeted_frac: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of queries about already-revised documents that are drawn from "
            "text existing only in the revision. At 0 the stale index answers almost "
            "everything and retrieval headroom is ~0, which makes selective re-index "
            "unmeasurable."
        ),
    )

    snapshots_dir: Path = REPO_ROOT / "data" / "snapshots"
    raw_dir: Path = REPO_ROOT / "data" / "raw"
    seed: int = 20260907

    # legacy Amazon path only
    hf_repo: str = "McAuley-Lab/Amazon-Reviews-2023"
    category: str = "Cell_Phones_and_Accessories"
    null_category: str = "Musical_Instruments"
    max_products: int = 20_000
    max_reviews: int = 100_000
    window_unit: Literal["monthly", "weekly"] = "monthly"


class EmbeddingConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="EMBED_", env_file=".env", extra="ignore")

    backend: Literal["minilm-numpy", "sentence-transformers"] = "minilm-numpy"
    model_name: str = Field(default="BAAI/bge-m3", description="sentence-transformers backend only")
    model_dir: Path | None = Field(default=None, description="Override the MiniLM checkpoint location")
    device: str = "cpu"
    batch_size: int = 32
    max_len: int = 256
    cache_dir: Path = REPO_ROOT / "embeddings" / "cache"
    normalize: bool = True


class DriftConfig(BaseSettings):
    """Drift statistics and their calibration.

    ``threshold_tau`` survives only for the legacy fixed-threshold path used by the
    transfer experiment to demonstrate that it fails.  The live alarm path keys off
    ``tau_z_*``, which are z-scores against a per-window permutation null.
    """

    model_config = SettingsConfigDict(env_prefix="DRIFT_", env_file=".env", extra="ignore")

    # estimators
    w1_n_projections: int = Field(default=256, ge=32)
    w1_bootstrap_iters: int = Field(default=200, ge=0)
    mmd_subsample: int = Field(default=800, ge=50)
    mmd_sigma: float | None = None

    # calibration
    n_perm: int = Field(default=199, ge=19, description="Permutations per window per statistic")
    null_n_projections: int = Field(
        default=64, ge=16, description="Projections inside the permutation null (kept low; it is called n_perm times)"
    )
    alpha_level: float = Field(default=0.05, gt=0.0, lt=0.5)

    # D(t) weights, applied to null-standardised z-scores
    w_query: float = Field(default=1.0, ge=0.0)
    w_corpus: float = Field(default=1.0, ge=0.0)

    # regime classification, in null-standardised units
    tau_z_query: float = Field(default=2.0, description="z above which query drift is called")
    tau_z_corpus: float = Field(default=2.0, description="z above which corpus drift is called")
    persistence_gate: int = Field(default=3, ge=1)
    lag: int = Field(default=1, ge=1)

    # legacy, for the transfer demonstration only
    threshold_tau: float = 0.15

    results_dir: Path = REPO_ROOT / "results"


class PolicyConfig(BaseSettings):
    """Costs are compute multiples; regret is expressed in the same units.

    The previous engine mixed a faithfulness fraction in [0, 1] with compute
    multiples in [1, 110] at lambda = 1, which made every action but no-op
    unreachable (FINDINGS.md F6).  ``regret_scale`` converts a unit of lost
    retrieval quality into compute-equivalent units so the argmin is meaningful.
    """

    model_config = SettingsConfigDict(env_prefix="POLICY_", env_file=".env", extra="ignore")

    lam: float = Field(default=1.0, ge=0.0)
    regret_scale: float = Field(
        default=100.0,
        description="Compute-equivalent cost of losing all retrieval quality for one window",
    )
    cost_no_op: float = 1.0
    cost_rerank: float = 4.0
    cost_reindex: float = 22.0
    cost_finetune: float = 110.0
    cost_escalate: float = 1_000.0
    reindex_budget: float = Field(default=0.20, gt=0.0, le=1.0)


class RetrievalConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="RETRIEVAL_", env_file=".env", extra="ignore")

    backend: Literal["numpy", "chroma"] = "numpy"
    chroma_path: Path = REPO_ROOT / "retrieval" / "chroma_db"
    top_k: int = 10
    mmr_lambda: float = 0.5
    mmr_fetch_k: int = 20

    # selective re-index (action a3)
    selector: Literal["probe_knn", "cluster_threshold", "random", "oracle", "changelog"] = "probe_knn"
    probe_frac: float = Field(default=0.02, gt=0.0, le=1.0)
    knn_k: int = 15
    knn_power: int = 8
    n_clusters: int = 40


class GenerationConfig(BaseSettings):
    """Answer generation is optional.

    The primary endpoint is retrieval quality (recall@k / MRR against a known gold
    document), because RAGAS faithfulness scores answer-to-context consistency and
    therefore scores stale-but-plausible context well -- it is blind to the failure
    mode this project is about (FINDINGS.md F17).
    """

    model_config = SettingsConfigDict(env_prefix="OPENAI_", env_file=".env", extra="ignore")

    enabled: bool = False
    api_key: str = "local"
    base_url: str = "http://127.0.0.1:1234/v1"
    generation_model: str = "meta-llama-3.1-8b-instruct"
    temperature: float = 0.0
    max_tokens: int = 512
    timeout_s: float = 60.0


class LogConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LOG_", env_file=".env", extra="ignore")

    level: str = "INFO"
    sink: str = "logs/drift_rag.log"
    rotation: str = "10 MB"
    enabled: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data: DataConfig = Field(default_factory=DataConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    log: LogConfig = Field(default_factory=LogConfig)

    # kept so `settings.openai` still resolves for any external caller
    @property
    def openai(self) -> GenerationConfig:
        return self.generation


settings = Settings()
