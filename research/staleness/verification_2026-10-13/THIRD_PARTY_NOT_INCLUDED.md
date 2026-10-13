# Third-party material not included in the repository

Two folders used by `check_claims.py` are kept locally and are **not** committed: we do not
redistribute other authors' papers or unlicensed code.

- `lit/`: plain-text extractions of the cited arXiv papers. Re-create them from
  `https://arxiv.org/html/<id>` (ids are listed in `../CLAIMS_LOCKED_2026-10-13.md`), using
  `clean.py`-style HTML-to-text extraction.
- `ase_artifact/`: copies of files from github.com/dbr7/RAG-metamorphic-mutation at commit
  **597fd6582b** (2026-07-29; the repository has no licence file). Re-create them with
  `git clone https://github.com/dbr7/RAG-metamorphic-mutation && git -C RAG-metamorphic-mutation checkout 597fd6582b`,
  and copy `mutation_operators.py`, `pipeline_config.py`, `step2_baseline_rag_filter.py`,
  `step3_mutation_campaign.py`, `step4_fault_repair.py`, `prompts/t2ragbench.txt` and
  `supplemental/rq3_repair/rq3_rerank_select_repair.py` into `ase_artifact/`.

The authors' raw outputs used by `../oracle_audit/analyze_ase_outputs.py` are their Figshare
archive (https://figshare.com/s/96fc3abb168180c0a215). Download it in a browser and unzip it to
`~/SemProject/ase-replication/output`.
