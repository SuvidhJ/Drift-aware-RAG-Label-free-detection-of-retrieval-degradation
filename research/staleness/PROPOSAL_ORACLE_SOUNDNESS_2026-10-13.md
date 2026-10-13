# Proposal: the knowledge-conflict oracle of RAG metamorphic testing is unsound, and how to fix it

> **CORRECTED 2026-10-13 20:40. `CLAIMS_LOCKED_2026-10-13.md` is authoritative.**
> - **D3 is WITHDRAWN.** ASE's numeric answers are a single JSON value, so the "last number"
>   parser never sees lists in their pipeline.
> - **The order table in §3 is WITHDRAWN.** It was contaminated by D3.
> - **D2:** the effect on ASE's published rates is UNMEASURED.
> - **D1** applies only to CN mutants labelled semantics-altering, a share that is not released.
> - The repair rule (A3) and the paper–code inconsistency (A4) are added, both verified.
> - p14 has been amended before running: the primary measure is now the committed value.

2026-10-13. For: Aditi Saxena, Suvidh Jain, Aryan Garg.
Status:
- the gap is code-verified;
- the evidence so far is observational;
- a randomized confirmation (`pretests/p14_order_oracle.py`) is pre-registered (sha256 in
  `pretests/p14_prereg.log`) and **not yet run**.

## 1. Thesis in one sentence
The oracle that the ASE 2026 framework (Kim, Pasini & Tonella, arXiv 2607.26843) uses for its
headline operator (CN, knowledge-conflict mutation) is **unsound**:
- its verdicts depend on *where* the counterfactual is placed and *which number an answer lists
  last*;
- on four of its five datasets, every reported CN "fault" is by construction a *correct* answer.

We show this from their released code and quantify it on real Wikipedia revisions. We then
replace it with **decidable, order-invariant, provenance-controlled metamorphic relations**.

## 2. The gap, from the released artefact (github.com/dbr7/RAG-metamorphic-mutation)
Copies are in `verification_2026-10-13/ase_artifact/`.

| # | Defect | Evidence (file:line) | Consequence |
|---|---|---|---|
| D1 | On the four numeric datasets the semantics-altering verdict is `SATISFIED if not answers_match else VIOLATED`, where match = the answer equals the *original* answer | `step3_mutation_campaign.py` L746–748 | Step 2 keeps only questions whose original answer equals the ground truth. So **every numeric CN violation is a correct answer** given despite evidence the code generates to "mislead a reader into giving a wrong answer" (`mutation_operators.py` L551–556). **Any changed answer passes**, including hallucinated numbers. |
| D2 | Post-chunk CN appends the counterfactual *after* the original: `c.content = c.content + " " + contradiction` | `mutation_operators.py` L651 | Placement is fixed, and LLMs have documented position biases (first-evidence preference: 2601.06189; EMNLP 2026 2609.03148). The reported rate is the rate at **one arbitrary position**. |
| ~~D3~~ WITHDRAWN | The parser strips commas and takes the **last** number: `float(matches[-1])` | `pipeline_config.py` L760–770 | For answers listing both values, the verdict is decided by the *order* of mention. "36.77,29.86" parses as 86. "million/billion" are stripped, so unit conversions mismatch. |
| D4 | The CN fault is defined two ways: "signalling or resolving the discrepancy" (fault taxonomy, §3.2.1) vs "y′ aligns with the injected fact" (relation, §4.5). The RQ1 example counts *ignoring* the counterfactual as a fault; the RQ3 example counts *adopting* it ("directly opposed the ground truth") as a fault that persisted. | paper §3.2.1, §4.5, §6.1, §6.3 | Without provenance (which value is newer or valid), the correct behaviour is **undecidable**, so the relation is not a valid metamorphic relation. |
| D5 | Meta-evaluation F1 = 1.000 on the numeric datasets | notebook cell 10: MR FP = 0, FN = 0 on all four; the paper admits "no room for classification error" | The headline "F1 0.927–1.000" is tautological on 4 of 5 datasets. |

## 3. ~~Evidence so far~~ WITHDRAWN (contaminated by D3). See CLAIMS_LOCKED §C1 for the clean, smaller, model-dependent order effects
This applies ASE's oracle **verbatim** to our existing p11 (real revisions) and p12 (CN) answers.
"Before/after" is the retrieved order of the mutation passage relative to the original:

| Model | Source | n | ASE violation, mutation *before* | ASE violation, mutation *after* (ASE's placement) |
|---|---|---|---|---|
| Qwen2.5-7B (Q4) | CN | 697 | **0.705** | **0.220** |
| Qwen2.5-7B | real | 605 | 0.527 | 0.225 |
| phi3-mini (Q4) | CN | 697 | 0.473 | 0.407 |
| phi3-mini | real | 605 | 0.357 | 0.309 |
| Qwen2.5-14B (Q4) | real | 605 | 0.105 | 0.023 |

Also:
- answers that match *neither* value are scored "satisfied" (Qwen-7B CN: 86 of 370 "satisfied");
- answers listing both values are decided by the last-listed number (Qwen-7B CN: 157 "violations").

Caveat: the order here came from retrieval and was not randomized. p14 tests it with
randomized order.

## 4. Contributions of the paper (if p14 supports H1)
1. **Oracle-soundness audit** of a published RAG metamorphic-testing framework: D1–D5, each
   verified from code. *Novel*: we found no work auditing RAG metamorphic oracles. Position bias
   in LLMs is known; its effect on *published testing verdicts* is not.
2. **Quantification:** how much of the reported CN violation rate is an oracle artefact, across
   orders, parsers and models, on real revisions and on CN.
3. **A sound replacement**, a small set of decidable relations:
   - **MR-Order (invariance):** permuting the conflicting passages must not change the committed
     value. Decidable without ground truth.
   - **MR-Provenance (variance):** give both passages explicit validity dates. The answer must
     follow the newer one, and must *flip* when only the dates are swapped. Decidable without
     ground truth.
   - **Value-level four-way oracle** (new / old / both / neither), with deterministic matching
     that is robust to lists, units and years. Validated by 2 human annotators (Suvidh, Aryan)
     with Cohen's κ.
4. **Real-revision validation:** real Wikipedia revisions supply *natural* provenance, so
   MR-Provenance is checked against real updates as well as synthetic CN. This reuses our
   verified pipeline.

**Why this is resource-light and robust.** The core claim is about the **oracle**, not the
models. D1, D3 and D4 are logical and code facts that hold for any model. The models only
supply the *magnitude* of D2. Small local models are therefore not a weakness for the main
claim (they are for external validity of the magnitudes, which we state).

## 5. Positioning (what is known; what we add)
- **Known:**
  - LLM order and position bias in conflicts (2601.06189; 2609.03148; Stable-RAG, ACL 2026,
    2601.02993);
  - permutation invariance as a classic ML-testing MR;
  - outdated-information harm (HoH, SDP);
  - desired behaviour per conflict type (DRAGged).
- **We add:**
  - the published RAG-testing oracle for conflicts is unsound and order-dependent (D1–D5);
  - a measured size of the artefact;
  - provenance-controlled, decidable relations, validated on real revisions.

The SE framing is the classic **test-oracle problem** and **MR validity**.

## 6. Shortest safe path
| Step | Cost | Go/kill |
|---|---|---|
| 0. Code audit and observational evidence | done (0 compute) | done |
| **1. p14 randomized order test** (120 questions × {REAL, CN} × 2 orders + adequacy; phi3 + Qwen-7B; 1,200 prompts) | about 20–40 min local | **H1 supported** (\|Δ\| ≥ 0.10, McNemar p < 0.01) → continue. **Not supported** (all \|Δ\| < 0.05) → drop D2 and keep D1, D3–D5 as a short paper or comment. |
| 2. Full study: MR-Order and MR-Provenance on all 392 eligible questions; 3 models; human labels on 200 cases; optionally ASE's pipeline end-to-end with a local model via an OpenAI-compatible Ollama endpoint, on one T2-RAGBench dataset | about 1–2 weeks (14B about 2–3 h of compute) | report whichever way |
| 3. Write-up; release the tool and data | about 2 weeks | — |

**Before submission:** email the ASE authors (Kim, Pasini, Tonella) the defect list. This is
standard collegial practice; they may confirm or fix it, and it protects us against "unfair
critique" reviews.

## 7. Venues (check the current calls)
- **Natural fit (test oracles):** software-testing venues, e.g. ICST and ISSTA, or the journal
  STVR.
- **Journals:** JSS / IST.
- **Top tier:** an ASE or ICSE research paper is plausible only with step 2 complete, including
  the end-to-end ASE replication.

## 8. Risks
| Risk | Mitigation |
|---|---|
| "A critique of one paper" | generalise to MR-soundness criteria for any conflict mutation; release the decidable MRs as a tool |
| The order effect does not replicate under randomization (p14) | the code defects D1, D3–D5 stand regardless; the paper becomes shorter |
| ASE authors patch the artefact | we cite the version (commit at 2026-07-29 push); the soundness criteria remain the contribution |
| Small 4-bit models | the core claims are model-independent; magnitudes are stated with their scope |

---

## 9. Journal-level package (added 2026-10-13 21:00, after the re-check; claims per CLAIMS_LOCKED)

**Working title:** "When the oracle is the fault: validity of metamorphic relations for
knowledge-conflict testing of RAG systems".

| # | Contribution | Status | Effort |
|---|---|---|---|
| 1 | **Validity criteria** for knowledge-conflict MRs: (V1) *decidable*, i.e. the follow-up input determines the expected output; (V2) *value-aligned*, i.e. the oracle checks agreement with the injected or updated value, not mere change; (V3) *placement-invariant*; (V4) *provenance*: a variance relation needs validity information (dates or supersession), otherwise only "acknowledge or abstain" is checkable | to write | 2–3 days |
| 2 | **Audit** of the ASE 2026 framework against V1–V4: A1–A6, A8, A9, all code-verified, with a re-runnable checker (29/29) | **done** | — |
| 3 | **Measured impact on the authors' own pipeline.** Run their artefact locally with three disclosed patches: CN generator → `provider="ollama"`; embeddings → a local model; LLM → `--base-url` Ollama. CN only, pre- and post-chunk, about 100 questions each of FinQA (T2-RAGBench, HuggingFace) and RepliQA (HuggingFace, CC-BY-4.0). Measure: (a) the share of "satisfied" answers that match **neither** the ground truth nor the counterfactual; (b) post-chunk placement **append vs prepend**, randomized (a one-line variant), giving the true placement effect in *their* setting; (c) how many verdicts and "repairs" flip under a value-aligned oracle | feasible (code supports Ollama); about 1 week including overnight runs | 1 week |
| 4 | **Sound replacement**, a drop-in for their artefact: value-aligned four-way oracle; MR-Order (invariance); MR-Provenance-Swap (dates swapped → answer must flip). Evaluated on their datasets *and* on our real Wikipedia revisions (natural provenance; pipeline already verified) | to build | 1 week |
| 5 | **Human validation:** 200 stratified cases labelled by Suvidh and Aryan; Cohen's κ against both oracles | to do | 2–3 days of their time |
| 6 (optional) | **Generality:** audit one more RAG metamorphic-testing framework against V1–V4 (e.g. MetaRAG; retromorphic testing, 2603.27752) | reading only | 1–2 days |

**Minimum for a journal** (JSS / IST / STVR): items 1 + 2 + 3 + 4. Items 5 and 6 strengthen it.
**Not enough alone:** item 2 (critique only), which suits a short or negative-results track.
**Before submission:** re-run the novelty check; email the ASE authors the defect list; agree
authorship.
