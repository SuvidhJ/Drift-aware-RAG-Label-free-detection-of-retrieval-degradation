# Locked claims register: ASE 2026 knowledge-conflict oracle, 2026-10-13 (20:40 IST)

**Purpose.** This fixes exactly what we claim, its scope and its evidence, so the claims cannot
drift later.
- Any change to a claim must be logged at the bottom, with the reason.
- **Statuses:** VERIFIED (code or text, re-checkable), WITHDRAWN, UNMEASURED, EXPLORATORY.

**Sources, frozen.**
- Paper: arXiv 2607.26843 **v1** (the only version), full text saved at
  `verification_2026-10-13/lit/2607.26843.full.txt`.
- Code: github.com/dbr7/RAG-metamorphic-mutation, **single commit `597fd6582b`** (2026-07-29).
  The files cited are copied to `verification_2026-10-13/ase_artifact/` (step2 and step4 copies
  added with this register).
- Scope: the ASE camera-ready version may differ. We cite v1 and that commit.

---

## A. Claims about the ASE paper and artefact

### A1. VERIFIED: the implementation does not check the specified relation for CN
- **The paper specifies:**
  - §4.5: "We expect y′≠y, specifically where y′ aligns with the injected fact."
  - §8: "the GT check verifies whether y′ reflects the injected counterfactual".
- **The code checks:**
  - MR verdict for the numeric datasets (`step3_mutation_campaign.py` L746–748): semantics-altering
    → `SATISFIED if not answers_match else VIOLATED`, where `answers_match` = y′ within 1% of
    y_base.
  - GT check (L1057–1069 numeric; L1028–1055 LLM, prompt L1002–1025): compares y′ with
    `baseline.ground_truth_short`, the **original** dataset answer.
  - Meta-evaluation (L1199): `system_good = (is_preserving == (gt_correct == "correct"))`.
- **The injected counterfactual value is never compared anywhere.**

### A2. VERIFIED (logical consequence of A1 and step 2), for CN mutants labelled semantics-altering
1. **Numeric datasets (ConvFinQA, FinQA, TAT-DQA, VQAonBD):**
   - Step 2 keeps only questions with y_base within 1% of the ground truth
     (`step2_baseline_rag_filter.py` L218–221).
   - Therefore **every VIOLATED verdict is an answer within about 2% of the dataset's ground
     truth**.
   - **Every non-abstaining answer that differs from y_base is SATISFIED**, including numbers
     that match neither the ground truth nor the counterfactual. "N/A" is also SATISFIED
     (L723–731).
2. **All datasets, meta-evaluation:** an answer judged *incorrect* against the original ground
   truth counts as **good system behaviour** for these mutants (L1199). A judge score of 3
   ("idk") counts as good for all mutants (L1199).
3. **Scope limit.** Altering vs preserving is decided per mutant by an LLM judge (default
   `--semantic-check llm`, L1256; L456, L486). The *share* of CN mutants labelled altering is
   **not released**. A2 applies to that subset; its size is **UNMEASURED**.
   - For pre-chunk CN, the judge is told "injected a counterfactual passage that alters key facts
     relevant to the question" (L650–653). That suggests "altering" is the usual label, but this
     is an inference, not a measurement.

### A3. VERIFIED: the repair criterion rewards leaving the correct answer
- `step4_fault_repair.py` L390–395: for non-preserving mutations, `fault_repaired = (gt_correct != "correct")`.
- A CN fault counts as repaired when the new answer is **not** the original correct answer,
  correct or not.

### A4. VERIFIED: paper prose is inconsistent with A1–A3
- **§6.3 example:** a CN fault "persisted after repair: the stronger model still adopted the
  contradictory evidence, producing an answer that directly opposed the ground truth".
- **Under the released code:** an answer opposing the ground truth is GT-"incorrect", so
  `fault_repaired = True`.
- **§6.1 example:** the system "still answered … verbatim, ignoring the contradictory evidence"
  is called a fault.
- **§3.2.1 fault definition:** "A robust system must exhibit explicit conflict awareness,
  signalling or resolving the discrepancy". This is a third criterion, which the numeric
  implementation (A1) does not check.

### A5. VERIFIED: placement of the counterfactual
- **Post-chunk:** appended at the end of each retrieved chunk, `c.content + " " + contradiction`
  (`mutation_operators.py` L651).
- **Pre-chunk:** inserted at a random sentence position (L602–603).
- The paper does not analyse placement.
- Whether placement changes ASE's *published* rates: **UNMEASURED** (it needs their pipeline).

### A6. VERIFIED: the counterfactual is generated to mislead
`mutation_operators.py` L551–556: "should mislead a reader into giving a wrong answer".
Together with A1–A2, the numeric CN "violations" are cases in which the system was **not
misled**.

### A7. NOT OURS: numeric F1 = 1.000 is tautological
The paper admits this (footnote 6). We may cite their admission; it is **not our finding**.

### A8. VERIFIED (2026-10-13 re-check): the RepliQA judge cannot see the injected fact
- **What the judge sees.** For semantics-altering mutations it is shown the *original*
  retrieved context (`step3_mutation_campaign.py` L814: `retrieved_content_original`) or the
  original document (L858: `document_extracted`), and a one-line mutation description. It never
  sees the counterfactual text or the mutated context.
- **What it rates.** Whether the answer "differs meaningfully from the original"
  (`pipeline_config.py` L585 onward). "Acknowledges conflicting information … or hedges" is
  rated 4–5 (L607), and SATISFIED means score ≥ 3 (step3 L621).
- **Consequence.** A1 holds on all five datasets: no path in the released code checks alignment
  with the injected fact.

### A9. VERIFIED code fact, impact UNMEASURED: silent fallback in counterfactual generation
- If the LLM call or its construction fails, CN silently uses a heuristic counterfactual
  (`mutation_operators.py` L539–544 and L568–574).
- The mutation log does not record which path was used.

### Automated check
`verification_2026-10-13/check_claims.py` asserts every code line and quote above against the
frozen copies: **29/29 PASS** on 2026-10-13. Three citation line numbers were corrected (1044,
814, 554–555); no claim changed.

## F. Measured on the authors' OWN released outputs (Figshare archive, GPT-5-mini; 2026-10-13 21:15)
Script `oracle_audit/analyze_ase_outputs.py`, pre-registered (hashes in `oracle_audit/prereg.log`).
Log: `oracle_audit/run1_ase_outputs.log`. Post-hoc checks are labelled.

- **F0. Data integrity, VERIFIED.** The recount of CN violations from their verdict files
  reproduces their Table 1 in **10/10** dataset × scope cells (`posthoc_checks.log`).
- **F1. M1, pre-registered, MEASURED.** **460/461** semantics-altering CN violations are answers
  matching the dataset ground truth (Wilson 0.988–1.000), including RepliQA by their own GT judge
  (64/65).
  - *Post hoc:* the altering branch holds 461 of all **594** published CN violations, so
    **at least 460/594 = 77.4% of the CN faults counted in Table 1 are correct answers**.
- **F2. M3, pre-registered, NULL.** Pre-chunk random placement: violation rate 0.344 with the
  counterfactual *before* the ground-truth value vs 0.368 *after*; Δ −0.025, Fisher p = 0.76,
  n = 207. **Placement does not change their verdicts. The order claim is dropped.**
- **F3. M4, pre-registered, NOT SUPPORTED.** Value-aligned re-scoring disagrees with their
  verdicts on 9.6% of lookup cases (threshold 10%). On hand reading (F4), most "OTHER" answers
  follow the counterfactual in other units, so the true disagreement is *lower*: their
  "satisfied" verdicts mostly do reflect the injected value.
- **F4. M2 and M5, pre-registered, MEASURE INVALID.**
  - The pre-registered outputs were M2 = 44/278 = 0.158 (H-M2 "supported") and
    M5 = 45/52 = 0.865.
  - Hand reading of samples (`posthoc_checks.log`) shows most of these answers *follow* the
    counterfactual, through unit scaling ("28 million" → 28000000) or arithmetic. Literal
    1%-tolerance matching cannot detect that.
  - **Neither result is claimed.** Both are reported as pre-registered measures that failed
    validation.
- **F5. POST HOC, MEASURED: refusals used as mutants.**
  - **334/2,671 (12.5%)** CN "counterfactuals" are LLM refusals (e.g. "Sorry, I can't help
    create misleading counterfactuals…"). The pattern's precision was 20/20 in a random sample.
  - Range: 1.7% (RepliQA pre-chunk) to **26.7%** (TAT-DQA post-chunk).
  - Some refusals insert the *correct* answer into the context.
  - Excluding them changes the published CN rates by up to **+7.7 points** (TAT-DQA post-chunk
    0.223 → 0.300; `posthoc_refusal_rates.log`).
  - Root cause: the generator prompt asks for text that will "mislead a reader" (A6).
- **F6. POST HOC.** Pre-chunk heuristic fallback (A9): **0 detected**. The silent-fallback risk
  did not materialise in pre-chunk.

## B. Withdrawn claims (do not use)
- **B1. WITHDRAWN:** "the parser decides answers listing both values by the last number" as a
  defect of ASE's system. Their numeric answers are a single JSON `computed_formula`
  (`pipeline_config.py` L819–833), so the case does not arise in their pipeline. It arose only
  when *we* applied their parser to *our* free-text answers.
- **B2. WITHDRAWN:** "Qwen-7B CN violation 0.705 vs 0.220 by order" (and the rest of that
  table). This was contaminated by B1.

## C. Our own evidence (exploratory)
- **C1. EXPLORATORY, observational** (`verification_2026-10-13/g_order_clean.py`): the share of
  answers committing to the *original value alone*, by retrieved order of the mutation passage
  (before vs after):

  | Model | Source | Before | After |
  |---|---|---|---|
  | phi3 | real | 0.241 | 0.370 |
  | phi3 | CN | 0.343 | 0.379 |
  | Qwen-7B | CN | 0.297 | 0.177 |
  | Qwen-7B | real | 0.214 | 0.180 |
  | Qwen-14B | real | 0.075 | 0.010 |

  The direction depends on the model, and the order is not randomized.
- **C2. PRE-REGISTERED, NOT RUN:** `pretests/p14_order_oracle.py`. Hashes are in
  `pretests/p14_prereg.log`: original `1fb7d9b8…`; Amendment 1 `1339539f…`, made before any
  generation.

## D. Novelty check (2026-10-13)
- **Searched:**
  - Semantic Scholar citations of 2607.26843: 3 citing papers (2608.08944 Pair-ID; 2609.38275
    U-Fuzz; 2609.23209 STELLAR-D). Their full texts cite it **neutrally**; none critiques its
    oracle.
  - The ASE repository: 0 issues, 0 forks.
  - The same authors' companion paper 2607.18155 (Chunk Coverage): about adequacy, not this
    oracle.
  - Web searches on RAG metamorphic oracles, MR validity, and order/position bias in conflicts.
- **Known (we cite, not claim):**
  - LLM position bias in conflicts: 2601.06189; 2609.03148 (EMNLP 2026); Stable-RAG 2601.02993
    (ACL 2026).
  - Permutation invariance as an ML-testing MR.
  - MR validity as a general concern (e.g. arXiv 2105.01407).
- **Not found:** any report of A1–A4 or of a provenance-controlled, decidable replacement relation
  for knowledge-conflict mutations in RAG metamorphic testing.
- **Limits:**
  - Searches are not exhaustive.
  - Papers under review are invisible.
  - The ASE camera-ready version may differ from v1.
  - **Re-run this check immediately before submission.**

## E. Change log
- 2026-10-13 21:20: section F added (authors' outputs). Order/placement claim dropped (F2 null). M2/M5 declared invalid after hand reading (F4). Refusal finding added (F5).
- 2026-10-13 21:00: re-check. A3 confirmed in all RQ3 code paths (step4 and supplemental `evaluate_repair`). A8 and A9 added. Citation line numbers fixed by `check_claims.py`.
- 2026-10-13 20:40: register created. B1 and B2 withdrawn after tracing their pipeline. A2
  narrowed to the altering-labelled subset. A5 narrowed: effect on published rates is
  unmeasured.
