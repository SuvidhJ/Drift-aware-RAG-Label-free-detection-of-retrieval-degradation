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
  - **CORRECTED 2026-10-17 → 694/2,671 (26.0%)** CN "counterfactuals" are LLM refusals
    (`oracle_audit/posthoc_refusal_v2.log`). The original 334/2,671 (12.5%) undercounted:
    the first pattern matched only the straight apostrophe in "can't", and most refusals
    use "can’t" (U+2019). Precision check of the v2 pattern: 30/30. Recall check: 0/60
    refusals among random non-matches.
  - (Superseded text follows.) 334/2,671 (12.5%) CN "counterfactuals" are LLM refusals (e.g. "Sorry, I can't help
    create misleading counterfactuals…"). The pattern's precision was 20/20 in a random sample.
  - Range, CORRECTED: 3.7% (RepliQA pre-chunk, 8/217) to **47.7%** (TAT-DQA post-chunk, 143/300).
    Post-chunk: 34.7–47.7% on four of five datasets. (Superseded: 2.3% to 26.7%.)
  - Some refusals insert the *correct* answer into the context.
  - Effect, CORRECTED (`oracle_audit/posthoc_refusal_v2_rates.log`):
    - their judge labelled 677/694 refusals as semantics-*preserving*;
    - 664 were SATISFIED;
    - excluding refusals raises the published CN rates by up to **+17.2 points** (TAT-DQA
      post-chunk 0.223 → 0.395).
  - (Superseded: excluding them changes the published CN rates by up to +7.7 points, TAT-DQA post-chunk
    0.223 → 0.300; `posthoc_refusal_rates.log`).
  - Root cause: the generator prompt asks for text that will "mislead a reader" (A6).
- **F6. POST HOC.** Pre-chunk heuristic fallback (A9): **0 detected**. The silent-fallback risk
  did not materialise in pre-chunk.

## G. p15 MR-Provenance-Swap: FINAL pre-registered results (run finished 2026-10-14 04:06; `pretests/p15_final_report.log`)
392 questions, 145 pages; 3 models; hash `508061cf…` (matches the pre-registration).
Adequacy: phi3 0.990, Qwen-7B 0.806, Qwen-14B 0.901 (all ≥ 0.60).

| Model | Source | Published relation violation | MR-PS violation | φ | PASS / INSENS / INV / ACK (of 784) |
|---|---|---|---|---|---|
| phi3 | REAL | 0.301 [0.258, 0.348] | 0.513 [0.478, 0.548] | −0.161 | 170 / 387 / 15 / 212 |
| phi3 | CN | 0.347 [0.302, 0.395] | 0.509 [0.474, 0.544] | −0.147 | 221 / 382 / 17 / 164 |
| Qwen-7B | REAL | 0.082 [0.058, 0.113] | 0.413 [0.379, 0.448] | 0.075 | 90 / 313 / 11 / 370 |
| Qwen-7B | CN | 0.089 [0.065, 0.122] | 0.356 [0.323, 0.390] | 0.110 | 98 / 250 / 29 / 407 |
| Qwen-14B | REAL | 0.015 [0.007, 0.033] | 0.017 [0.010, 0.028] | 0.323 | 97 / 11 / 2 / 674 |
| Qwen-14B | CN | 0.010 [0.004, 0.026] | 0.005 [0.002, 0.013] | −0.009 | 121 / 3 / 1 / 659 |

- **H1 SUPPORTED.** Pooled PS violation: phi3 0.511, Qwen-7B 0.385, Qwen-14B 0.011.
- **H2 NOT SUPPORTED** (the rule required φ < 0.30 for every model).
  - The Qwen-14B REAL φ is 0.323, computed on only 6 published-relation violations and
    6 PS violations (cells a=2, b=4, c=4, d=382).
  - *Post hoc* (not a rescue): with so few events φ is unstable. H2 holds for phi3 and
    Qwen-7B.
- **H3: SAME ORDER** (14B < 7B < phi3 under both relations). There is no rank reversal; the
  relations differ in magnitude for the small models.
- **H4: 2 of 3 within 0.05.** phi3 −0.004 [−0.043, +0.037]; Qwen-7B −0.057 [−0.087, −0.028];
  Qwen-14B −0.011 [−0.024, −0.001].
- **Descriptive.** With dated conflicts, Qwen-14B acknowledges or lists both values in 86%
  (REAL 674/784) and 84% (CN 659/784) of pairs. It commits to the later-dated value alone in
  only 12–15% of pairs. Neither relation counts this as a fault. The prompt allows "number(s),
  separated by commas", which may induce listing: an untested confound, noted as a limitation.
- **Post hoc** (p15 interim analysis, phi3/Qwen-7B): on PS-violating questions, the published
  relation mostly saw a commitment to the mutated value and counted it as a pass (Qwen-7B
  REAL 142/208; phi3 REAL 138/214).

## H. p16 RQ5 strategies: PILOT (2026-10-17; pre-registered gate; `pretests/p16_pilot.log`)
qwen2.5:7b, 20 pilot questions (disjoint from the planned main sample).

| Strategy | REAL PS-pass | PS-viol | ACK | CN-viol (published) | CN-adopt |
|---|---|---|---|---|---|
| BASE | 0.075 | 0.550 | 0.375 | 0.100 | 0.325 |
| ONE | 0.225 | 0.750 | 0.025 | 0.250 | 0.575 |
| CAREFUL | 0.150 | 0.525 | 0.325 | 0.000 | 0.275 |
| GULLIBLE | 0.200 | 0.650 | 0.150 | 0.050 | 0.450 |

- **GATE: STOP.** PS-pass(CAREFUL) − PS-pass(GULLIBLE) = −0.050, below the 0.15 threshold.
  The MAIN run was NOT executed. RQ5 is not answered.
- **Exploratory only (n = 20):**
  - a prompt-level date instruction did not make Qwen-7B follow dates;
  - the single-answer prompt removed listing (ACK 0.375 → 0.025) but increased date-insensitivity
    (0.55 → 0.75).
- **Follow-up:** a separately pre-registered p17 tests prompt-induced listing for Qwen-14B.

## I. p17: Qwen-14B single-answer prompt (pre-registered; finished 2026-10-17 13:03; `pretests/p17_final_report.log`)
392 questions, REAL, dates CORRECT/SWAPPED × 2 orders = 784 pairs; hash `3edbffd9…` (matches).

| Prompt | ACK | PASS | INSENSITIVE | INVERTED |
|---|---|---|---|---|
| BASE (p15) | 674 (0.860) | 97 | 11 | 2 |
| ONE | 247 (0.315) | 464 (0.592 [0.557, 0.626]) | 73 | 0 |

- **D1 SUPPORTED.** Listing is substantially prompt-induced: ACK drops by 0.545. Paired:
  443 BASE-only vs 16 ONE-only ACK; McNemar p = 1.98e-109.
- **D2 MIXED** (pre-registered rule). PASS 0.592 is just below the 0.60 bar; PS-viol 0.093 is
  below 0.30. Under forced choice, Qwen-14B follows the later date in 59% of pairs and ignores
  dates in 9%, versus 41–51% PS-viol for the two smaller models under the BASE prompt.

## J. Adversarial re-verification before publication (2026-10-17)
- **A1, codebase-wide.** `git grep` over all 17 tracked Python files of the artefact. The
  counterfactual text appears only in mutation logs, in a generic judge-prompt sentence with no
  value, in one read that is deliberately unused (step3 L651–653, "just note the action"), and in
  a corpus-replay script. No verdict, correctness or repair computation compares the answer with
  the injected value. **HOLDS.**
- **A4, now DATA-BACKED** (`oracle_audit/verify_paper_examples.log`):
  - **§6.1 ZenSpace Yoga** (pre-chunk, altering): answer unchanged and ground-truth correct;
    verdict VIOLATED; classified TP.
  - **§6.3 InspirePay "currency conversion"** (post-chunk, altering): the original answer was
    correct, and the verdict VIOLATED (TP). The GPT-5.4 repair adopted the counterfactual
    ("lacks currency conversion"), so `repaired_gt_correct = incorrect` and
    **`is_repaired = True`**. The paper describes the same case as a fault that "persisted after
    repair". The paper text contradicts the authors' own released data. **HOLDS.**
- **F5 wording.** Of the 694 refusal texts:
  - most decline and offer help;
  - about 33% restate the document's actual value (keyword heuristic, 227/694);
  - few append a labelled fictional example (the heuristic "substitute" matches, 182, are mostly
    *offers* to do so; samples read in `oracle_audit/posthoc_refusal_substitutes.log`).

  The paper's wording was revised to "refusal rather than a counterfactual", with this
  composition stated.
- **Wording audit.** "correct answers" was replaced by "answers that match the (dataset's)
  ground truth" in the paper, highlights and conclusion.

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


### D2. Novelty re-check, 2026-10-15
- **Source unchanged.** arXiv 2607.26843 is still v1 only. The code has still a single commit,
  `597fd6582b`, and 0 issues/PRs.
- **Citations unchanged.** Semantic Scholar lists the same 3 citing papers (2608.08944,
  2609.23209, 2609.38275), all neutral. OpenAlex lists 0.
- **arXiv sweep** (7 keyword queries, papers since 2026-07-29; 29 unique hits). None reports
  A1–A9, F1, F5 or a provenance-swap relation for RAG testing. Related work to add:
  - **2608.20685** (Temporal Validity on Real Software Histories): without markers, RAG
    "cannot tell which is current"; stale value 36–38%. Supports V1/V4; it is a memory-system
    paper.
  - **2610.14443** (LEST): automated failure detection in LLM safety testing "substantially
    overestimates behavioral failures" (6.1% confirmed). General oracle-validity concern; a
    different domain.
  - **2609.17081** (EviScope): paired counterfactual evidence diagnostics. An evaluation
    benchmark, not oracle validity.
  - **2609.16073** (GC-Mem): agent-memory conflict resolution. A method, not testing.
  - **ICSME 2025** (MT of LLMs for NLP; Cho et al.): MT oracle violations for LLMs include many
    false positives. General; to cite.
- **Web searches:** no critique, replication or external review of the ASE paper found.
- **OPEN ITEM, must be done before submission.** The paper is now PUBLISHED: ASE 2026
  proceedings, DOI 10.1145/3832783.3837543, pp. 3140–3151, 2026-10-07, CC BY 4.0. Our evidence
  cites arXiv v1 and code commit 597fd6582b. ACM blocks automated download (HTTP 403), so the
  user must download the PDF manually. Then diff it against v1 on the quoted passages (the §4.5
  relation; §5.2.3 judge rubric; §6.1 and §6.3 examples; Table 1; footnote 6) before citing.


### D3. Version of record verified, 2026-10-16 (closes the D2 open item)
- **Source:** the published paper (ASE '26 proceedings, pp. 3140–3151, DOI
  10.1145/3832783.3837543, CC BY 4.0), downloaded by the user. Local copy:
  `~/SemProject/ase-replication/ASE2026_published.pdf`,
  sha256 `8595b7c87cc7210a78a2d9ebbb4ae1fbf5a8de233a86ab9145359ffc8d396ed1`.
- **Result: 15/15 checked passages are identical** to arXiv v1. Two differences were
  extraction-format only: the PDF typesets top-𝑘 as math italic, and Table 1 uses pipe
  separators in the HTML extraction; all 15 CN Table 1 values match.
- **Passages and published pages:**

  | Passage | Page |
  |---|---|
  | §4.5 relation "aligns with the injected fact" | 3145 |
  | §3.2.1 "signalling or resolving the discrepancy"; "v1 vs. v2 manuals" | 3143 |
  | §5.2.3 rubric "appropriately acknowledges conflicting information" / "remains essentially identical" | 3146 |
  | §6.1 ZenSpace example | 3148 |
  | §6.3 currency-conversion example | 3149 |
  | footnote 6 | 3148 |
  | CN 27.6/27.0 and Table 1 | 3147 |
  | §8 "the GT check verifies whether y′ reflects the injected counterfactual" | 3150 |
  | Table 4 union 830/1,925 | 3150 |
  | CN operator description | 3144 |
  | top-k = 1 | 3147 |

- **The published Data Availability** points to the same repository (no commit pinned). Its
  only commit is still `597fd6582b`.
- **Conclusion:** all claims A1–A9 and F0–F6 hold against the version of record. Cite the
  published version, with page numbers.

### D4. Final novelty sweep, 2026-10-17
arXiv topic queries since 2026-10-14: 1 hit (2610.21309, SAST-tool testing; unrelated). ASE repository: still one commit (`597fd6582b`), 0 issues. Novelty holds.

## E. Change log
- 2026-10-18: paper v5, plain-language revision for a broad readership. Short sentences; every technical term explained on first use; one worked example (sorting) for MR. Readability (verification_2026-10-13/readability.py): Flesch Reading Ease 43.9 -> 61.0, grade 12.3 -> 9.0. Same set of numbers as v4; all source quotations word-for-word unchanged (one closing comma became a full stop). Three simplified phrasings corrected for accuracy before compiling (A2 abstention, A9 fallback, pilot observation). v4 in quarantine/paper_versions/.
- 2026-10-18: paper v4, prose only. Rewritten in the register of an empirical testing paper of 2012 (Fraser & Arcuri, ICSE 2012), compared with `verification_2026-10-13/style_compare.py`. v3 -> v4 -> 2012: 'we' per 1k words 3.0 -> 19.0 (16.5); passives per 100 sentences 39 -> 28 (26); connectives per 1k 3.7 -> 7.4 (10.5); semicolons per 1k 5.4 -> 0.2 (1.6). No number added or removed relative to v3; all source quotations unchanged. v3 in quarantine/paper_versions/.
- 2026-10-18: guide briefing rewritten (v3). Attribution corrected from the records (quarantine/README.md: Aryan wrote the base pipeline, Aditi owned the repo and ran the first analysis; Suvidh's audit, gates and rebuild per FINDINGS.md and RESEARCH_HANDOFF.md). Adds what passed in the drift phase, the null results on ASE data, the limits of the date-swap test, the void p9 run, and an AI-assistant disclosure. Previous version in quarantine/paper_versions/.
- 2026-10-18: wording only. Word-joining dashes in the paper replaced (no em dashes remain except the two IEEEtran inserts after 'Abstract' and 'Index Terms'); cover letter and ASE email rewritten as plain letters with the v3 title. No numbers added. Previous drafts in `quarantine/paper_versions/`.
- 2026-10-18: paper v3. Restructured after the ACM SIGSOFT Empirical Standards (study-type and author-involvement statement, statistics paragraph, conceptual assessment and limitations of MR-PS, conclusion-validity threats); neutral register; title changed to 'On the Validity of Metamorphic Relations for Knowledge-Conflict Testing of Retrieval-Augmented Generation Systems'. V1 now anchored in the necessary/expected-property distinction of Segura et al. (TSE 2016, checked against the accepted-version text). No result changed: v3 introduces no number absent from v2, and every source quotation is unchanged. IEEE version is now generated by `paper/ieee/make_ieee.py`. v2 copies in `quarantine/paper_versions/`.
- 2026-10-17: IEEE-format version added (`paper/ieee/main_ieee.tex`). Text identical to `main.tex` v2; a script confirmed that the only numbers that differ are layout widths. Four run-in subheadings in §6 reworded from questions to statements in both versions.
- 2026-10-17: prose of `paper/main.tex` (v2), `highlights.txt` and the guide briefing rewritten for journal style. No claim changed: a script confirmed that the set of numbers in the paper is identical to v1 and that every source quotation is unchanged. v1 copies are in `quarantine/paper_versions/`.
- 2026-10-17 14:00: section J (adversarial re-verification). A4 confirmed from the authors' released data; wording tightened.
- 2026-10-17 13:15: section I added (p17: D1 supported, D2 mixed).
- 2026-10-17 12:00: section H added (p16 pilot, gate STOP; main run not executed).
- 2026-10-17: F5 refusal count CORRECTED from 334 (12.5%) to 694 (26.0%). The original pattern missed curly apostrophes. The v2 pattern was validated (precision 30/30; 0/60 misses). The 77% finding (F1) is unaffected: altering-branch violations involve almost no refusals (17/694 refusals labelled altering).
- 2026-10-16: D3: version of record verified identical to v1 on all quoted passages; paper to cite the ACM version with pages.
- 2026-10-15: novelty re-check D2 added; version-of-record check opened (ACM DOI 10.1145/3832783.3837543).
- 2026-10-14 07:20: section G added (final p15). H2 NOT supported (Qwen-14B φ 0.323 on 6 events), H3 same order, H4 2/3.
- 2026-10-14 01:05: F5 minimum corrected from 1.7% to 2.3% (RepliQA pre-chunk, 5/217, per `posthoc_cf_quality.log`). The 1.7% figure was a transcription error.
- 2026-10-13 21:20: section F added (authors' outputs). Order/placement claim dropped (F2 null). M2/M5 declared invalid after hand reading (F4). Refusal finding added (F5).
- 2026-10-13 21:00: re-check. A3 confirmed in all RQ3 code paths (step4 and supplemental `evaluate_repair`). A8 and A9 added. Citation line numbers fixed by `check_claims.py`.
- 2026-10-13 20:40: register created. B1 and B2 withdrawn after tracing their pipeline. A2
  narrowed to the altering-labelled subset. A5 narrowed: effect on published rates is
  unmeasured.
