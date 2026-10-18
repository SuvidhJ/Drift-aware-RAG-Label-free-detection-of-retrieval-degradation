> **SUPERSEDED (2026-10-17).** The canonical manuscript is `oracle_audit/paper/main.tex`. Numbers here are outdated; in particular the refusal rate is 26.0% (694/2,671), not 12.5%.

# When the Oracle Is the Fault: On the Validity of Metamorphic Relations for Knowledge-Conflict Testing of RAG Systems

**First draft v1, 2026-10-14.**
- Authors: Aditi Saxena, Suvidh Jain, Aryan Garg (order and affiliations to be agreed).
- Target: *Journal of Systems and Software* / *Information and Software Technology* /
  *Software Testing, Verification and Reliability*.

> **Draft status.**
> - Numbers come from `../CLAIMS_LOCKED_2026-10-13.md` (sections A and F), from
>   `../pretests/p15_interim_phi3_qwen7b.log`, and from the pre-registered scripts listed in the
>   Data Availability section.
> - Items marked **[PENDING]** await the Qwen2.5-14B run (in progress) or the human labels.
>   Items marked **[CITE]** need a verified reference.
> - All p15 results are **interim (2 of 3 models)** and must be replaced by the final
>   pre-registered report.

---

## Abstract

Metamorphic testing (MT) is increasingly used to test retrieval-augmented generation (RAG)
systems as their document corpora evolve. Its most revealing operator injects a
*counterfactual* fact into the corpus and expects the system to "update" its answer.

We argue that this relation is not, in general, a valid metamorphic relation. Without
information about which of two conflicting facts is current, the follow-up input does not
determine the expected output.

**Criteria and audit.** We formulate four validity criteria for knowledge-conflict
relations: decidability, value alignment, placement invariance and provenance. We use them to
audit a state-of-the-art framework published at ASE 2026. Its implementation checks only that
an answer *changed*, never that it reflects the injected fact.

**Impact on the authors' own results.** Using the authors' released outputs (GPT-5-mini), we
find:
- **at least 77.4% (460/594)** of the counterfactual faults reported in the paper are answers
  that match the dataset's ground truth, i.e. the system resisted the injected claim;
- **12.5% (334/2,671)** of the counterfactual "mutants" are refusals by the generating model,
  not counterfactuals.

**A valid replacement.** We propose a decidable replacement, *MR-Provenance-Swap*: attach
validity dates to the conflicting facts and swap only the dates; the answer must follow the
later date. We evaluate it on real Wikipedia revisions and on synthetic counterfactuals,
across 392 questions and three open models.

**Interim results (two of three models).** The published relation reports fault rates of 8–35%
across models and conflict sources. The valid relation finds violations (date-insensitive or
inverted answers) in 36–51% of cases. The two relations flag largely *different* cases (|φ| ≤ 0.16). Under the
valid relation, synthetic counterfactuals approximate real updates for one model and
understate them by 5.7 points for the other.

[PENDING: 14B results, human validation.]

---

## 1. Introduction

Retrieval-augmented generation (RAG) systems answer questions by retrieving passages from a
document corpus and conditioning a language model on them. Corpora evolve: values are
revised, policies are superseded, documents are re-indexed. Testing whether a RAG system
behaves correctly under such evolution is hard, because a fixed reference answer goes out of
date with the corpus. This is the test-oracle problem [CITE oracle survey].

Metamorphic testing addresses the oracle problem by checking *relations* between outputs on
related inputs instead of the outputs themselves [CITE Segura et al. 2016]. Kim, Pasini and
Tonella recently brought MT to RAG under corpus evolution [ASE 2026]. They define 11 mutation
operators and two classes of metamorphic relations, and report violation rates of
4.9–10.2% over 28,500 mutants. Their **counterfactual noise** operator (CN) produces the highest
violation rates on every dataset and in both mutation scopes: up to 27.6% (VQAonBD,
pre-chunk). That makes knowledge-conflict mutation the most consequential part of the framework.

This paper asks whether the CN relation is a **valid** metamorphic relation, and what happens
to the reported results when it is not. Our starting observation is simple. When a corpus
contains two conflicting values for the same fact and nothing indicates which is current, a
reader cannot determine the correct answer:
- following the injected value,
- keeping the original value,
- reporting both

are all defensible. A relation that demands one of them labels the others as faults *by
construction*.

**Contributions.**
- **C1. Validity criteria.** Four validity criteria for knowledge-conflict metamorphic
  relations: decidability, value alignment, placement invariance and provenance (§3).
- **C2. Code-level audit.** An audit of the ASE 2026 framework against these criteria. Every
  finding is traced to the released code and paper text, and checked automatically (29/29
  citations verified) (§4).
- **C3. Measured impact.** A quantification of the impact on the authors' *own* released
  outputs, pre-registered before the data were opened (§5). It also reports a null result
  (placement) and two pre-registered measures that failed validation.
- **C4. A decidable replacement.** *MR-Provenance-Swap*, evaluated on real Wikipedia revisions
  and synthetic counterfactuals with three open models. We compare it with the published
  relation on the same cases (§6).

**Research questions.**
- **RQ1.** Do the knowledge-conflict relations of the ASE 2026 framework satisfy basic validity
  criteria?
- **RQ2.** How much do the validity issues affect the published results?
- **RQ3.** Does a valid, decidable relation detect faults that the published relation misses?
- **RQ4.** Under a valid relation, are synthetic counterfactual mutants a valid substitute for
  real knowledge updates?

---

## 2. Background

### 2.1 RAG under corpus evolution
When a document is revised, an index may hold the outdated version, the new version, or both.
Prior work shows the harm of outdated evidence at the answer level:
- HoH builds QA pairs from Wikipedia snapshot diffs, indexes current and historical versions,
  and finds that outdated passages reduce answer accuracy even when current evidence is
  retrieved [HoH, ACL 2025];
- Stale-Document Poisoning shows that outdated evidence overturns otherwise-correct answers,
  and that "dates alone produce only modest adaptation" [SDP 2026];
- DRAGged into Conflicts prescribes that for conflicts due to outdated information, models
  should "prioritize the up-to-date information …, while optionally acknowledging the presence
  of outdated sources" [2506.08500];
- WikiContradict treats reflecting a conflict as the desired behaviour for real contradictions
  [2406.13805].

We do not claim these behaviours as new. We study whether **testing techniques** can detect
them.

### 2.2 The ASE 2026 framework
The framework mutates either the corpus before chunking (pre-chunk) or the retrieved context
(post-chunk). Defaults: top-k = 1, GPT-5-mini as generator and judge, text-embedding-3-small,
and 300 base questions per dataset over five datasets (RepLiQA; ConvFinQA, FinQA, TAT-DQA and
VQAonBD from T2-RAGBench).

**The CN operator** "prompts a separate LLM to generate a plausible counterfactual statement
given the original fact with the user query."

**Relation classes:**
- *invariance* (semantic-preserving mutations): y′ ≈ y;
- *variance* (semantic-altering mutations): "We expect y′ ≠ y, specifically where y′ aligns
  with the injected fact."

**Checking:**
- a semantic-preservation judge labels each mutant;
- numeric datasets compare answers within a 1% tolerance;
- RepLiQA uses an LLM judge.

**Filtering:** questions are kept only if the closed-book model fails and the baseline RAG
answer matches the ground truth.

---

## 3. Validity criteria for knowledge-conflict relations (C1)

Let *q* be a question, C a corpus and y = S(q, C). A conflict mutation δ adds a passage d′
asserting a value v′ ≠ y for the queried fact. A variance relation P(y, y′, δ) constrains the
follow-up output y′ = S(q, C ∪ {d′}).

**V1 Decidability.** The expected output must be determined by the follow-up input alone.
- If C ∪ {d′} contains y and v′ and nothing indicates which is valid, "follow v′", "keep y"
  and "report both" are all defensible.
- A relation that demands one of them is undecidable, and its "violations" are not
  evidence of faults.
- Without provenance, the only decidable expectation is *acknowledgement or abstention*.

**V2 Value alignment.** If a relation demands an update, the oracle must check that y′ reflects
v′ (or the answer v′ implies), not merely that y′ ≠ y.
- A change-only oracle **accepts** wrong answers.
- It also **rejects** every answer that resisted the mutation.

**V3 Placement invariance.** If the position of d′ relative to the original evidence carries no
meaning, the verdict must not depend on it. Given documented position biases of LLMs in
conflicts [2601.06189; 2609.03148; Stable-RAG], a placement-dependent verdict would measure the
*test design*, not the system.

**V4 Provenance.** A variance relation ("the system must update") is valid only if the
follow-up input makes d′ authoritative, e.g. by an effective date, a supersession statement or
version metadata.

V1 and V4 are two sides of the same requirement. V2 and V3 concern how a decidable relation is
checked.

---

## 4. RQ1: Audit of the ASE 2026 framework (C2)

**Method.**
- **Sources:** we read the paper (arXiv 2607.26843 v1) and the released artefact
  (github.com/dbr7/RAG-metamorphic-mutation, single commit 597fd6582b) in full.
- **Tracing:** each finding below is traced to a file and line.
- **Verification:** a script asserts every citation and quote against frozen copies; 29/29
  checks pass.

**Table 1. Findings against V1–V4.**

| ID | Criterion | Finding | Evidence |
|---|---|---|---|
| A1 | V2 | For numeric datasets, a semantics-altering mutant is SATISFIED iff the answer *differs* from the baseline answer. The injected value is never compared anywhere in the released code | `step3_mutation_campaign.py` L746–748; GT check L1057–1069 |
| A1′ | V2 | Ground-truth meta-evaluation: `system_good = (is_preserving == (gt_correct == "correct"))`, so for altering mutants an answer judged *incorrect* against the original ground truth counts as good behaviour | L1199 |
| A2 | V1/V2 | Baseline answers are kept only if they match the ground truth (step 2, L218–221). Hence every altering-branch violation is an answer within ~2% of the ground truth, and every changed non-abstaining answer passes | logical consequence of A1 and step 2 |
| A3 | V2 | Repair (RQ3 of ASE): for non-preserving mutants, `fault_repaired = (gt_correct != "correct")`. A CN fault counts as "repaired" when the new answer is *not* the original correct answer | `step4_fault_repair.py` L390–395; same rule in the supplemental repair scripts |
| A4 | V1 | The paper defines the CN fault inconsistently. §3.2.1: "signalling or resolving the discrepancy". §4.5: "y′ aligns with the injected fact". The §6.1 example calls *ignoring* the counterfactual a fault; the §6.3 example calls *adopting* it ("an answer that directly opposed the ground truth") a fault that "persisted after repair", although under A3 that answer counts as repaired | paper text |
| A5 | V3 | Post-chunk CN appends the counterfactual after the original text (`mutation_operators.py` L651); pre-chunk inserts it at a random sentence position (L602). Placement is not analysed in the paper | code |
| A6 | V1 | The counterfactual is generated to "mislead a reader into giving a wrong answer" (L551–556) and carries no provenance | code |
| A8 | V2 | The RepLiQA judge for altering mutants is shown the *original* context or document (L814, L858) and a one-line description of the mutation. It never sees the counterfactual, so it cannot check alignment with it | code |
| A9 | (generation) | On any exception, counterfactual generation silently falls back to a heuristic, without logging it | L539–544, L568–574 |

**Answer to RQ1.** The CN relation violates V1 and V4 by design: it has no provenance and its
fault definition is inconsistent. Its implementation violates V2 on all five datasets.
Placement is fixed in one scope (A5); whether this matters is an empirical question (§5).

---

## 5. RQ2: Impact on the published results (C3)

**Data.** The authors' released outputs (Figshare archive): baseline results, mutation records,
metamorphic verdicts, ground-truth validation and repair results, for both scopes and all five
datasets.

**Pre-registration.** The analysis script was hashed and logged before the archive was opened.
One amendment, made after seeing only file and field names, implemented a measure that had
been defined but not coded.

**Integrity.** Our recount of CN violations from the verdict files reproduces the paper's
Table 1 in **10/10** dataset × scope cells.

**Measures and results** (all CN mutants; Wilson 95% CIs):
1. **Violations are correct answers (M1, pre-registered).**
   - 460 of 461 altering-branch CN violations are answers matching the ground truth
     [0.988, 1.000]. This includes RepLiQA, according to the authors' own ground-truth judge
     (64/65).
   - The altering branch holds 461 of all 594 CN violations reported in the paper. Hence
     **at least 77.4% of the CN faults reported in the paper are answers that match the
     dataset's ground truth**: the system was not misled by the injected claim.
2. **Refusals used as mutants (post hoc, descriptive).**
   - 334 of 2,671 CN "counterfactuals" (12.5%) are refusals by the generating model, e.g.
     "Sorry, I can't help create misleading counterfactuals…". A random sample of 20 matches
     contained 20 genuine refusals.
   - The rate ranges from 2.3% (RepLiQA pre-chunk) to 26.7% (TAT-DQA post-chunk).
   - Some refusals insert the correct answer into the context.
   - Excluding refusals changes the published CN rates by up to +7.7 points (TAT-DQA
     post-chunk: 22.3% → 30.0%).
3. **Placement (M3, pre-registered): null.**
   - Pre-chunk insertion is random, which gives a natural randomized comparison.
   - Violation rate with the counterfactual before the ground-truth value: 0.344 (45/131).
     After it: 0.368 (28/76). Δ = −0.025, Fisher p = 0.76.
   - **Placement does not affect the published verdicts; V3 holds in practice for this
     framework.**
4. **Value alignment of passes (M4, pre-registered): not supported.**
   - On lookup questions, a value-aligned re-scoring disagrees with the published verdicts in
     9.6% of cases (threshold 10%).
   - Most "satisfied" answers do reflect the injected value. The practical effect of A1 lies in
     the *violation* side (item 1), not in false passes.
5. **Two pre-registered measures failed validation.**
   - M2 (passes matching neither value) and M5 (repairs matching neither value) used literal
     numeric matching.
   - Hand inspection showed that most such answers follow the counterfactual in other units
     (e.g. "28 million" → 28000000) or through arithmetic.
   - We report them as invalid measures and draw no conclusion from them.
6. **Heuristic fallback (A9):** none detected in pre-chunk CN mutants.

**Answer to RQ2.**
- The validity issues materially change the meaning of the headline CN results: at least
  three quarters of the reported CN faults are correct answers.
- One in eight CN mutants is not a counterfactual at all.
- Placement and false passes have little effect in this framework.

---

## 6. RQ3–RQ4: A decidable replacement relation (C4)

### 6.1 MR-Provenance-Swap
Given a conflict between an original value and a mutated value, attach validity dates to both
passages ("(as of YYYY-MM-DD) …").
- **CORRECT:** the mutated value carries the later date.
- **SWAPPED:** the dates are exchanged; nothing else changes.
- **Requirement:** a correct system commits to the later-dated value in both conditions, so
  its answer must switch when only the dates are swapped.

The expected output is fully determined by the follow-up input, so the relation satisfies V1
and V4 without ground truth. We check committed values deterministically (four classes: new,
old, both, neither), which satisfies V2. We counterbalance passage order, which satisfies V3.

**Outcome classes per question × source × order:**

| Class | Meaning |
|---|---|
| PASS | follows the later date in both conditions |
| INSENSITIVE | same single value in both conditions |
| INVERTED | follows the earlier date in both |
| ACK | acknowledges or abstains in either condition |

A PS *violation* is INSENSITIVE or INVERTED.

### 6.2 Setup
**Questions.** 392 questions on 145 pages, built from real Wikipedia infobox revisions. We
take the last revision before 2025-10-01 and the revision current in October 2026, across UN
member states and samples of Nasdaq/NYSE-listed companies.
- Old and new values are extracted deterministically.
- Years are excluded as values, and questions do not contain their answers.
- For 5 random pages, both stored revisions are byte-identical to the live MediaWiki API.

**Conflict sources:**
- **REAL:** old-revision value vs new-revision value;
- **CN:** current value vs an LLM-written counterfactual (qwen2.5:7b-instruct), with a fresh
  non-year value.

**Context and design.** Each prompt shows the two conflicting field passages and one same-page
distractor (post-chunk style; no retrieval). Factors: dates {none, correct, swapped} × order
{mutation last, mutation first}, plus an adequacy control with the new value alone. That is 13
prompts per question per model.

**Models.** All run locally through Ollama: phi3-mini 3.8B (Q4_0), Qwen2.5-7B-Instruct (Q4_K_M)
and Qwen2.5-14B-Instruct (Q4_K_M), at temperature 0 with at most 24 output tokens.

**Published relation, applied to the "no dates" condition with the mutation last** (ASE's
post-chunk placement): VIOLATED iff the answer commits to the original value alone.

**Pre-registration.** Hypotheses H1–H4 and their thresholds were fixed in the script before
generation (hash in `pretests/p15_prereg.log`).

### 6.3 Results (interim: phi3-mini and Qwen2.5-7B; 14B [PENDING])
Adequacy: phi3 0.990, Qwen-7B 0.806 (threshold 0.60).

**Table 2. Fault rates under the published and the valid relation** (Wilson 95% CIs; 392
questions, 784 date-swap pairs per source).

| Model | Source | Published relation: violation | MR-PS: violation | φ (same cases) |
|---|---|---|---|---|
| phi3-mini | REAL | 0.301 [0.258, 0.348] | 0.513 [0.478, 0.548] | −0.161 |
| phi3-mini | CN | 0.347 [0.302, 0.395] | 0.509 [0.474, 0.544] | −0.147 |
| Qwen2.5-7B | REAL | 0.082 [0.058, 0.113] | 0.413 [0.379, 0.448] | 0.075 |
| Qwen2.5-7B | CN | 0.089 [0.065, 0.122] | 0.356 [0.323, 0.390] | 0.110 |
| Qwen2.5-14B | REAL / CN | [PENDING] | [PENDING] | [PENDING] |

**Table 3. MR-PS outcome classes on REAL updates** (784 pairs per model).

| Model | PASS | INSENSITIVE | INVERTED | ACK |
|---|---|---|---|---|
| phi3-mini | 170 | 387 | 15 | 212 |
| Qwen2.5-7B | 90 | 313 | 11 | 370 |

**RQ3, decidable faults exist and differ from the published faults.**
- **H1 (≥ 10% PS violations, pre-registered): supported** for both models so far. Pooled
  violation: phi3 0.511 [0.486, 0.536]; Qwen-7B 0.385 [0.361, 0.409].
- **H2 (φ < 0.30 for every model, pre-registered): supported** for both models so far
  (|φ| ≤ 0.161). The published relation and the valid relation flag largely different cases.
- **The published relation understates date-insensitivity.** For Qwen-7B on real updates it
  reports 8.2% violations, while 41.3% of date-swap pairs are date-insensitive or inverted.
- **Why (post hoc).** On the questions where the model ignores the dates, the published
  relation mostly sees the model commit to the mutated value, and counts that as a pass:
  - Qwen-7B REAL: 142/208, with "both" answers only 42/208;
  - phi3 REAL: 138/214.

  These "passes" reflect *which* value the model happens to pick, not whether it can tell
  which value is current; the date swap reveals that it cannot.
- **H3 (model ranking, descriptive):** both relations order the two models the same way
  (Qwen-7B below phi3). [PENDING 14B]

**RQ4, mutant validity under a valid relation (H4: |PS(CN) − PS(REAL)| < 0.05;
question-clustered bootstrap):**
- phi3: −0.004 [−0.043, +0.037] (within);
- Qwen-7B: −0.057 [−0.087, −0.028] (synthetic counterfactuals slightly *understate*
  real-update faults).
- Mixed so far. [PENDING 14B]

---

## 7. Discussion

**What the CN relation measures.** In the evaluated framework, a CN "violation" is almost always
a system that kept the correct answer despite an injected claim written to mislead. The
relation therefore measures *resistance to misinformation*, with the sign inverted, not
failure to adopt knowledge updates. Practitioners who rank systems by CN violations would
favour systems that are *more* easily misled.

**Why provenance is necessary.** The underlying behaviour, models not reliably using dates, is
known [SDP; HoH]. Our contribution is a *testing* one. Without provenance in the mutation, no
oracle can tell an update failure from correct scepticism. With provenance and a swap, the
expectation becomes checkable without ground truth, and the faults found are different from
those the published relation reports.

**Recommendations for MT of RAG.**
1. Give conflict mutations provenance, and use swap-style relations.
2. Check value alignment, not change.
3. Filter generator refusals and log generation fallbacks.
4. Report acknowledgement and abstention as a separate class.

**Ethics and collegiality.** We will share our findings with the ASE authors before
publication. [PENDING]

## 8. Threats to validity

**Construct validity.**
- Value alignment (M2, M4, M5) is defined only for lookup questions; literal matching misses
  unit-scaled and computed answers. We therefore draw no conclusion from M2 and M5.
- The four-way label of committed values was hand-checked on 90 outputs from an earlier
  experiment on the same questions (3.3% disagreement, all unit conversions). Formal human validation is
  [PENDING]: 200 cases, two annotators, Cohen's κ.

**Internal validity.**
- The analyses of the authors' outputs were pre-registered (hashes logged). The refusal
  analysis is post hoc and descriptive.
- MR-PS hypotheses were pre-registered before generation. The results here are interim.

**External validity.**
- Three small, 4-bit quantized open models; one domain for C4 (Wikipedia infobox numbers);
  one framework audited in depth.
- The audit findings A1–A8 are properties of the released code, independent of the models.

**Versions.** Our findings refer to arXiv v1 and code commit 597fd6582b. The published
proceedings version may differ.

## 9. Related work [to expand]
- **Metamorphic testing of RAG and LLMs:** ASE 2026; MetaRAG [CITE]; MeTMaP (2402.14480);
  retromorphic testing (2603.27752); MT for LLM-based NLP (ICSME 2025) [CITE].
- **Test oracles and MR validity:** [CITE oracle survey]; ML oracle issues (2105.01407).
- **Mutant validity:** Just et al., FSE 2014 ("Are mutants a valid substitute for real faults
  in software testing?").
- **Outdated information and conflicts:** HoH; SDP; Temporal Wiki (2506.07270); WikiContradict;
  DRAGged into Conflicts; FRESCO (2604.14227); Re³ (ACL 2026); Kortukov et al. (2404.16032).
- **Position bias in conflicts:** 2601.06189; 2609.03148; Stable-RAG (2601.02993).

## 10. Conclusion
The counterfactual relation of a state-of-the-art RAG metamorphic-testing framework is not a
valid metamorphic relation. In the framework's own released results:
- most reported counterfactual faults are correct answers;
- one in eight counterfactual mutants is a model refusal.

A provenance-controlled swap relation is decidable without ground truth. It finds
substantially more date-insensitive behaviour, on different cases, than the published relation.
[Final numbers PENDING the 14B run.]

## Data availability
- **Repository:** `staleness-oracle-audit` branch.
  - Scripts: `oracle_audit/analyze_ase_outputs.py`; `pretests/p15_provenance_swap.py`;
    `verification_2026-10-13/check_claims.py`.
  - Pre-registration logs: `oracle_audit/prereg.log`, `pretests/p15_prereg.log`.
- **Authors' outputs:** Figshare archive of the ASE artefact (not redistributed).
- **Wikipedia revisions:** retrievable by page ID and timestamp.

---
*Draft notes for co-authors:*
1. Replace interim p15 numbers with the final pre-registered report (3 models).
2. Run the human-validation study.
3. Verify every [CITE].
4. Re-run the novelty search before submission.
5. Agree authorship.
6. Convert to the target journal's LaTeX template.
