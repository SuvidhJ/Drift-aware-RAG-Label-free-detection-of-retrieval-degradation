> **SUPERSEDED (2026-10-17).** The canonical manuscript is `oracle_audit/paper/main.tex`. Numbers here are outdated; in particular the refusal rate is 26.0% (694/2,671), not 12.5%.

# When the Oracle Is the Fault: Validity of Metamorphic Relations for Knowledge-Conflict Testing of RAG Systems

*Working draft. Started 2026-10-13. Authors: to be agreed (Aditi Saxena, Suvidh Jain, Aryan Garg).
Target: JSS / IST / STVR. Every claim must be in `../CLAIMS_LOCKED_2026-10-13.md` and pass
`../verification_2026-10-13/check_claims.py`.*

## Abstract (placeholder; numbers to be filled from pre-registered analyses only)
Metamorphic testing (MT) is being adopted to test retrieval-augmented generation (RAG) systems
under corpus evolution. Its most revealing operator injects a counterfactual fact and expects the
system to "update" its answer. We show that this relation is, in general, not a valid metamorphic
relation:
- without provenance, the follow-up input does not determine the expected output;
- a state-of-the-art implementation (ASE 2026) checks only that the answer *changed*, never
  that it reflects the injected fact.

We define four validity criteria for knowledge-conflict MRs and audit the framework against
them; all findings are code-verified and machine-checked. On the authors' own released outputs,
we measure [M1–M4]. We then propose value-aligned, placement-invariant and provenance-controlled
relations and evaluate them on the original benchmarks and on real Wikipedia revisions.

## 1. Introduction (outline)
1. MT for RAG is growing (ASE 2026; MetaRAG; retromorphic testing). Knowledge-conflict mutation
   gives the highest violation rates, so its validity matters most.
2. The test-oracle problem: an invalid MR produces false alarms and missed faults (oracle survey;
   MR-validity literature).
3. Contributions C1–C4 (below).

## 2. Validity criteria for knowledge-conflict metamorphic relations (contribution C1)
**Setting.** Let *q* be a question and *C* a corpus with answer *a* = S(q, C). A conflict mutation
δ produces C′ = C ∪ {d′}, where d′ asserts a value *v′* ≠ *a* for the queried fact. A
*variance relation* states that the follow-up output y′ = S(q, C′) must satisfy a predicate
P(y, y′, δ).

**V1 Decidability.** P must be determined by (q, C′, δ) alone: a reader given exactly the
follow-up input must be able to tell which answer is correct.
- If C′ holds *a* and *v′* with no information about which is valid, then "follow *v′*",
  "keep *a*" and "report both" are all defensible.
- An MR that demands one of them is undecidable, so its violations are not faults.
- *Consequence:* without provenance, the only decidable expectation is **conflict
  acknowledgement or abstention** (an invariance-type relation over the *set* of defensible
  outputs).

**V2 Value alignment.** If P demands an update, the oracle must check that y′ reflects *v′* (or
the answer implied by *v′*), not merely that y′ ≠ y.
- A change-only oracle accepts arbitrary wrong answers (false passes).
- It also rejects every correct answer that resisted the mutation (false alarms).

**V3 Placement invariance.** The verdict must not depend on where d′ is placed relative to the
original evidence when position carries no meaning. Otherwise the measured fault rate is a
property of the *test design*, given the documented position biases of LLMs.
- *Operationalisation:* MR-Order. Permuting the conflicting evidence must not change the
  committed value.

**V4 Provenance.** A variance relation ("the system must update") is valid only when the
follow-up input carries validity information that makes d′ authoritative: an effective date, a
supersession statement, or version metadata.
- *Operationalisation:* MR-Provenance-Swap. Attach validity dates to both values. The answer
  must follow the newer value, and must **flip when only the dates are swapped**. The check is
  decidable without ground truth.

**Relation to existing MR-validity work.** [to cite: MR validity and false positives; LLM
position bias (2601.06189; 2609.03148; Stable-RAG); conflict taxonomies (DRAGged: "prioritize
the up-to-date information"); temporal applicability (SDP)].

## 3. Audit of a state-of-the-art framework (contribution C2)
Against V1–V4, from `../CLAIMS_LOCKED_2026-10-13.md`:

| Criterion | Finding | Evidence |
|---|---|---|
| V2 | Numeric MR checks y′ ≠ y only; the GT check uses the original answer; the meta-evaluation counts GT-incorrect as good; repair counts "not the original correct answer" as repaired; the LLM judge never sees the injected fact | A1, A3, A8 |
| V1 | Counterfactual generated "to mislead"; no provenance; the paper's own examples call both following and ignoring it a fault | A4, A6 |
| V3 | Post-chunk always appends; pre-chunk random, and on their data placement has **no** effect (F2): V3 is satisfied in practice | A5, F2 |
| generation validity | 12.5% of CN mutants are refusals, not counterfactuals; silent fallback 0 in pre-chunk | F5, F6, A9 |

## 4. Impact on the authors' own results (contribution C3)
Pre-registered analysis of the authors' own GPT-5-mini outputs (see CLAIMS_LOCKED §F):
- **Table 1 reproduced, 10/10 cells.**
- **≥ 77.4% (460/594) of the published CN faults are correct answers**; 460/461 in the altering
  branch.
- **12.5% (334/2,671) of CN mutants are LLM refusals, not counterfactuals**; up to 26.7% per
  cell; rates shift by up to +7.7 points when they are excluded.
- **Placement has no effect** (Δ −0.025, p = 0.76): a null result, reported.
- **Their satisfied verdicts mostly do follow the counterfactual** (value-aligned disagreement
  < 10%).
- **Two pre-registered measures (M2, M5) failed validation** (unit and arithmetic); reported,
  not claimed.

## 5. Sound relations and their evaluation (contribution C4)
- **Implementation:** a value-aligned four-way oracle (FOLLOWED / KEPT / ACK / OTHER),
  MR-Order and MR-Provenance-Swap, as a drop-in for the released artefact.
- **Evaluation:** (a) the original benchmarks (local models; patches disclosed); (b) real
  Wikipedia revisions with true dates (our pipeline; live-API verified); (c) human labels,
  200 cases, Cohen's κ.

## 6. Threats to validity
- **Artefact versions:** arXiv v1 and code commit 597fd658; the camera-ready version may differ.
- **Lookup restriction:** value alignment is defined only for lookup questions.
- **Models:** local 4-bit models in C4.
- **Single framework audited:** mitigated by the general criteria and a second framework
  (optional).

## 7. Related work; 8. Conclusion. [to write]

---

## Schedule (target: complete draft by 2026-11-13)
| Week | Deliverable | Gate |
|---|---|---|
| 1 (Oct 13–19) | Authors' archive analysed (M1–M5); local pipeline running with disclosed patches; MR-Order and MR-Provenance implemented | pre-registered results in hand |
| 2 (Oct 20–26) | C4 evaluation on benchmarks + real revisions (overnight runs); human-label CSV sent to Suvidh and Aryan | runs complete, logs kept |
| 3 (Oct 27–Nov 2) | Human labels back, κ; all sections drafted; figures | full draft |
| 4 (Nov 3–13) | Internal review; re-run the novelty check; email the ASE authors; authorship agreed; submit or arXiv | submission-ready |
