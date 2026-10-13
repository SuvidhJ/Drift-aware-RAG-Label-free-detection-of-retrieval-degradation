# Publication, novelty and IP assessment, 2026-10-13

For: Aditi Saxena, Suvidh Jain, Aryan Garg.

Basis:
- `CRITICAL_REVIEW_2026-10-13.md` (evidence, scripts and quotes);
- `VERIFICATION_2026-10-13.md` (reproduction checks);
- the full texts of ASE (2607.26843), HoH (ACL 2025) and Stale-Document Poisoning (2609.31342);
- the abstracts of nine further related papers.

**Not legal advice.** §6 gives a researcher's view of the IP position; a patent attorney or the
university's technology-transfer office decides patent questions.

---

## 1. Verdict

**As it stands, the work is not publishable** in a reputable journal or main conference. That
holds whether it is framed as §0 of the handoff frames it (RQ1–RQ3) or in any other way. A
competent program committee would reject it on three independent grounds:
1. **Novelty.** RQ1 replicates HoH (ACL 2025: real Wikipedia diffs, both versions indexed,
   Qwen-2.5-7B tested, order and date effects) and Stale-Document Poisoning ("dates alone
   produce only modest adaptation").
2. **Validity.**
   - In 61% of the no-label cases the context does not say which value is newer.
   - The models' choice follows passage position.
   - RQ2's registered contrast mis-maps ASE's oracle; under ASE's oracle its conclusion
     reverses.
   - RQ3's monitors are uninformative or circular by construction.
3. **Process.** Two void runs, a redesign, a rerun and a scorer change made after results were
   seen. The pre-registrations are self-dated and uncommitted.

**Has the work produced something of value? Yes, three things:**
- **(a) A verified pipeline.** It turns real Wikipedia revisions into matched old/new test
  cases; 5/5 pages were byte-identical against the live API.
- **(b) A reproducible pilot dataset** of 5,275 generations, with independent re-analysis
  scripts.
- **(c) One open, community-relevant question with promising pilot evidence:**
  *Is a synthetic counterfactual mutant (ASE's CN operator) a valid substitute for a real
  knowledge update when testing RAG systems?*
  - This is the RAG version of the question Just et al. asked in "Are mutants a valid
    substitute for real faults in software testing?" (FSE 2014, ACM SIGSOFT Distinguished
    Paper).
  - Post hoc, under ASE's own oracle, CN violation rates match real-update violation rates
    within +0.05 to +0.06 for two models on matched questions.
  - Kortukov et al. (2024) found that synthetic conflicts *overstate* failures for
    context–memory conflicts. A result for inter-context version conflicts, in either
    direction, would therefore be informative.
  - We found no prior paper answering it, from a non-exhaustive search.

**Significance today:** low, because the only novel finding is post hoc, comes from two small
4-bit models, one templated domain and one RAG configuration.

**Significance if done properly** (§4): moderate. That means a solid empirical SE paper that
tells testers whether a cheap synthetic operator can stand in for real update histories. It is
not a breakthrough. The ceiling is set by the scope: local ≤14B models, no paid APIs, one or
two domains.

## 2. Novelty scorecard

| Contribution | Status | Closest prior work | Novelty | Evidence quality |
|---|---|---|---|---|
| Real revisions with a diff oracle | method exists | HoH (2025); TemporalWiki, EvolvingQA, FRESCO | none | n/a |
| Both versions in context lead models to old answers | known | HoH; Temporal Wiki (2506.07270); WikiContradict | none (replication) | pre-registered; holds with dated labels: 0.20 / 0.22 |
| Date labels help only partly | known | SDP; HoH time-aware prompt | none (replication) | pre-registered |
| Order and value magnitude drive the choice | known in general (HoH order effects) | HoH Table 5 | low | post hoc |
| Monitors (retrieval, drift, faithfulness, informed judge) | uninformative or circular | Numeracy Gap; ASE (RAGAS weak) | none | n/a |
| **CN mutant ≈ real update (mutant validity)** | **open question** | Just et al. (code); Kortukov et al. (context–memory) | **medium, if confirmed** | post hoc pilot only |
| Real-revision RAG test-case generator plus pilot data | engineering asset | HoH and FRESCO are larger | low alone | verified |

## 3. Venue readiness

| Venue type | Ready now? | What it would take |
|---|---|---|
| ICSE / FSE / ASE / ISSTA research track | No | the full study in §4, plus wider model and domain coverage than our hardware allows; even then competitive |
| TOSEM / TSE / EMSE regular | No | as above; a multi-configuration correlation design is essential |
| **MSR 2027 Registered Reports → EMSE** | **Stage 1 feasible by 20 Nov 2026** | write the protocol (§4); the pilot data is disclosed as pilot. Risk: scope fit for MSR (corpus revision histories as "repositories"); confirm with the track chairs or consider ICSME's RR track (software evolution), checking its current dates |
| JSS / IST | No, plausible after the study | the full study in §4 |
| ACL / EMNLP (NLP) | No | RQ1 duplicates HoH; the models are too small for the NLP bar |
| Workshops / new-ideas tracks | possible, framed as preliminary | not recommended: it spends the idea on post-hoc evidence and complicates a later RR |
| arXiv preprint now | not recommended | would publish claims that §0 and the critical review have shown to be wrong |

**Recommended path: a Registered Report at MSR 2027** (abstract 13 Nov 2026, Stage 1
20 Nov 2026, Stage-1 notification 4 Feb 2027, full paper to EMSE by 30 Sep 2027; dates from
2027.msrconf.org).
- It fits our situation exactly: the problem we have is forking paths and HARKing, and a
  Registered Report has the design reviewed *before* execution.
- An accepted Stage 1 is presented at MSR and, if the study follows the protocol, the result
  is accepted in principle at EMSE whichever way it comes out.

## 4. The study that would be a real contribution

**RQ (mutant validity).** Do CN-mutant violations predict real-update violations?
1. **Rate level.** Equivalence test: |CN − real| < δ, with δ pre-registered (for example 0.05).
   Use question-clustered CIs and the TOST procedure.
2. **Ranking level, the Just et al. analogue and the main upgrade.** Across many RAG
   configurations, does the CN violation rate *rank* configurations the same way as the real
   violation rate?
   - Configurations: generator × retriever × k ∈ {1, 3, 5} × prompt × date metadata.
   - Statistic: Kendall τ with a bootstrap CI.
   - Practitioners use mutants to *compare* systems, so ranking agreement is what matters.
3. **Coupling.** Per question: does a CN violation predict a real violation on the same
   question and configuration? Report the phi coefficient and the conditional probabilities.

**Design requirements** (each answers a finding in the critical review):
- Order counterbalanced within each question, as a factor (F2).
- Date-cue factor: none vs explicit dates; recency-undecidable cases analysed separately (F1).
- ASE's oracle: a violation is staying with the pre-mutation value; acknowledging both
  satisfies it, as in ASE §5.2.3 (F5, F10).
- A closed-book filter per model (F6).
- Baselines: pick-first-listed, pick-larger, random (F3, F4).
- CN generated with **ASE's released artefact** (github.com/dbr7/RAG-metamorphic-mutation),
  using a local model that is not under test, so nobody can say "your CN is not their CN"
  (F11).
- A second domain with real versioned *text*, not templated infobox fields, e.g. Python
  documentation revisions, already partly in the repo (F7). This needs a feasibility pre-test.
- Human validation: 100–200 stratified cases labelled by Suvidh and Aryan; report Cohen's κ
  against the oracle.
- Clustering unit = question; units = distinct questions, not retriever duplicates (F9).
- Reproducibility: commit and hash the protocol; log model digests and quantization; release
  the artefact with a DOI.

**Feasibility.**
- The three local models (4-bit) can do everything on the laptop. The limit on external
  validity must be stated plainly; a free university GPU allocation for one larger open model
  (e.g. 32B–72B) would strengthen it considerably. No paid APIs.
- Time a 50-prompt pilot before committing to the 14B.

## 5. What we must not claim (current §0 text that is wrong or overstated)

- "CN overstates the real failure rate for phi3 (+15 points)": mis-mapped oracle.
- "Real revisions with a diff oracle" as our innovation: that is HoH's method.
- "Only 10–45% commit to the new value" presented as failure: "both" is satisfactory under
  ASE and WikiContradict.
- "Monitors cannot see the fault / an ASE judge told the change catches it": tautological or
  circular.
- The no-label stale rates (0.31 / 0.19) as fault rates: confounded by missing recency cues
  and by position.
- `PAPER_BASIS.md` is public in the GitHub repository and was **not** re-reviewed here; check
  it before anyone cites it.

## 6. IP and patents (researcher's view, not legal advice)

1. **Empirical findings are not patentable.** "Models choose by position" or "CN tracks real
   updates" are discoveries about the world, not inventions.
2. **Method ideas face strong prior art.** Diff-based QA from Wikipedia revisions (HoH,
   TemporalWiki, EvolvingQA); mutation operators and metamorphic oracles for RAG (ASE, whose
   code is public); recency-aware re-ranking (SDP, FRESCO); version-aware RAG (VersionRAG).
3. **Our own disclosure.** The GitHub repository `SuvidhJ/Drift-aware-RAG-…` is **public**
   (created 2026-09-09, last push 2026-10-09). Anything already in it is disclosed. In
   absolute-novelty jurisdictions such as the EPO, prior public disclosure generally bars a
   patent. The US offers a limited grace period for the inventors' own disclosures.
4. **Software and method patentability** is restricted in the US (the Alice test) and needs a
   technical effect at the EPO. "A procedure for testing a RAG system" is a weak candidate.
5. **Ownership.** Student inventions are usually governed by the university's IP policy, and
   any patent would have to name all inventors (Suvidh and Aryan as applicable).

**Recommendation:** do not pursue a patent. The appropriate strategy is open publication plus
an open artefact, which also acts as a defensive publication. If anyone wants to explore
commercial value (e.g. a "RAG update regression-testing" tool), speak to the
technology-transfer office **before** the next public disclosure; note that a Stage-1
Registered Report is itself a public disclosure. The value would lie in software and service,
not in a patent.

## 7. Plan and timeline (if the RR path is chosen)

| When | What |
|---|---|
| This week | Agree authorship and CRediT roles with Suvidh and Aryan. Correct handoff §0 (done 2026-10-13, see §0.0). Check MSR scope fit with the track chairs. |
| By 27 Oct | Pre-tests, labelled exploratory: ASE artefact runs with a local model; second-domain feasibility; 50-prompt timing; power analysis for δ and τ from the pilot variances |
| By 6 Nov | Stage-1 protocol draft (introduction, related work, hypotheses, design, analysis plan, pilot disclosure); internal review |
| 13 Nov / 20 Nov 2026 | Abstract / Stage-1 submission |
| Feb 2027 | Stage-1 decision. On acceptance: run the protocol exactly, with the human labelling in parallel |
| by 30 Sep 2027 | Stage-2 paper to EMSE |

## 8. Advice in brief

- Science advances on the questions that are *still open*. Ours is now one question, mutant
  validity, and it is genuinely useful to the SE-testing community. Do it properly instead of
  publishing the current results.
- Correct the record first. A paper that later turns out to be built on a mis-mapped oracle
  costs far more than five weeks.
- Pre-registration you can *prove* (commit, hash, RR) is worth more than any p-value we could
  report.
