# Critical review of the real-revision metamorphic-testing study, 2026-10-13

**Scope.** An adversarial review of the design, the claims and the novelty, written as a hostile
SE/NLP reviewer would. It supersedes the *validity* conclusions of
`VERIFICATION_2026-10-13.md`. That report checked that numbers **reproduce**; it did not check
that they **mean** what §0 says.

**Evidence.**
- Every number below comes from scripts in `research/staleness/verification_2026-10-13/`
  (`c_stress.py`, `c_rq2.py`, `c_dated.py` and the `v_*` files) run on the raw records.
- Every analysis in this file is **post hoc**.
- Paper quotes come from the full texts in `verification_2026-10-13/lit/`.

**Severity.** FATAL = must be fixed before any submission. MAJOR = a reviewer will raise it and
it must be addressed. MINOR = disclose.

---

## 0. Bottom line

The data pipeline is genuine and the numbers reproduce. Four claims in §0, however, would not
survive review:

1. **RQ2's headline reverses under the ASE paper's own oracle** (FATAL). "CN overstates phi3's
   failure rate by +0.15" compares CN's *satisfied* outcome with the real update's *violated*
   outcome. With ASE's violation definition the gap is +0.06 (phi3) and +0.05 (Qwen-7B). Both
   are below our pre-registered 0.10 threshold, so CN is a **fair proxy**.
2. **RQ1's "stale rate" is mostly not a measure of a fault** (FATAL):
   - 61% of the no-label cases (368/605) contain no date that says which value is newer, so
     the correct answer cannot be inferred from the context.
   - The model's choice follows passage *position*, not date: phi3 picks the first-listed
     value 62% of the time.
   - Where dates *are* present, phi3's choice does not change: P(old | single answer) = 0.40
     both with and without dates.
3. **RQ1 is not new** (FATAL for novelty):
   - HoH (ACL 2025) already built QA from real Wikipedia snapshot diffs. It indexed both
     versions, tested Qwen-2.5-7B with current + outdated passages in context, and reported
     order and date effects.
   - Stale-Document Poisoning already reports that "dates alone produce only modest
     adaptation".
   - Temporal Wiki (arXiv 2506.07270) reports that models "can be misled when multiple versions
     of a fact appear in context".
4. **RQ3 is largely tautological or circular** (FATAL as a contribution):
   - M1 cannot fail by construction.
   - M4 is handed the diff, which *is* the oracle.
   - M2's judge flags about 84% of all answers.
   - M3 compares near-identical texts, and for bge it is borderline anyway.

**What survives** (§8): a real, matched-question finding that ASE's CN operator *tracks* real
update failures within about 5 points for two small models. This is a "mutant validity"
question in the Just et al. (FSE 2014) sense that we did not find answered for RAG. It is only
publishable if re-run with a pre-registered, counterbalanced design (§9).

## 1. Where the earlier verification was biased

- It treated **reproduction as validation**. "MATCH" meant "same number from the same records",
  not "the number measures the claimed construct".
- It accepted p12's pre-registered contrast without re-deriving it from ASE's oracle. I had
  quoted the decisive sentence ("y′ aligns with the injected fact") and still marked RQ2 as
  holding.
- It knew from §5 that HoH existed but did not read it, and so missed that RQ1 replicates it.
- It did not test the alternative explanations: no-cue cases, position, a larger-number
  heuristic, duplicate prompts, the parametric prior.
- It reported the 14B "both" caveat as a *parsing* issue. It is a *construct* issue: under ASE,
  WikiContradict and DRAGged, acknowledging both values can be the desired behaviour.

## 2. Novelty against the literature (full texts read for ASE, HoH and SDP)

| Our claimed contribution (§0.1) | Already done by | Exact evidence |
|---|---|---|
| Real revisions with an oracle from the old/new diff | **HoH** (Ouyang et al., ACL 2025, `2025.acl-long.301`) | "We propose a two-stage approach that combines diff algorithms (Myers, 1986) with LLM pipelines to extract and verify factual changes from Wikipedia snapshots." It also generates "the current correct answer and its outdated counterpart"; 96,124 QA pairs. |
| Both versions in the index | HoH | "The underlying corpus comprises both current and historical versions of all articles used in the QA dataset construction." |
| Generators choose the old value when both are in context | HoH (incl. Qwen-2.5-7B); Temporal Wiki (2506.07270); WikiContradict (2406.13805) | HoH: "when introducing just one outdated passage … Llama-70B's perfect scores drop by over 10%". 2506.07270: "can be misled when multiple versions of a fact appear in context". |
| Date labels help only partly | **SDP** (2609.31342); HoH's time-aware instruction | SDP: "dates alone produce only modest adaptation". HoH adds "Current Date" and "Last Modified Time" to all documents. |
| Passage order matters | HoH Table 5 | "Qwen-7B displays extreme sensitivity with score variations exceeding 50%" across orderings |
| Retrieval cannot tell old from new | HoH Table 3; FRESCO (2604.14227) | HoH: "models that perform better at retrieving relevant information also show proportionally better performance at retrieving outdated information". FRESCO: re-rankers show "a strong bias toward older, semantically rich documents". |
| What the right behaviour is | DRAGged into Conflicts (2506.08500) | "The expected behavior is to prioritize the up-to-date information …, while optionally acknowledging the presence of outdated sources." |
| Synthetic conflicts vs real ones | **Kortukov et al.** (2404.16032), for context–memory conflicts | "In this realistic scenario, we find that knowledge updates fail less often than previously reported." |

**Not found in this search** (the search was not exhaustive): a matched comparison between an
SE **mutation operator** (ASE's CN) and **real updates** on the *same* questions, for
*inter-context* version conflicts. That is the one gap left.

## 3. RQ1 (real updates): construct validity

**F1 (FATAL). The no-label condition usually contains no recency signal.** (`c_stress.py`, A)

| Context of the 605 append cases | n | phi3 old-only | Qwen-7B old-only | 14B old-only | 14B both |
|---|---|---|---|---|---|
| no date inside either passage | 368 | 0.30 | 0.20 | 0.06 | 0.77 |
| years in both passages, new one later | 228 | 0.32 | 0.20 | 0.02 | 0.95 |

- Without a recency signal, choosing the old value is not an avoidable error; the task is
  undecidable from the context.
- Among single-value answers, P(old) is 0.40 for phi3 *in both strata*. For Qwen-7B it is 0.41
  without dates and **0.55 with dates**: the models do not use the dates that are there.
- Required change: report RQ1 only where a recency signal exists (the dated variant or
  in-passage years), and counterbalance the order.

**F2 (FATAL). Position, not content, drives the choice.** (B)
- phi3's old-only rate is 0.37 when the old passage is listed first, against 0.24 when the new
  one is.
- Among single-value answers, phi3 picks the first-listed value 62% of the time and Qwen-7B
  56%. The 14B picks the *last-listed* value 93% of the time (n = 84).
- Retrieval order is essentially a coin flip: 311 cases old-first, 294 new-first.
- The stale rate is therefore a product of retrieval-order noise and position bias. The
  experiment did not counterbalance order. HoH had already shown order effects.

**F3 (MAJOR). A trivial heuristic is competitive.** (F)
- The new value is the larger number in 69–74% of single-value cases, so "pick the larger
  number" would be right about 70% of the time.
- phi3 picks the larger value 64% of the time, and the 14B 87%.
- Neither of these baselines was reported.

**F4 (MAJOR). The replication criterion could hardly fail.** A model choosing at random
between two values would show old-only around 0.3–0.5, well above the 0.10 bar. The
criterion tests "not zero", not "worse than chance or than the position baseline".

**F5 (MAJOR). The "both" answers are mislabelled as a failure mode.**
- ASE §5.2.3 counts "appropriately acknowledges conflicting information" as **satisfying** the
  relation.
- WikiContradict treats reflecting a conflict as the *desired* behaviour.
- DRAGged prefers prioritising the up-to-date value.
- Our prompt explicitly allows "number(s), separated by commas".
- The 14B's 84% "both" is therefore not evidence of a fault. The phrase "only 10–45% commit to
  the new value" frames a reasonable behaviour as a failure.

**F6 (MAJOR). There is no parametric (closed-book) control.**
- ASE filters out questions the model answers closed-book.
- SDP measures poisoning only among questions answered correctly without retrieval.
- Kortukov et al. found a "parametric bias".
- Country GDP and population figures are likely memorised, and the old IMF estimates are the
  ones closer to the models' training cutoffs.
- A crude filter (keeping cases where the model was right under replace) leaves the rates
  unchanged (phi3 0.313, Qwen-7B 0.228, 14B 0.042). That is not a closed-book test.

**F7 (MAJOR). Ecological validity is overstated.**
- The passages are synthetic templates ("Mexico -- gdp nominal: $2.121 trillion."), one per
  infobox field. In *form* they are as synthetic as ASE's CN passages; only the *values* are
  real.
- The questions are templates built on raw infobox keys ("The gdp nominal per capita rank of
  Niger is ___."), which the p9 docstring calls "natural".
- The premise that "both versions are indexed" is unverified for real frameworks (§0.1 already
  says so).
- Text cleaning leaves `&nbsp;` in passages and drops minus signs.

**F8 (MAJOR). Narrow models and settings.**
- Three small **4-bit quantized** models: phi3:mini Q4_0; qwen2.5 7B and 14B Q4_K_M.
- One prompt template; temperature 0 with a single sample; a 24-token output cap that
  truncates phi3 lists.
- One domain (infobox numbers) and one language.
- HoH and SDP evaluated full-precision Qwen-2.5-7B, so any direct comparison must state the
  quantization.

**F9 (MAJOR, statistics). Units are not independent.**
- The 605 cases come from **361 distinct questions** on 142 pages. The minilm and bge copies
  of a question are correlated, and 20 have byte-identical prompts.
- Wilson intervals on n = 605 are therefore too narrow. Use page- or question-clustered
  intervals, or model retriever as a factor.
- Rates are unchanged when de-duplicated (phi3 0.315 on 585 distinct prompts).
- The top 10 pages contribute 108 of the 605 cases.

**What survives in RQ1.** With explicit "as of" dates on every passage (the pre-registered
dated variant):
- phi3 is still old-only in 0.20 of cases and Qwen-7B in 0.22 (`c_dated.py`).
- Position matters less there: phi3 0.23 (old first) vs 0.17 (new first); Qwen-7B 0.22 vs 0.21.

This is a defensible *replication* of SDP and HoH in a metamorphic-testing setting, not a new
finding.

## 4. RQ2 (CN vs real updates)

**F10 (FATAL). The registered contrast does not operationalise ASE's oracle.**
- ASE §5.2.3: "High scores of 4–5 indicate successful adaptation, where y′ clearly reflects the
  mutated context or appropriately acknowledges conflicting information … scores of 1–2
  indicate a failure to adapt, where y′ remains essentially identical to y_base".
- So a CN violation is the answer *staying with the original (current) value*.
- The real-update analogue maps base = old value and mutation = adding the new version, so a
  violation is the answer staying with the old value.
- p12 instead compared CN "injected-only" (ASE: *satisfied*) with real "old-only" (*violated*).

`c_rq2.py`, page-clustered 95% CIs:

| Contrast | phi3 CN | phi3 real | diff | Qwen-7B CN | Qwen-7B real | diff |
|---|---|---|---|---|---|---|
| p12 as registered (injected-only vs old-only) | 0.453 | 0.307 | **+0.146** | 0.236 | 0.197 | +0.039 |
| **ASE oracle** (current-only vs old-only) | 0.369 [0.33, 0.41] | 0.307 [0.26, 0.35] | **+0.061** | 0.244 [0.21, 0.29] | 0.197 [0.16, 0.23] | **+0.048** |
| ASE oracle, non-abstaining "neither" counted as a violation | 0.383 | 0.331 | +0.052 | 0.342 | 0.286 | +0.056 |
| Paired subset (same retriever and question; n = 601) | 0.363 | 0.303 | +0.060 | 0.248 | 0.196 | +0.052 |

- ASE's own text is internally inconsistent. Its fault taxonomy frames the contradiction as
  "stale data", while its oracle treats the injected fact as the new state. The p12 mapping
  followed the first; ASE's *reported* violation rates (27.6% etc.) follow the second.
- To ask whether ASE's CN violation rate estimates real failure rates, the oracle mapping is
  the only valid one.
- This re-analysis is post hoc and must be pre-registered before any claim is made.

**F11 (MAJOR). Our CN is not ASE's CN.**
- **Generator of the counterfactual:** ours is qwen2.5:7b, which is also a model under test
  (self-generated distractors); ASE uses GPT-5-mini.
- **Closed-book filter:** ASE has one; we don't.
- **Retrieval:** ASE uses k = 1 by default; we use k = 3, conditioned on both passages being in
  the context, which is closer to ASE's *post-chunk* scope.
- **Injection:** ASE injects one passage per document; we inject one per queried field.
- **Defects** (see the verification report): 8/400 fallbacks, some implausible ("119.78th");
  131/400 counterfactual value sets include a year.

**F12 (MAJOR). Mechanism.** In both conditions the generator chooses between two conflicting
numbers mostly by position and magnitude (F2, F3). CN ≈ real may simply mean that *neither
condition engages truth or recency*. A counterbalanced design is needed to separate these.

## 5. RQ3 (monitors)

**F13 (FATAL as a contribution). Each monitor is uninformative by construction:**
- **M1:** gold-page hit@3 = 1.000 for the stale, append *and* replace indexes. A page-level hit
  cannot distinguish versions of the same page; it cannot fail.
- **M4:** the judge is told "<field> changed from <old> to <new>". That is the diff oracle in
  natural language, so 98.9% agreement with the diff oracle is circular. §0.2 already admits
  that "knowing the diff is the signal".
- **M2:** the faithfulness judge flags about 84% of all answers, right or wrong (FPR 0.85 /
  0.84). It is a strawman baseline, not RAGAS.
- **M3a:** old and new passages differ by one number, so near-zero drift is expected (and the
  numeracy gap is known: Deng et al., EACL Findings 2026, in §5). For bge it is not clearly
  null: independent subsample reruns give p = 0.014–0.21.
- **M3b:** AUROC about 0.5. It tests whether the size of the edit predicts a stale answer, an
  unmotivated hypothesis.

## 6. Process and statistics

**F14 (MAJOR). Forking paths.** The current design was reached through:
1. p9 void twice (a year bug, then adequacy 0.51);
2. a redesign (p11);
3. a sample-order bug, rerun;
4. a scorer amendment written *after* the p11 rerun results were printed (`p9_wiki_infobox.py`
   last written at 16:54, results at 16:52);
5. a scope cut for the 14B, with about 220 orphan 14B outputs from a stopped run.

Each step is disclosed or disclosable, but together they invite a "researcher degrees of
freedom" critique. The pre-registrations are uncommitted and self-dated, so they cannot be
verified.

**F15 (MAJOR). Post-hoc analyses drive the narrative.** The 14B "both" diagnostic and the RQ2
"both" columns are post hoc, yet §0.2 leads with them.

**F16 (MINOR).**
- No multiplicity control across the generator × variant × retriever comparisons.
- Thresholds (0.10, 0.75 AUROC, 0.60 adequacy) are conventional, not justified.
- FreshQA (10 cases, 6 pages) supports no statistical claim.

## 7. Engineering and reproducibility

- **Version control:** code, pre-registrations and the handoff are untracked in git; results
  live in caches under `embeddings/cache/`.
- **Model and run records:** Ollama model IDs (`7cdf5a0187d5`, `845dbda0ea48`, `4f2222927938`)
  were not logged at run time. Quantization was not recorded.
- **Determinism:** temperature 0 is not guaranteed to be deterministic across Ollama versions
  or hardware, and only one run exists.
- **Testing:** the pretests are not under test, and the scorers (`P9.commit`, `facts.NUM`)
  have known edge cases (unit conversions, dot as the thousands separator, signs).

## 8. Claims that can be defended now (wording)

1. "On 605 both-versions-in-context cases built from real Wikipedia infobox revisions, with an
   explicit 'as of' date on every passage, 4-bit phi3-mini and Qwen2.5-7B still answered with
   the superseded value alone in 20% and 22% of cases (pre-registered). This is consistent with
   HoH and Stale-Document Poisoning."
2. "Without date labels, the choice between two conflicting values was dominated by passage
   order and value magnitude (post hoc)."
3. "Under ASE's own oracle, the counterfactual (CN) violation rate was within 0.05–0.06 of the
   real-update violation rate on the same questions for both models (post hoc; requires
   confirmation)."
4. "Retrieval hit rate and embedding displacement are uninformative about version conflicts by
   construction."

This one is a design observation, not a finding. State it once and do not make it an RQ.

## 9. What would make the work publishable

**Reframe** around the one open question: *are synthetic counterfactual mutants a valid
substitute for real knowledge updates in RAG metamorphic testing?* This is the Just et al.
(FSE 2014) question, "Are mutants a valid substitute for real faults in software testing?",
asked for RAG. Kortukov et al. found that synthetic conflicts *overstate* failures for
context–memory conflicts; whether the same holds for inter-context version conflicts is open.

**Replace p14 with a pre-registered, counterbalanced, paired experiment** (and commit or hash
the script before running).

- **Units:**
  - the 361 distinct questions;
  - each question gets a fixed constructed context of {conflict passage A, conflict passage B,
    one same-page distractor}, i.e. ASE's post-chunk scope, which removes retrieval noise;
  - **source** ∈ {real (old vs new), CN (current vs injected)}.
- **Factors, crossed within each question:**
  - **order:** A first / B first;
  - **date cue:** none / explicit "as of" dates;
  - **prompt:** comma-list / exactly one number (the old p14 question becomes one factor).
- **Oracle:** ASE's mapping, i.e. a violation is the answer staying with the pre-mutation
  value, with "both" reported separately and counted as satisfied, as in ASE.
- **Pre-registered hypotheses:**
  - **H1 (mutant validity):** |CN − real violation| < 0.05, tested by equivalence (TOST) with
    question-clustered CIs.
  - **H2:** order has an effect (paired McNemar), and so does the date cue.
- **Controls:**
  - a closed-book run per question, then analysis restricted to questions the model cannot
    answer closed-book;
  - "pick the first-listed number" and "pick the larger number" baselines reported alongside;
  - counterfactuals generated by a model *not* under test, with years excluded and fallbacks
    logged.
- **Models:** the three local models, with the quantization stated.
- **Cost estimate:** 361 × 2 × 2 × 2 × 2 = 5,776 prompts per model, plus 361 closed-book.
  From the 2026-10-13 chain timing, the 14B is plausibly a few hours. Measure on 50 prompts
  before committing.
- **Paper changes:**
  - RQ1 becomes a replication;
  - RQ3 is cut to one paragraph;
  - HoH, SDP, DRAGged, WikiContradict, Temporal Wiki, FRESCO and Kortukov et al. are cited
    accurately.

## 10. Corrections required in RESEARCH_HANDOFF §0 (pending the user's approval)

- §0.1 "Ours: 1. real revisions with an oracle derived from the old/new diff" → credit HoH's
  method; our addition is the metamorphic and mutant-validity framing.
- §0.2 RQ1: report the stale rate under dated labels as the primary result. Describe the
  no-label rate as confounded by missing recency cues and by position. Do not call "both" a
  failure.
- §0.2 RQ2: "CN overstates the real failure rate for phi3 (+15 points)" → "under ASE's oracle
  CN tracks real failures within 0.05–0.06 for both models (post hoc). The registered contrast
  mis-mapped ASE's oracle."
- §0.2 RQ3: drop "M4 catches it" as a result; state that M1, M3 and M4 are uninformative or
  circular by construction.
- §0.3: add the quantization; add the distinct-question n (361); label M3b as per generator.
- §5 table: add HoH's real-diff method, Temporal Wiki, FRESCO, DRAGged, WikiContradict and
  Kortukov et al.; fix the SDP rows (geometry; retrieval selection).
