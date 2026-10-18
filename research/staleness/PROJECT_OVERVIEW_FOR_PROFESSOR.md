# Project overview: from "drift-aware RAG" to testing the testers

*Prepared 13–14 October 2026 for a meeting with the professor.*
*Team: Aditi Saxena, Suvidh Jain, Aryan Garg. Authorship to be agreed among us.*

---

## In one paragraph

We set out to detect, without labels, when a retrieval-augmented AI system (RAG) starts giving
worse answers because its documents have changed.
- **That turned out not to work, and others had largely solved the parts that do work.** So we
  narrowed the question to what happens when a fact in a document is updated.
- **Our first answer-level results mostly repeated prior work, and some of our own measurements
  were flawed.** We found this in a strict self-review and corrected it.
- **Along the way we found a genuine gap.** A 2026 software-engineering paper (ASE 2026) offers a
  testing method for exactly this situation, and its main test mislabels correct behaviour as
  failure. On the authors' own published data, at least 77% of the faults that test reports are
  correct answers.

We are now building and evaluating a valid version of that test. The target is a mid-tier
software-engineering journal, with a full draft by mid-November 2026.

---

## 1. The problem in plain words

A **RAG system** answers questions by first *retrieving* relevant documents, then letting a
language model *write* an answer from them. Documents change: a company's revenue is updated, a
country's population estimate is revised.

When a document changes, two things can go wrong:
1. **Stale index:** the system still holds the *old* version and answers with an outdated fact.
2. **Conflicting versions:** the system holds *both* old and new versions and must decide which
   to believe.

How do you **test** whether a RAG system handles updates correctly? That is where we ended up.

---

## 2. How we started: label-free drift detection (summer 2026)

- **Original idea** (the repository's name): watch the *embeddings*, the numerical
  representations of the documents. If their distribution shifts ("drift"), raise an alarm that
  answer quality may be degrading. No human labels needed.
- **Origin:** Aryan built the original pipeline. Suvidh ran an independent audit and the first
  experiments.
- **What we learned:**
  - **Detectors can't see the changes that matter.** When a single number in a document changes
    (e.g. revenue 3.71 → 4.25 billion), its embedding barely moves. A drift detector sees almost
    nothing, even though the answer is now wrong.
  - **The general problem is already solved, or provably limited.** Label-free drift detection
    is well covered by existing tools and theory, and there are known limits to estimating
    accuracy without labels.
- **Outcome:** a well-documented **negative result**. We kept it (in `quarantine/FINDINGS.md`)
  rather than discarding it.

## 3. Second step: staleness at the answer level (early October)

- **What we measured:** a stale index **still retrieves the right document** about as often as a
  fresh one, but that document now carries the **outdated fact**, and the model repeats it.
  Retrieval metrics therefore cannot detect staleness.
- **Two practical fixes we tried, each with pass/fail rules set beforehand; both failed**
  (recorded as "no-go"):
  - **smarter re-fetch scheduling:** predicting which documents' *facts* will change did not beat
    simple baselines;
  - **"keep the old vectors, refresh only the text":** it saved computation only by losing answer
    accuracy.
- **What we learned:** the core finding is real, but it had already been shown by others (the HoH
  benchmark, ACL 2025, and "Stale-Document Poisoning", 2026). Not enough for a paper on its own.

## 4. Third step: testing RAG with real document updates (October 10–13)

We then targeted a recent software-engineering paper: **Kim, Pasini & Tonella, "When Knowledge
Changes: Metamorphic Testing of RAG Systems with Mutations" (ASE 2026)**.
- **What their method does:** it tests RAG systems by *mutating* the documents (e.g. injecting a
  made-up counterfactual fact) and checking whether the answer changes as expected.
- **Our improvement idea:** use *real* updates instead of invented ones.
- **What we built:**
  - a pipeline that takes **real Wikipedia revisions** (e.g. a country's GDP before and after an
    update) and turns them into test cases;
  - we verified it against the live Wikipedia API: the stored revisions are byte-identical to
    Wikipedia's;
  - we ran three local open models (phi3-mini, Qwen2.5-7B and Qwen2.5-14B). All work is on a
    laptop; no paid services.

**First results:** models often picked the outdated value when both versions were present. The
synthetic method seemed to overestimate failures.

## 5. Turning point: a strict self-review (October 13)

Before going further, we deliberately tried to *break* our own results:
- re-derived every number with independent scripts;
- read the reference papers in full;
- hand-checked model outputs.

**What we learned, honestly:**

| What we had claimed | What we found | Status |
|---|---|---|
| "Real revisions + automatic oracle" is our innovation | The HoH benchmark (ACL 2025) already does this at large scale | credited to HoH |
| Models pick the outdated value 19–31% of the time | In 61% of those cases nothing in the text says which value is newer, so the "right" answer is undecidable. The choice also follows the *position* of the passage | reframed |
| The synthetic method overestimates failures by 15 points | We had mapped the published paper's definition of failure backwards. With the correct mapping, the two agree within 5–6 points | **corrected** |
| Monitoring tools "can't see" the failure | One monitor could not fail by design; one was circular (it was told the answer key); one flagged almost every answer; the drift test compared near-identical texts | dropped |

Every correction is recorded, with the old and new values, in our files. That includes the
`CRITICAL_REVIEW` file, a "locked claims" register, and an automated checker that re-verifies
every quote and code citation (29/29 pass).

## 6. The gap we found (October 13)

While re-reading the ASE 2026 paper and its **released code** line by line, we found that its
key test does not measure what the paper says it measures.

- **What the paper says:** after a counterfactual fact is injected, the system should "align with
  the injected fact".
- **What the code checks:** only that the answer *changed*. It never compares the answer with the
  injected fact.
- **Consequences:**
  - a system that **keeps the correct answer** (resisting a planted claim that the generator was
    told to make "mislead a reader") is counted as **faulty**;
  - **any changed answer passes, even a wrong one**;
  - the paper's own examples contradict each other: one calls *ignoring* the counterfactual a
    fault, another calls *adopting* it a fault.

**Measured on the authors' own released data** (their GPT-5-mini outputs, downloaded from their
Figshare archive; our recount reproduces their Table 1 exactly in all 10 cells):
- **At least 77% (460 of 594) of the faults their counterfactual test reports are correct
  answers.**
- **26% (694 of 2,671) of their "counterfactual mutants" are not counterfactuals at all.**
  Corrected on 17 Oct; an earlier count of 12.5% missed refusals written with a curly
  apostrophe. In some settings it is almost half. The
  model refused to write misleading text ("Sorry, I can't help create misleading
  counterfactuals…"), and the refusal was used as the mutation anyway. In one setting it is
  27%.

**What we checked and did *not* find** (reported just as plainly):
- the placement of the injected text does not affect their verdicts;
- when their test says "pass", the answer usually does follow the injected fact.

We checked novelty carefully. The three papers that cite the ASE paper do so neutrally, and we
found no prior report of these problems.

## 7. Where we are headed

**The paper (working title):** *"When the Oracle Is the Fault: Validity of Metamorphic Relations
for Knowledge-Conflict Testing of RAG Systems."*

1. **Validity rules** for this kind of test. A test is valid only if:
   - the input determines the correct answer;
   - it checks agreement with the injected value, not just "did it change";
   - it is not sensitive to where text is placed;
   - an "update" carries information about which version is current (e.g. a date).
2. **Audit** of the ASE 2026 framework against these rules: done, code-verified.
3. **Measured impact** on the authors' own results: done (section 6).
4. **A valid replacement test (running now).** We attach dates to the two conflicting values,
   then swap only the dates. A correct system must switch its answer. This needs no answer key,
   and it works on real Wikipedia updates as well as on invented ones.
   - The experiment is pre-registered and running overnight: 392 questions × 3 models.
   - It answers the decisive question: **does a valid test find different faults, or rank models
     differently, than the published one?**
5. **Human check:** Suvidh and Aryan will each label about 200 cases, to confirm the automatic
   judgements.

**Timeline:**

| When | What |
|---|---|
| Week of Oct 13 | Audit and impact measurement done; date-swap experiment running |
| Oct 20–26 | Analyse the date-swap results; prepare cases for human labelling |
| Oct 27 – Nov 2 | Human labels; write all sections |
| Nov 3–13 | Internal review; re-check novelty; inform the ASE authors of our findings (standard courtesy); submit |

**Target venue:** a mid-tier software-engineering journal, such as the *Journal of Systems and
Software*, *Information and Software Technology*, or *Software Testing, Verification and
Reliability*.

---

## 7a. Latest result (overnight experiment finished 14 Oct, 04:06)

**Design.** The "date-swap" test was pre-registered: 392 questions built from real Wikipedia
updates, 3 models, about 15,000 answers in total.

| Model | Published test: fault rate (real updates) | Our date-swap test: fault rate |
|---|---|---|
| phi3-mini | 30% | 51% |
| Qwen2.5-7B | 8% | 41% |
| Qwen2.5-14B | 1.5% | 1.7% |

**In plain words:**
- **Small models.** The published test badly underestimates how often they ignore which fact is
  newer (8% vs 41% for Qwen-7B), and it flags *different* questions from the ones that are
  really faulty.
- **The largest model.** It rarely ignores the dates, but in about 86% of cases it simply
  **lists both values** instead of giving the current one. Neither test counts that as a fault.
  Our test reports it separately; the published test silently calls it a pass.
- **What didn't hold.** One pre-registered prediction failed. We had predicted the two tests
  would disagree on *which* cases are faulty **for every model**. That holds for the two smaller
  models but not for the 14B, where both tests find almost nothing. We report this as a failed
  prediction.
- **Ranking.** Both tests rank the three models in the same order; they differ in how big the
  problem is, not in which model is better.

- **Follow-up (17 Oct, pre-registered).** The 14B's listing of both values was mostly caused by our
  prompt, which allowed "number(s)". With a "give exactly one number" prompt, listing fell from 86%
  to 32%, and the model then followed the newer date in 59% of cases. It ignored dates only 9% of
  the time, versus 41–51% for the smaller models.
- **A pilot that failed its gate (17 Oct).** We tried comparing answering *strategies*: "careful"
  vs "gullible" prompts. The pilot failed its pre-set gate, because the small model did not follow
  a "use the newest date" instruction. As pre-registered, we did not run the full experiment, and
  the paper reports this.

## 8. What we learned about doing research

- **Check the source, not the summary.** Our most important finding came from reading the other
  paper's *code*, not its text.
- **Try to break your own results before anyone else does.** Our self-review removed three
  claims that a reviewer would have rejected.
- **Pre-register.** We write the hypothesis and pass/fail rules *before* running, record a
  fingerprint (hash) of the script, and report results whichever way they go, including null
  results and measures that failed.
- **Negative results are results.** Drift detection, the scheduling idea and the keep-vectors
  idea all failed, and each failure narrowed the question until we found a real gap.

## 9. Honest limitations

- **Scope of models and frameworks:** we test small open models on a laptop, and we audit one
  framework in depth.
- **Partly predictable parts:** some of the critique follows directly from the framework's own
  definitions. Its value is in exposing and measuring it, and in proposing a valid alternative.
- **A known mechanism:** that models do not fully use dates is already known. We use it only as
  the *mechanism*, not as a claim of novelty.
- **Versions and verification:** results refer to the arXiv v1 version and the released code (one
  commit). Novelty will be re-checked just before submission.

## 10. Questions we would value your advice on

1. Is the framing as a **test-validity** paper (software-testing journal) the right one, or would
   you position it differently?
2. Is auditing one framework in depth plus general validity rules sufficient, or should we audit
   a second framework?
3. Should we contact the ASE authors **before** submission (our current plan) or at submission?
4. Any guidance on authorship and order among the three of us?

---

*Supporting files (in `research/staleness/`):*
- `CLAIMS_LOCKED_2026-10-13.md`: every claim with evidence;
- `CRITICAL_REVIEW_2026-10-13.md`: the self-review;
- `PUBLICATION_ASSESSMENT_2026-10-13.md`: the venue and novelty assessment;
- `oracle_audit/PAPER_DRAFT.md`: the paper skeleton;
- `verification_2026-10-13/check_claims.py`: the automated evidence checker.
