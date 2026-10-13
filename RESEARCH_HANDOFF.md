# Research handoff: Drift-aware RAG → real-revision metamorphic testing of RAG

**Read §0.0 first in every new chat** (critical review, 2026-10-13 late). It overrides the
claims in §0.1–§0.5. Sections
§1–§9 are the older history (2026-10-09 framing); they are kept for provenance, and §7 holds
every pre-registered experiment record (items 0, 0b–0h).

Companion files: `PAPER_BASIS.md` (the answer-level staleness result),
`research/staleness/pretests/` (every pre-test script, log and pre-registration),
`quarantine/_AUDIT.md` (cleanup record), `quarantine/FINDINGS.md` (the original negative results).

---

## 0.0 STATUS AFTER THE CRITICAL REVIEW (2026-10-13, late). This overrides §0.1–§0.5

Read these three files before anything else:
- `research/staleness/CRITICAL_REVIEW_2026-10-13.md`: what fails and why, with scripts and
  quotes;
- `research/staleness/PUBLICATION_ASSESSMENT_2026-10-13.md`: novelty, venues, IP and plan;
- `research/staleness/VERIFICATION_2026-10-13.md`: the numbers reproduce from the records.

The data pipeline is genuine and every §0.3 number reproduces. The *interpretation* in
§0.1–§0.2 does not hold up; the original text below is kept for provenance:

| §0 claim | Status | Correct statement |
|---|---|---|
| Real revisions + diff oracle are our innovation | **wrong** | This is HoH's method (ACL 2025). Credit it. |
| RQ1 stale 0.31 / 0.19 (no labels) | **not a fault rate** | 61% of these cases (368/605) have no recency cue; the choice follows position (phi3 picks the first-listed value 62%). Report the **dated-label** rates as primary: phi3 0.20, Qwen-7B 0.22 (pre-registered). These replicate HoH and SDP. |
| Qwen-14B "lists both in 84%" as hedging or failure | **mis-framed** | Under ASE §5.2.3, acknowledging the conflict satisfies the relation; the prompt allows lists. |
| RQ2: CN overstates phi3 by +0.15 | **reversed** | The registered contrast mis-mapped ASE's oracle. Under ASE's oracle: phi3 +0.061, Qwen-7B +0.048, so CN is a fair proxy (post hoc). |
| RQ3: monitors blind; an informed judge sees the fault | **uninformative by construction** | M1 cannot fail; M4 is given the diff (circular); M2 flags about 84% of everything; M3a is borderline for bge. |
| Models | add | All 4-bit (phi3 Q4_0; Qwen Q4_K_M). 605 cases = 361 distinct questions. |

**RUNNING JOB (started 2026-10-13 21:21 IST): p15 MR-Provenance-Swap**, detached via nohup + caffeinate.
- Log: `research/staleness/pretests/p15_provenance_swap.log`; chain:
  `pretests/chain_p15_2026-10-13.log`; pre-registration hash in `pretests/p15_prereg.log`.
- Check with `pgrep -fl p15_`. Answers are cached, so re-running the same command resumes it.
- Results so far on the ASE authors' own data are in `CLAIMS_LOCKED_2026-10-13.md` §F.

**ACTIVE PLAN (2026-10-13 21:05): journal paper in one month.**
- Draft and schedule: `research/staleness/oracle_audit/PAPER_DRAFT.md`.
- Pre-registered analysis of the ASE authors' own outputs:
  `research/staleness/oracle_audit/analyze_ase_outputs.py` (hash in `oracle_audit/prereg.log`).
  It waits for the user to download the Figshare archive (https://figshare.com/s/96fc3abb168180c0a215,
  which blocks bots) to `~/SemProject/ase-replication/output.zip`.
- The ASE code is cloned at `~/SemProject/ase-replication` (commit 597fd65).

**Newest direction (2026-10-13, 20:30): see `research/staleness/PROPOSAL_ORACLE_SOUNDNESS_2026-10-13.md`.**
- ASE's knowledge-conflict oracle is unsound (code-verified defects D1–D5) and order-dependent
  (the 0.705 vs 0.220 order figure was WITHDRAWN at 20:40; see `research/staleness/CLAIMS_LOCKED_2026-10-13.md`).
- Randomized confirmation: `pretests/p14_order_oracle.py`, pre-registered (sha256 in
  `pretests/p14_prereg.log`), NOT YET RUN.
- This supersedes the mutant-validity plan below, which becomes a secondary analysis.

**Earlier direction (2026-10-13 late; superseded by the line above).**
- Reframe the paper around **mutant validity**: is ASE's CN operator a valid substitute for real
  knowledge updates? (cf. Just et al., FSE 2014).
- Test it in a pre-registered, counterbalanced, paired, multi-configuration study:
  - order, date-cue and prompt factors;
  - ASE's oracle, with ASE's released CN artefact;
  - a closed-book filter;
  - baselines: first-listed, larger value, random;
  - a second domain;
  - human labels.
- Preferred route: **MSR 2027 Registered Reports → EMSE** (abstract 13 Nov 2026, Stage 1
  20 Nov 2026).
- The old p14 (prompt-only) is **dropped**; its question becomes one factor of the new design.
- Patent: not recommended. The repository is public.

---

## 0. PREVIOUS STATE (written 2026-10-13 evening; §0.1–§0.5 SUPERSEDED by §0.0; §0.6–§0.7 still apply)

### 0.1 Vision, in one paragraph
RAG systems answer from documents that change. When a document is updated, an index may
keep **both the old and the new version**. (We have not verified how common this is in real
frameworks; do not claim it as "the default".) We test what the generator then does, using
**real Wikipedia revisions** rather than invented changes. The study improves on Kim,
Pasini & Tonella, *"When Knowledge Changes: Metamorphic Testing of RAG Systems with
Mutations"* (ASE 2026, arXiv 2607.26843).
- **Their method:** they inject an LLM-written counterfactual passage ("CN" operator) and
  check whether the answer follows it.
- **Ours:**
  1. real revisions, with an oracle derived from the old/new diff;
  2. a measurement of whether their synthetic operator estimates the real failure rate;
  3. a test of which monitors can see the fault.
- **Target:** a software-engineering journal (JSS/IST realistic; EMSE/TOSEM only if the
  hedging result holds up).

### 0.2 The result in plain words (all numbers are in §0.3 with their sources)
1. **RQ1, real updates.** With both versions in the top-3 context, the small models give
   only the old value often: phi3 31%, Qwen-7B 19%. The pre-registered stale rate does
   **not** replicate on Qwen-14B (4%), which instead **lists both values in 84%** of cases.
   Only 10–45% of answers commit to the new value alone. The hedging finding is **post hoc**,
   and our prompt allows "number(s), separated by commas", so p14 must test it.
2. **RQ2, synthetic vs real.** On the same pages, their CN operator overstates the real
   failure rate for phi3 (+15 points) and is close for Qwen-7B (+4).
3. **RQ3, monitors.** These cannot see the fault:
   - retrieval (hit@3 = 1.0 for every index);
   - drift detectors (no significant shift);
   - embedding displacement (AUROC about 0.5);
   - a faithfulness judge (FPR about 0.85).
   An ASE-style judge that is **told the change** catches it (TPR 0.99–1.00, FPR
   0.01–0.02). It agrees with the free diff oracle 98.9% of the time, so knowing the diff is
   the signal.

**Correct statement of the ASE paper.**
- **Their setup:** default top-k = 1; GPT-5-mini; text-embedding-3-small.
- **Their numbers:** the CN violation rate peaks at 27.6% (VQAonBD, pre-chunk).
- **Their scoring:** exact numeric match on their financial datasets; an LLM judge checks
  that a mutation preserves the meaning.
- **Do not claim** they "only use an LLM judge".
- **Their CN operator is explicitly framed as stale data** (v1 vs v2), and its expected
  behaviour is ambiguous.

These facts were checked on 2026-10-10 from the arXiv HTML. Re-verify before writing.

### 0.3 Evidence (pre-registered unless marked post hoc; all from logs/records listed in §0.6)
| Test | Data | Result |
|---|---|---|
| p6 (retrieval only) | arXiv + Python docs, both versions indexed, k=1 | retrieved chunk carries only the old value 26–42%; replace-in-place 0% |
| p7 | 161 both-in-context cases | stale 17% with phi3 and with Qwen-7B (adequacy 0.70 / 0.78) |
| p8 | same 161, version labels | labels cut stale to 7–11%, not to 0 |
| p11 run 1 (2026-10-12; sample later found unstable) | 826 Wikipedia pages, 609 cases, 144 pages | stale 0.29 (phi3, adequacy 0.98), 0.21 (Qwen-7B, adequacy 0.80) |
| **p11 rerun (fixed sample, 2026-10-13)** | same pages, **605 cases, 142 pages**, k=3, minilm + bge | Pre-registered stale: **phi3 0.31** [Wilson 0.27–0.35; page-boot 0.27–0.35], **Qwen-7B 0.19** [0.16–0.23] → replicates. **Qwen-14B 0.03 [0.02–0.05] → does NOT replicate** (adequacy 0.91). Dated labels: phi3 0.31 → 0.20 (McNemar p = 1e-7); Qwen-7B 0.19 → 0.22 (p = 0.28); 14B not run with labels (scope cut decided before results). |
| p11 diagnostic (**post hoc**) | same records, `P9.commit` | new only / **old only** / **both** / neither. phi3 0.45 / 0.31 / 0.20 / 0.04; Qwen-7B 0.24 / 0.20 / 0.43 / 0.13; **Qwen-14B 0.10 / 0.04 / 0.84 / 0.02**. Replace control: old only = 0.00 for all. |
| p12 (RQ2) | CN vs real, same 400 questions, `P9.commit` on both sides | phi3: CN injected-only **0.45** [0.41–0.50] vs real old-only 0.31 → **+0.15, CN overstates**; Qwen-7B 0.24 vs 0.20 → +0.04, fair proxy. "Both": CN 0.16 / 0.40, real 0.20 / 0.43. Not paired (708 vs 605 cases). 400/400 counterfactuals had a fresh value (fallback count not logged; check). |
| p13 (RQ3) | p11 append cases, old-only (positive) vs new-only (negative); 376 "both" excluded; phi3 + Qwen-7B | M1 hit@3 = 1.000 under stale, append and replace indexes. M3a: sliced-W1 p = 0.70 / 0.105, MMD p = 0.12 / 0.09 (minilm / bge; 199 permutations; n = 4,834 changed passages, subsampled to 1,000). M3b AUROC 0.465 / 0.502. M2 faithfulness TPR 0.83 / 0.88, FPR 0.85 / 0.84. **M4 TPR 1.00 / 0.99, FPR 0.02 / 0.01, agreement 0.989.** |
| p11 FreshQA (descriptive) | 23 real human questions; 10 both-in-context cases on 6 pages | stale phi3 4/10, Qwen-7B 2/10, Qwen-14B 0/10 (14B correct 7/10); with dates 0/10 for phi3 and Qwen-7B |

**Void, failed or superseded (all disclosed):**
- p9: void twice (a year bug; then adequacy 0.51). p11 is the post-void redesign.
- E0 (harm-aware scheduling) and p5 (keep-vectors): no-go.
- Drift-detector variants and agentic RAG: already in the literature.

### 0.4 Fixes made on 2026-10-13 (disclose in the paper)
1. **Sample order.** `p9.build()` iterated over a set, so each process drew a slightly
   different 400-question sample (about 20% of prompts differed).
   - Fixed with `sorted`; the sample hash is identical under PYTHONHASHSEED 1, 2 and 3.
   - Run-1 records are kept: `p11/records_*_run1_*.jsonl`.
2. **Scorer.** `facts.NUM` read "36.77,29.86" as one number, and `generation.classify`
   counts "both" as correct.
   - `P9.value_in` / `P9.commit` match at digit boundaries; checked on 8 hand cases.
   - p12 and p13 were amended to use `P9.commit` before they produced any result. The
     amendment notes are in their docstrings.
   - The p11 pre-registered verdicts still use the old classifier, as registered.
3. **p12 asymmetry.** "Both" counted as a violation for CN but as correct for real updates.
   Fixed before results (same amendment).
4. **p13.**
   - Labels now come from `P9.commit`, and the run is restricted to the pre-registered
     generators.
   - It crashed once on a `median_heuristic_sigma(X, Y)` call bug
     (`p13_monitors_CRASH_mmdsig.log`), which was fixed and rerun.
5. **Speed.** Generators now run one at a time and are unloaded afterwards; three models
   overflowed 16 GB of RAM.
   - Long chains run detached (`nohup`), with Low Power Mode off.

Nothing is running now (chain log: `research/staleness/pretests/chain_2026-10-13.log`).
Check with `pgrep -fl "p1[1-9]_"` before starting anything.

### 0.5 Next steps, in order
1. **Verify** (§0.6) before any new experiment or writing.
2. **p14: single-answer prompt.** Pre-register it first.
   - Same 605 append cases and 750 replace cases, same contexts, all 3 generators.
   - The prompt asks for exactly one number.
   - Decides whether the 14B "both" pattern is prompt-induced. Reuse retrieval; only
     generation is new.
3. **RQ1 robustness:** a k = 1/3/5 sweep; the full ~1,400-page set (`p9_wiki_infobox.py
   fetch` is resumable); a second prompt template.
4. **RQ2 paired analysis:** the same (retriever, question) pairs in CN and real. Mark it
   post hoc, or pre-register it.
5. **RQ4 mitigations:** replace, labels, a recency instruction, re-ranking.
6. **Human check:** 100 outputs graded by two annotators (Suvidh and Aryan), with agreement
   against `P9.commit`.
7. **Code:** promote the pretests to tested modules with `results_*.json`.
8. **Then:** write the paper; release the code and dataset; agree authorship.

### 0.6 Verification protocol (run in every new chat before claiming anything)
- **Recompute from raw data, not prose.**
  - Records: `embeddings/cache/staleness/p11/records_scale.jsonl`, `p12/records.jsonl`,
    `p13_cases.jsonl`, `p13_results.json`.
  - Logs: `research/staleness/pretests/*.log`.
  - Write an independent script; do not import the experiment's report code. Every number
    in §0.3 must match to 2 decimals.
- **Read real outputs.** Print 30 random records per generator (question, context, output,
  label) and hand-check the label.
- **Check that the data is real.** For 5 random pages, re-fetch the two revisions from the
  MediaWiki API by revision timestamp. Confirm that the old and new values appear in them.
- **Check the literature against the source.** Re-read arXiv 2607.26843 and confirm every
  sentence in §0.1/§0.2 about it. Do the same for any other paper we cite.
- **Check pre-registration order.**
  - Script docstring dates must precede the result logs (mtimes and `chain_*.log`).
  - Amendments must precede the first result of that script.
- **Check the controls.** Replace old-only must be 0.00; adequacy must be ≥ 0.60.
- **Run the tests:** `DRIFT_MINILM_DIR=~/.cache/minilm-l6-v2 .venv/bin/python -m pytest`.
- **Report.** Anything that does not match is reported to the user, not fixed silently.

### 0.7 Guardrails (non-negotiable)
1. **Pre-register** every experiment in its script docstring before running.
   - Adequacy ≥ 0.60 is mandatory.
   - Never change criteria after seeing results; report void runs.
   - Mark post-hoc analyses as post hoc.
2. **Disclose** redesigns (p9 → p11), scope cuts (14B), and the fixes in §0.4.
3. **No paid APIs.** Use local Ollama only (phi3:mini, qwen2.5:7b-instruct,
   qwen2.5:14b-instruct).
   - Run one model at a time.
   - Plugged in, lid open, Low Power Mode off; run long chains detached with `nohup`.
4. **Don't commit or push** unless asked. The origin is Suvidh Jain's; authorship must be
   agreed with him and Aryan Garg.
5. **Don't delete anything.** Stale material moves to `quarantine/`.
6. **No overclaiming.**
   - Small local models; one domain (Wikipedia infobox numbers); one prompt template so far.
   - FreshQA is illustrative.
   - "Consistent with", not "proves".
7. **Numbers come from logs and records**, never from prose, including this file.


### 0.8 Prompt for the next session (written 2026-10-13)

```text
Continue the Drift-aware RAG project. Use the rag-staleness-research skill and read
RESEARCH_HANDOFF.md §0 completely before doing anything else.

Your first job is to VERIFY, not to extend. I need to know that every claim we make is real,
grounded in data on disk or in a source you have actually opened, and not a hallucination or
an artefact. Treat §0 (and everything a previous chat wrote, including memory) as unverified
claims to check, not as facts.

PHASE 1: VERIFICATION (no new experiments, no edits to experiment code)
1. State of the machine. Confirm nothing is running (`pgrep -fl "p1[1-9]_"`). Run
   `git status`. Run the test suite:
   `DRIFT_MINILM_DIR=~/.cache/minilm-l6-v2 .venv/bin/python -m pytest`.
2. Recompute every number in §0.3 independently.
   - Write a fresh script in the scratchpad. Do not import the experiments' report code.
     Read the raw records directly: embeddings/cache/staleness/p11/records_scale.jsonl,
     p11/records_fresh.jsonl, p12/records.jsonl, p13_cases.jsonl, p13_results.json.
   - Rebuild the 400-question sample and check that every record's `qtext` matches it.
   - Produce a table: claim | value in §0.3 | recomputed value | source file | MATCH or
     MISMATCH.
   - Also compare each claim with the matching line in research/staleness/pretests/*.log.
3. Read real outputs. Print 30 random append records per generator (question, the 3
   context passages, raw output, old/new values, `P9.commit` label) and judge each label
   yourself. Report the disagreement rate, with examples.
4. Check that the data is real. For 5 random pages from p11_wiki_frozen:
   - re-fetch both revisions from the live MediaWiki API, using the stored revision
     ids/timestamps;
   - confirm the old value is in the old revision and the new value in the new one;
   - confirm the question's field really changed.
5. Check the literature against the sources. Open arXiv 2607.26843 (the HTML version) and
   quote the exact sentences that support or contradict each statement about it in §0.1
   and §0.2: top-k, models, the 27.6% figure, exact numeric matching, LLM judge, the CN
   operator, "v1 vs v2". Do the same for every other paper the handoff or the team doc
   cites: Stale-Document Poisoning (2609.31342), HoH (ACL 2025), FreshQA (2310.03214),
   Podkopaev & Ramdas, LegalSearch-R1, "Revoked but Still Authoritative". If you cannot
   open or find a paper, say so. Never fill the gap from memory.
6. Check pre-registration order.
   - Compare each script's docstring dates and amendment notes with its log times
     (research/staleness/pretests/chain_2026-10-13.log, file mtimes).
   - Be explicit about the limit: the pretests are uncommitted, so there is no git
     timestamp, and the docstring dates are self-reported.
   - Suggest how to make future pre-registrations verifiable, e.g. commit or hash the
     script before running.
7. Check controls and leakage.
   - Replace-policy old-only must be 0.00, and adequacy must be ≥ 0.60.
   - The question text must not contain the answer.
   - Years must be excluded from values.
   - In every "both-in-context" case, both values must actually be in the context.
   - Count how many p12 counterfactuals used the ×1.13 fallback instead of the LLM.
8. Write the report to research/staleness/VERIFICATION_<today>.md:
   - pass or fail per claim, with evidence (file + line, or URL + quote);
   - a separate list of anything that could not be verified.
   Then tell me in plain words what holds up, what doesn't, and what changed. Do not fix
   anything silently. If a number is wrong, show the old and the corrected value and
   update §0 only after telling me.

PHASE 2: THE NEXT EXPERIMENT (only once Phase 1 passes, and only after I approve)
9. Pre-register p14, the single-answer prompt test.
   - Put it in the script docstring before any run: hypothesis, rules, adequacy,
     kill/decision criteria.
   - Same 605 append and 750 replace cases, identical contexts, all three generators
     (phi3:mini, qwen2.5:7b-instruct, qwen2.5:14b-instruct). The only change is a prompt
     asking for exactly one number.
   - Score with `P9.commit` (new / old / both / neither).
   - The question it must decide: is the 14B's 84% "both" caused by our prompt ("number(s),
     separated by commas"), or by the model? Propose the decision thresholds and show them
     to me before running.
10. Before running: tell me to plug in, keep the lid open and turn Low Power Mode off.
    - Run one generator at a time, detached with nohup, with a chain log.
    - Report void runs and keep every log.

PHASE 3: AFTER p14 (propose each one, pre-register it, then run)
11. RQ1 robustness: a k = 1/3/5 sweep, the full ~1,400-page set (`p9_wiki_infobox.py fetch`
    resumes), and a second prompt template.
12. RQ2: a paired comparison of synthetic vs real on identical (retriever, question)
    cases.
13. Human check: export 100 stratified cases as a CSV for Suvidh and Aryan to label (new /
    old / both / neither), then report agreement with `P9.commit`.
14. Promote the pretests to tested modules (research/staleness/<topic>.py, tests,
    results_*.json).

RULES
- No paid APIs (local Ollama only).
- No commits or pushes unless I ask.
- Never delete anything (move to quarantine/).
- Never change criteria after seeing results; label post-hoc analyses as post hoc.
- Paper authorship is to be agreed with Suvidh Jain and Aryan Garg.
- Every number you tell me must come from a file you read or a script you ran in this
  session. Say "I could not verify X" rather than guessing.
- Explain results to me in plain words, then the numbers.
```

---

## 1. Where things stand

- **Repo:** `SuvidhJ/Drift-aware-RAG-Label-free-detection-of-retrieval-degradation`
  (`origin`). All staleness work is on branch **`staleness-answer-level`** (commit
  `d3ddd5d`), pushed, **no PR opened**. `research/staleness/pretests/` and this file are
  uncommitted.
- **Authorship is unresolved.** The motivating negative results (`quarantine/FINDINGS.md` §3–§13) are
  Suvidh Jain's, on Aryan Garg's base pipeline. Settle this before submission.
- **Established result** (tested module, controls, two models, two corpora): a fully stale
  index retrieves the right document as often as a fresh one on questions whose answer
  changed (hit@10 gap −0.029 to +0.010), but carries the correct value far less often
  (answer@10 gap +0.52 to +0.79). A 1.5B generator states the outdated value (arXiv: 81% of
  committed answers vs 0% from the fresh index; measured before the 2026-10-09 cleanup and
  due for regeneration). Controls agree within ±0.013. Full table
  in `PAPER_BASIS.md` §4.

## 2. The proposed paper

**Working title:** *Fact-Invariant by Construction: Why Dense Retrieval Cannot See Corpus
Staleness, and Why Fixing It Does Not Require Re-embedding*

**Thesis:** dense encoders are nearly invariant to the content that makes a document
stale (numbers, and anything past the token window), and a revision moves a document
almost at right angles to the queries about it. That one geometric fact explains:

1. why retrieval metrics cannot measure staleness (established result + a margin bound);
2. why label-free, embedding-distribution drift detectors failed (`quarantine/FINDINGS.md`); this is a
   transport bound, **not yet checked numerically**;
3. why the index cannot order two versions of a document when both are present (the
   framework default);

and it implies a practical policy:

4. **refresh the text, keep the vector**. This gives fresh-index answer accuracy at zero
   re-embedding cost, with re-embedding triggered only by new content.

## 3. The mathematics

Unit-norm embeddings, revision displacement `Δ = d₁ − d₀`.

- **Score identity:** `s(q,d₁) − s(q,d₀) = q·Δ`, so `|Δs| ≤ ‖Δ‖` by Cauchy–Schwarz.
- **Certified invariance:** with competitors fixed, hit@k cannot change if
  `‖Δ‖ < m_k(q) = |s(q,d₀) − s_(k)(q)|`, the margin to the k-th competitor. This is the
  standard robustness-radius argument (cite RetrievalGuard / RobustMask); it is **not**
  new math. What is new is applying it to revisions, decomposed as
  `Δ = Δ_num + Δ_rest`.
- **Random-projection model:** `q·Δ ≈ N(μ, ‖Δ‖²/d_eff)`, with `d_eff` the participation
  ratio of the document embeddings. Predicted flip rate `≈ mean Φ̄(|m_k|/σ)`. An
  order-of-magnitude model only (within ×2), not a law.
- **Detection bound:** pairing each document with its own revision gives
  `W₁(P₀,P₁) ≤ (1/N) Σᵢ ‖Δᵢ‖`. For numeric-only edits this is at most
  `(fraction edited) × ‖Δ_num‖`; this needs comparing with the detectors' null noise floor.
- **Utility model:** expected stale-answer rate
  `= Σᵢ λᵢ · pᵢ · P(retrieve i) · P(commit | stale context)`. Here `λᵢ` is query rate and
  `pᵢ` is the probability of a fact edit. Payload refresh drives `pᵢ` to 0 at text-update
  cost; re-embedding only changes `P(retrieve i)`, which §4 shows barely moves.

## 4. Pre-test evidence (exploratory; `research/staleness/pretests/`)

All per document, MiniLM and bge-small (both 384-d), arXiv (6,247 papers after dedup)
and Python docs (Sphinx footer cut). These are **not** paper results until rebuilt as a
tested module. Rerun on 2026-10-09 after the corpus cleanup, with every perturbation
classified by whether the encoder can see it (`data.visible`: token ids inside the
256-token window change). "Inside" numbers exclude edits past the window, which cannot
move the vector at all. Settings in the order MiniLM arXiv, bge arXiv, MiniLM Python
docs, bge Python docs.

| Idea | Pre-test result | Script |
|---|---|---|
| Numeric edits are certifiably invisible | **holds on arXiv, weakens on Python docs.** Past the window: 13 of 50 arXiv and 28 of 54 Python-docs questions. Inside the window: certified top-10-invariant 0.95, 0.92 (arXiv, n=37) but 0.77, 0.77 (Python docs, n=26); top-1 0.84, 0.84, 0.58, 0.62. ‖Δ_num‖ median 0.016, 0.006, 0.061, 0.027 vs margin to 10th 0.147, 0.086, 0.153, 0.080 | p3 |
| Sensitivity depends on token type | single-token change, doc-level 1−cos, inside the window, number / word / name: 0.00044 / 0.0029 / 0.0097; 0.00004 / 0.0006 / 0.0028; 0.00064 / 0.0039 / 0.0089; 0.00009 / 0.0025 / 0.0031. The ordering holds in 4/4; 17–51% of number substitutions fell past the window | p1 |
| Truncation makes edits exactly invisible | fact-edit sentences starting past token 256: arXiv 17%, Python docs **55%**; past 510: arXiv 0%, Python docs 32% | p2, p3 |
| Fact sentence vs number only | inside the window, swapping the whole aligned sentence gives 44–70% of revision ‖Δ‖ (0.53, 0.44, 0.70, 0.67); swapping only the numbers gives 7.4× and 9.8× less than the revision on arXiv but only **2.7× and 2.3×** on Python docs (per-document medians, n=28–43) | p2 |
| Revision ⟂ queries | \|cos(q,Δ)\| median (Δ≠0) 0.073, 0.046, 0.068, 0.044, against a random-direction median 0.674/√d_eff of 0.079, 0.098, 0.079, 0.071: about as aligned as a random direction, not less. Random-projection flip prediction vs observed: 0.018/0.019, 0.033/0.020, 0.026/0.037, 0.033/0.037 | p2 |
| Dilution mechanism | signed cos(q,Δ): retained-text queries −0.008 to −0.023, new-text queries +0.055 to +0.113; all CIs exclude 0, 4/4 settings | p4 |
| Keep vectors, update text | answer@10: arXiv hybrid = fresh (0.829 / 0.800); Python docs 0.748 vs fresh 0.718, 0.728 vs 0.738; retrieval cost = stale index (arXiv −0.2 to −0.5 pt hit@1; Python docs +0.4 to +1.1 pt) | p1 |
| Novelty-triggered re-embedding | re-embed top 10% most-novel revised docs: arXiv recovers ~40% (MiniLM) to all (bge) of a small full-refresh hit@1 gain (+0.005, +0.0015); Python docs: any re-embedding hurts | p1 |
| Mixed index (both versions) | version score gap median 0.003–0.010; stale outranks fresh 30–43%; both in top-3 in 42–73% | p1 |

What changed against the earlier draft of this table: the certified-invariance and
numbers-only rows were inflated by the truncation leak (`quarantine/_AUDIT.md` §8.2) and
by footer questions. On the cleaned Python docs, the edits the encoder can see move the
document 2–3× less than the revision, not 5–25× less, and about a quarter of them can
change hit@10. The "revision ⟂ queries" row no longer supports "less aligned than
random".

## 5. Literature check (done 2026-10-09; must be extended before writing)

| Work | What it already shows | Consequence for us |
|---|---|---|
| [HoH](https://arxiv.org/abs/2503.04800) (Ouyang et al., 2025) | outdated KB entries reduce RAG answer accuracy | answer-level harm is **not novel** |
| [Stale-Document Poisoning](https://arxiv.org/abs/2609.31342) (Ahmed, Galke Poech, Röttger, 25 Sep 2026) | 317 knowledge reversals, 12 LLMs, poisoning 17–91%; recency-aware re-ranker; "dates alone produce only modest adaptation"; closed-book conditioning. **Corrected 2026-10-13:** it *does* have a geometry analysis (App. M, of internal states, not of embeddings). It *does* report up-to-date-document selection alongside poisoning (App. N, MiniLM-L6-v2, top-1). No re-embedding. | closest work; cite accurately |
| HoH, re-read in full 2026-10-13 | real Wikipedia snapshot diffs (Myers diff + LLM screening); current and outdated answers; index with both versions; Qwen-2.5-7B tested; order effects (Table 5); time-aware prompt | our "real revisions + diff oracle" is **their method**; RQ1 is a replication |
| [Temporal Wiki / QA under Temporal Conflict](https://arxiv.org/abs/2506.07270) (2025) | Wikipedia snapshots; LLMs "can be misled when multiple versions of a fact appear in context" | RQ1 not novel |
| [FRESCO](https://arxiv.org/abs/2604.14227) (2026) | historical Wikipedia revisions; re-rankers biased to older documents | retrieval side covered |
| [DRAGged into Conflicts](https://arxiv.org/abs/2506.08500) (2025) | conflict taxonomy; for outdated info "prioritize the up-to-date information …, while optionally acknowledging" | defines the desired behaviour; needs a recency cue |
| [WikiContradict](https://arxiv.org/abs/2406.13805) (NeurIPS 2024 D&B) | real Wikipedia contradictions; two-passage RAG; reflecting the conflict is the desired answer | "both" answers are not failures |
| [Kortukov et al.](https://arxiv.org/abs/2404.16032) (2024) | synthetic vs real context–memory conflicts: real updates "fail less often than previously reported"; parametric bias | precedent for synthetic-vs-real; closed-book control needed |
| Just et al., FSE 2014 | "Are mutants a valid substitute for real faults in software testing?" | the framing for our only open contribution |
| [Revealing the Numeracy Gap](https://arxiv.org/abs/2509.05691) (Deng et al., EACL Findings 2026); [NumColBERT](https://arxiv.org/abs/2605.10109) | 13 embedding models fail to encode numeric detail | numeric insensitivity is **known**; our addition is its consequence for staleness and its certified magnitude |
| [SmartVector](https://arxiv.org/abs/2604.20598) (Xu, 2026) | temporal validity in scoring; diff-based re-embedding cuts cost 77% per single-word edit; stale-answer rate 35% → 13% | selective re-embedding is **known**; *not re-embedding at all* (keep vectors) was not found |
| [NuggetIndex](https://arxiv.org/abs/2604.27306) (SIGIR '26) | atomic facts with validity intervals filtered before ranking | alternative architecture; cite as the fact-level approach |
| [Freshness in temporal RAG](https://arxiv.org/abs/2509.19376) (Grofsky) | cosine baseline scores 0.00 on freshness; recency prior helps | cosine cannot pick the latest version, which is consistent with our mixed-index result |
| [RetrievalGuard](https://arxiv.org/abs/2206.11225), [RobustMask](https://arxiv.org/abs/2512.23307) | margin/Lipschitz certified top-k robustness | the bound is borrowed; cite it |
| [When Claims Evolve](https://arxiv.org/abs/2503.03417) (2025) | embedding robustness to claim edits | related to the token-type sensitivity result |
| Practitioner posts (e.g. [Oracle on RAG index drift](https://blogs.oracle.com/developers/how-to-detect-rag-index-drift-deleted-docs-stale-chunks-and-duplicate-embeddings)) | "current versions should beat stale versions" checks | the problem is recognised; the measurement is not |

**Not found (apparent gaps):**
- a paired, same-question demonstration that retrieval metrics are blind to staleness;
- a geometric explanation of why embedding drift detectors cannot see fact edits;
- the keep-vectors / update-payload policy with an accuracy and cost evaluation.

Searches were not exhaustive. Before writing, check: temporal QA benchmarks (StreamingQA,
RealTimeQA, FreshQA), knowledge-conflict studies, "lazy re-embedding", and SIGIR/ECIR/
CIKM 2025–26 proceedings.

## 6. Publishability verdict

**As the current results stand** (answer-level harm only): not an IR-journal paper. HoH and
Stale-Document Poisoning already show the harm, so reviewers would call it incremental.

**With the mechanism framing (§2) and the go/no-go experiments passing:** a plausible
submission to a mid-tier IR journal, e.g. *Information Retrieval Journal* (Springer) or
*Information Processing & Management*. A plausible alternative is a SIGIR/ECIR
short or reproducibility paper. ACM TOIS would need substantially more scale (modern
and larger encoders, real queries, several fact types). This is a judgement, not a
probability.

**What carries the novelty:**
1. the paired measurement plus certified bound: retrieval metrics are invalid for
   freshness evaluation;
2. keep-vectors / update-payload with a cost frontier; this is the most practical and
   least-covered piece;
3. one geometric account unifying retrieval blindness, detector failure and
   mixed-index version noise.

**What could still kill it:**
- **Chunking:** a fact is a larger share of a chunk, so ‖Δ_num‖ grows and certification
  may weaken. Highest risk.
- **Modern encoders:** e5, gte, bge-base/large and API models may be more numerate (see
  NumColBERT); the effect may shrink.
- **"Composition of known facts"** framing by reviewers: mitigate with the bound, the
  policy and the unification.

## 7. Next experiments, in order, with go/no-go criteria

**Correction (SE literature check, 2026-10-10):**
- The ASE paper already scores numeric answers by exact match on its financial datasets.
- It still makes one LLM call per mutant, a "semantic preservation judge", to decide
  whether the mutation changed the meaning. It uses a second judge call for answer
  equivalence on RepLiQA.
- Our oracle gain is therefore narrower than "no judge vs judge". A real diff determines
  both the semantics-altering status and the expected new value, so no judge is needed at
  any step. It also resolves CN's ambiguous expected behaviour.
- No other real-revision metamorphic testing of RAG was found in ASE, ICSE, ISSTA or
  arXiv. "RAG or Learning?" (arXiv 2604.05096) is an evaluation benchmark on time-stamped
  events, not a testing framework.

**In progress (2026-10-12 night), pre-registered before any result:**
- **p11 with qwen2.5:14b.** Scope cut before any 14B output, for runtime (~20 s per
  answer): adequacy plus the no-label stale rate only; the dated variant is not run for
  14B. Disclose this.
- **p12 (RQ2), `pretests/p12_counterfactual.py`.** The ASE counterfactual operator,
  re-implemented on the p11 data: an LLM-written counterfactual passage injected beside the
  current version. Compares P(choose injected) with p11's P(choose old) under real
  updates. A difference of >= 0.10 means CN mis-estimates real updates.
- **p13 (RQ3), `pretests/p13_monitors.py`.** Five monitors on p11's both-in-context cases:
  - M1: retrieval hit@3 across stale, append and replace indexes;
  - M2: a RAGAS-style faithfulness judge (local Qwen-7B);
  - M3: embedding drift, both batch-level sliced-W1/MMD and per-passage displacement;
  - M4: an ASE-style LLM-judge oracle given the diff.

  A monitor "sees" the fault if AUROC >= 0.75, or TPR >= 0.5 at FPR <= 0.10.

**0h. p11, done 2026-10-12: the effect REPLICATES at scale on real updates. Version
labels are not a reliable fix.**
- **Setup:** pre-registration in `pretests/p11_field_passages.py`; outputs in
  `p11_scale.log` and `p11_fresh.log`. This is a redesign after the two void p9 runs, and
  must be disclosed as such: one passage per infobox field, plus the prose chunks.
- **SCALE:** 826 frozen pages (UN members, Nasdaq, NYSE), 659 questions, 400 sampled;
  609 both-in-context cases over 144 pages.

| generator | adequacy | stale, no labels | stale, dated labels | McNemar |
|---|---|---|---|---|
| phi3:mini | 0.98 (743/761) | **0.29** [0.26, 0.33], page-boot [0.25, 0.33] | 0.20 [0.17, 0.23] | p = 2e-6 |
| qwen2.5:7b | 0.80 (607/761) | **0.21** [0.18, 0.24], page-boot [0.18, 0.24] | 0.22 [0.19, 0.26] | p = 0.57 |

- Both generators pass adequacy and replicate.
- **Breakdowns (no labels):** countries 0.25–0.29, companies 0.17–0.29; MiniLM 0.16–0.28,
  bge 0.27–0.30.
- **Dated labels:** help phi3, but not Qwen overall. For Qwen they make countries worse
  (0.25 → 0.32) and companies better (0.17 → 0.13), and "other" answers rise from 82 to 117.
- **Audit:** the sampled stale answers are numbers taken from the old passage.
- **FRESH** (23 real FreshQA questions; descriptive only): 10 both-in-context cases over 6
  pages. Stale with no labels: phi3 4/10, Qwen 2/10. With dates: 0/10 for both.

**0g. p10, 2026-10-11: FreshQA gives 23 usable real-question cases, too few for the main
dataset.**
- **Setup:** `pretests/p10_freshqa_feasibility.py`; output in `p10_freshqa_feasibility.log`.
  FreshQA (Apache-2.0) versions of Feb 2024 and Apr 2026, matched on question text; the
  `id` column shifts between versions. That was a bug in the first count, fixed.
- **Funnel:** 549 shared questions → 267 slow- or fast-changing with true premises → 121
  whose answer changed beyond formatting and has a Wikipedia source → 117 pages fetched →
  **23 usable** (old answer in the 2024 page, new answer in the current page and absent from
  the 2024 one, exact phrase match).
- **Use:** a small natural-question validation set. It is not the scale-up.

**0f. p9, 2026-10-11: the scale-up on Wikipedia infobox updates is VOID, twice. Neither run
counts as evidence for or against.**
- **Data:** 402 pages, frozen (UN members plus a Nasdaq sample); 435 natural questions such
  as "The gdp ppp of Austria is ___."; snapshots at 2025-10-01 and 2026-10-11.
- **Run 1** (`p9_run_primary_VOID_yearbug.log`): void. As-of years such as "(2025)" were
  counted as answer values, which faked "answer in context" and produced year-only
  questions. The adequacy rule was also missing from p9's pre-registration. It was copied
  from p7 before run 2.
- **Run 2** (`p9_run_primary_VOID_adequacy.log`): void.
  - Qwen adequacy 42/83 = 0.51, below 0.60; phi3 0.71.
  - Only 13 both-in-context cases: stale (no labels) 7/13 and 4/13; with dates 4/13 and 1/13.
  - Root cause: retrieval coverage. About 15 chunks per page with the infobox lumped
    together; the gold page reaches the top 3 for about 55% of questions, the gold value for
    about 14%.
- **Proposed redesign** (to be pre-registered before running; a redesign after a void run,
  to be disclosed): one passage per infobox field, with the same reading rules.

**0e. p8, done 2026-10-11: version labels halve the outdated answers but do not remove
them.**
- **Setup:** pre-registration in `pretests/p8_version_labels.py`; output in
  `p8_version_labels.log`. p7's 161 both-in-context cases, with every passage labelled with
  its document and version (arXiv v1/vN; Python docs 3.8/3.13).
  - Variant A: labels only.
  - Variant B: labels plus "If passages disagree, use the one from the most recent version."

| generator | no labels (p7) | A: labels | B: labels + instruction |
|---|---|---|---|
| phi3:mini | 0.17 | 0.11 [0.07, 0.16] | 0.11 [0.07, 0.16] |
| qwen2.5:7b | 0.17 | 0.08 [0.05, 0.13] | 0.07 [0.04, 0.12] |

- **Paired McNemar vs p7:** p = 0.002–0.04 (B for phi3 on clean cases: p = 0.06).
- **Contamination check:** 20 Python-docs cases have an answer that is itself a version
  number in a label. Excluding them gives phi3 0.11–0.12 and Qwen 0.06; same conclusion.
- **Reading (pre-registered):** the in-between outcome, a partial fix. Labels cut outdated
  answers by about 40–60%. The explicit instruction adds nothing beyond the labels.
- **Audit:** the remaining stale answers are genuine (e.g. 52.34 for 50.84; 147 for 1407).
- **Open:** dates instead of version strings. "3.8" vs "3.13" may be misread as decimals.

**0d. p7, done 2026-10-10: the generator does state the outdated value when both versions
are indexed. The decisive pre-test for the ASE-paper improvement: PASS.**
- **Setup:** pre-registration in `pretests/p7_generator_conflict.py`; output in
  `p7_generator_conflict_phi3.log`.
- **Generator:** phi3:mini (3.8B) via local Ollama, temperature 0, with the project's
  control-validated prompt. Qwen2.5-7B was intended, but its download stalled.
- **Adequacy:** replace, k = 3, new value in context: 135/192 = 0.70 [0.64, 0.76], ok.
- **Primary:** append, k = 3, both values in context: 28/161 = **0.17** [0.12, 0.24].
  The kill threshold was 0.10.
  - Per retriever: MiniLM 0.19 [0.12, 0.29], bge 0.16 [0.10, 0.26].
  - Per unique question (n = 89): 0.15 stale under both retrievers.
- **Generator-level at k = 1 (append):** stale 0.28, 0.32, 0.22, 0.28, comparable to the ASE
  paper's worst operator (27.6%, GPT-5-mini, LLM judge). Replace-in-place gives 0.00–0.03.
- **Audit:** the sampled stale answers are genuine (e.g. 2.1 for 1.6, 3.6% for 3.17%).
  "Other" (36%) is mostly phi3 writing stray small numbers on Python-docs questions, which
  is conservative for the stale rate.
- **Confirmed with Qwen2.5-7B-Instruct** (same pre-registration; log in
  `p7_generator_conflict_qwen7b.log`):
  - Adequacy: 149/192 = 0.78 [0.71, 0.83].
  - **Primary: 27/161 = 0.17 [0.12, 0.23]. PASS.**
  - By retriever: MiniLM 0.21, bge 0.13. By corpus: arXiv 0.20 [0.13, 0.30], Python docs
    0.12 [0.07, 0.22]; the subset CIs reach below 0.10.
  - The two generators fail on different cases (stale under both: 13 of 161), but at the
    same rate.
- **Caveats:**
  - two small generators (3.8B, 7B); an API model would be a third check;
  - noisy Python-docs questions;
  - no version metadata in context (the ASE paper's setting too).

**0c. p6, done 2026-10-10 (verified against the paper's text the same day): a candidate
improvement to "When Knowledge Changes: Metamorphic Testing of RAG Systems with Mutations"
(Kim, Pasini, Tonella; ASE 2026; arXiv 2607.26843).**

What the paper does:
- Default top-k = 1, generator GPT-5-mini, embeddings text-embedding-3-small, a single RAG
  architecture.
- Its CN operator "adds one counterfactual passage per document", generated by an LLM and
  meant to simulate "stale data ... v1 vs. v2 manuals".
- The expected behaviour under CN is not consistent within the paper: the metamorphic
  relation says follow the injected fact, the fault taxonomy says signal or resolve the
  conflict.
- Worst violation rate is CN at 27.6% (VQAonBD, pre-chunk).

Our pre-test:
- **Setup:** real revisions with both versions indexed (`append`), k = 1. We count how often
  the one retrieved chunk is an *old-version* chunk carrying only the replaced value. This
  is a deterministic, context-level lower bound, with no LLM.
- **Result** (`pretests/p6_metamorphic_k1.log`): 18/50, 21/50, 15/58, 20/58
  = 0.36, 0.42, 0.26, 0.35.
- **Comparison:** replace-in-place gives exactly 0 everywhere. At k = 3, stale-only falls to
  0.06–0.10 and both values are in context 48–88% of the time.
- **Verdict:** passes the pre-registered kill criterion in 3 of 4 settings.
- **Not like-for-like:** their 27.6% is a generator-level, LLM-judged rate over all CN
  mutants. The decisive next test needs a generator at k = 3.

**0b. p5, done 2026-10-10: "keep vectors, refresh text" is a no-go once documents are chunked.**
The pre-registration is in `research/staleness/pretests/p5_chunked.py`, and the output is
in `p5_chunked_128.log`.

Setup:
- 128-token chunks with content-defined boundaries, which is fair to diff re-embedding.
- An edited chunk keeps its old vector when its closest old chunk has Jaccard ≥ θ.
- Kill criteria:
  - K1: answer@10 ≥ full − 0.02;
  - K2: new and retained hit@10 ≥ full − 0.01;
  - K3: embeds ≤ 70% of diff.

**No θ passes all three in any of the 4 settings.** θ = 0.7 matches a full re-embed but
saves only 19–29% over diff (K3 fails). θ = 0.3–0.5 saves 35–67%, but loses up to
0.09 answer@10 and up to 0.03 new-text hit@10. The trade-off is roughly linear, so there
is no free lunch beyond lossless diff re-embedding. The per-document pre-test (p1) missed
this, because a whole document dilutes the edit and a chunk does not.

Side results, exploratory:
- **Chunked answer@10:** a full re-embed reaches 0.89–0.96, against 0.01–0.18 for the
  stale index.
- **Append (both versions indexed):** 0.88–0.94, within 0.015 of a full re-embed. So the
  correct chunk is retrieved almost as often when the stale one is also present. What the
  generator does with both in context is not measured.

**0. E0, done 2026-10-09: harm-aware index maintenance (G6) is a no-go.** Pre-registered
in `research/staleness/pretests/e0_fact_rates.py` before any data was fetched.
- **Data:** 343 of the 344 `wp_corpus.json` pages (one fetch failed). Each page's lead
  section plus infobox, as it stood every Monday from 2024-09-30 to 2026-09-28: 104 weekly
  intervals, the view a crawler would get.
- **Page change:** the lead text changed. This is the strongest baseline available.
- **Fact change:** the paper's `aligned` edit, or a numeric infobox value change.
- **Event counts:** 1,288 page-change intervals and 126 fact-change intervals (50 prose,
  79 infobox). 78% of pages changed at least once, and 14% had a fact change.

| criterion | result | verdict |
|---|---|---|
| K1: Spearman ρ(fact rate, page rate) ≥ 0.8 kills | ρ = 0.41 [0.30, 0.51] (268 changing pages) | pass: the rates do diverge |
| K2: year-1 fact ranking must catch more year-2 fact changes at a 20% refresh budget than page ranking, CI excluding 0 | 0.729 vs 0.693, difference +0.036 [−0.012, +0.108], 48 year-2 events | **fail** |
| control: on year-2 page changes, fact ranking must not win | −0.009 [−0.030, +0.012] | pass |
| sensitivity: infobox-only facts | K1 ρ = 0.28; K2 +0.066 [+0.000, +0.187] | K2 fails |

**What this means:**
- Fact changes are rare: 0.4% of page-weeks.
- They are no more predictable from fact history than from page history. The
  year-to-year Spearman is 0.37 for fact rates and 0.44 for page rates.
- So the rates diverge, but a scheduler cannot exploit that divergence on this sample.

**Limits:**
- 48 year-2 events give a wide interval, so a modest advantage (up to about +0.1) is not
  excluded.
- The sample leans academic.
- Not a reason to retest until G6 passes. Following the plan, the direction falls back
  to **G1**: an evaluation paper on which monitors can and cannot see staleness, starting
  with the chunked replication below.

1. **Chunked replication** (decides the paper). Chunk both versions (e.g. 128/256 tokens,
   aligned by sentence), rerun answer-level, certified invariance and keep-vectors.
   **Go** if the retrieval gap on changed facts stays within about ±0.02 and keep-vectors
   stays within 0.02 of fresh on answer@10. **No-go** if chunking makes retrieval clearly
   see fact edits; then the paper reverts to a short evaluation paper.
2. **bge at its native 512-token window:** recompute the embeddings, then rerun §4.
3. **Third and fourth encoders,** one larger (768-d+, e.g. bge-base, e5-base, gte-base). Test
   whether certification and the d_eff model hold; report if numeracy improves.
4. **Detection bound vs null floor:** compute (1/N)Σ‖Δ_num‖ against the null distribution
   of the sliced-W₁ / MMD statistics from `quarantine/FINDINGS.md` §12.9 / §13.3 at matched n.
5. **Mixed-index generation:** both versions indexed, top-3 to Qwen2.5-1.5B (and a larger
   model). What does the model state when given conflicting values?
6. **Entity and date edits** on Wikipedia (`wp_corpus.json`): test the token-type
   sensitivity prediction on real edits.
7. **Natural questions:** LLM-written questions for a sample of fact edits, to answer the
   "synthetic sentence windows" criticism.

Package each passing pre-test as `research/staleness/<topic>.py` with tests and a
`results_<topic>.json`, as `answer_level.py` does.

## 8. Rules learned the hard way (apply to every new number)

- **Ask "what would make this fail?"** A number that cannot fail, like the earlier "100% edit
  detection", is circular.
- **Use independent ground truth:** never define truth with the same predicate the
  detector uses.
- **Count denominators once:** cross-listed IDs were double-counted once (93.8% recall).
- **Redraw randomness per repetition:** reusing one traffic draw inflated 1.30× into
  20.8×.
- **Bootstrap over queries, not seeds.** Use Wilson intervals for proportions and Fisher exact
  for 2×2 tables.
- **Compare numbers by value:** `84.3%` = `84.3`, `450K` = `450,000`; keep version
  strings verbatim.
- **Watch truncation:** an edit past the token window has exactly zero effect.
- **Watch construction bias:** fact questions are built from the *revised* sentence, so
  they can favour the fresh document. Check with controls or strict (wording-identical)
  edits.
- **Every comparison needs a control** where no effect is expected.
- **Say what is known:** answer-level harm (HoH, Poisoning), numeric insensitivity
  (Numeracy Gap) and margin bounds (RetrievalGuard) are prior work.
- **Pre-test before proposing.** State the kill criterion before running.

## 9. Environment

- Project venv: `.venv` (Python 3.11). Tests: `DRIFT_MINILM_DIR=~/.cache/minilm-l6-v2
  .venv/bin/python -m pytest` gives 71 passed (CI style, `-m "not slow"` without the
  model: 59 passed, 10 skipped).
- Encoders: `~/.cache/minilm-l6-v2`, `~/.cache/bge-small-en-v1.5`. Generators:
  `~/.cache/qwen2.5-1.5b-instruct` (and 0.5b, rejected on controls).
- The torch/transformers venv used for generation was in a session scratchpad and is
  **gone**. Recreate it:
  `python3 -m venv ~/venvs/torchgen && ~/venvs/torchgen/bin/pip install torch transformers`.
  Then run `~/venvs/torchgen/bin/python research/staleness/generate_hf.py ~/.cache/qwen2.5-1.5b-instruct`.
- Embedding cache: `embeddings/cache/staleness/<model>/` (gitignored, keyed by model).
- Don't commit or push without the user asking. Commits end with the
  `Co-Authored-By` line from the system attribution reminder.
