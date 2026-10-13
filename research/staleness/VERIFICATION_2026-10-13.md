# Verification of RESEARCH_HANDOFF §0, 2026-10-13 (late evening)

> **Superseded on validity: read `CRITICAL_REVIEW_2026-10-13.md`.** This report checks that
> numbers *reproduce* from the records. It does not check that they measure what §0 claims.
> In particular, the p12 "MATCH" below is a reproduction match only: the registered RQ2
> contrast mis-maps ASE's oracle, and under that oracle the conclusion reverses.

Every check below was run in this session. Scripts and their raw outputs are in
`research/staleness/verification_2026-10-13/`. Run them from that folder with the repo `.venv`;
`v_ctx.py` writes `ctx.json` and `v_p12.py` writes `cfs.json`, which later scripts read.

**Independence.** The scripts import only the *data-construction* code: p9 `build`/`fields`,
p5 `chunks`, e0 `prose`, plus `data.digest` to locate cached embeddings. Without these, the
400-question sample and the passages cannot be rebuilt. Everything else is written from
scratch: scoring, top-3 retrieval, Wilson intervals, page bootstrap, exact McNemar, AUROC,
sliced-W1, MMD and the permutation tests. `P9.commit` and `generation.classify` are called
only to *compare* against my own scorer.

**Verdict.**
- Every number in §0.3 reproduces to 2 decimals from the raw records.
- The data is real.
- The controls pass.
- Five things need correcting or disclosing (§9). None reverses a headline result, but two
  weaken RQ3 claims, and the literature table has two wrong statements about
  Stale-Document Poisoning.

---

## 1. State of the machine

| Check | Result |
|---|---|
| `pgrep -fl "p1[1-9]_"` | nothing running |
| git | branch `staleness-answer-level`, up to date with origin, HEAD `d3ddd5d` |
| Tracked files shown as deleted | 133. All 133 basenames are present under `quarantine/` (`verification_2026-10-13/deleted.txt`; basename match only) |
| Untracked | `RESEARCH_HANDOFF.md`, `research/staleness/pretests/`, `tests/test_transfer_power.py` |
| Tests | `DRIFT_MINILM_DIR=~/.cache/minilm-l6-v2 .venv/bin/python -m pytest`: **71 passed**, 0 failed, 0 skipped |
| Ollama models | `qwen2.5:14b-instruct 7cdf5a0187d5`, `qwen2.5:7b-instruct 845dbda0ea48`, `phi3:mini 4f2222927938` (record these IDs in the paper) |

## 2. Every §0.3 number, recomputed

Sources:
- records: `embeddings/cache/staleness/{p11/records_scale.jsonl, p11/records_fresh.jsonl, p12/records.jsonl, p13_cases.jsonl, p13_results.json}`;
- logs: `research/staleness/pretests/*.log`.

My page-bootstrap uses a different RNG and 4,000 resamples, so its CI ends can differ from the
log by ±0.01.

**Sample.** Rebuilt sample: 826 pages, 659 questions, 400 sampled (hash `34e0c441d8e88f97`).
All 5,275 p11 records have `qtext` equal to the rebuilt statement and `page` equal to its
doc (0 mismatches). The 605 append and 750 replace case sets are identical across the three
generators. My own top-3 retrieval selects exactly the same cases (minilm 322 + bge 283
append; 387 + 363 replace).

| Claim | §0.3 | Recomputed | Source | Log | |
|---|---|---|---|---|---|
| p11 rerun cases / pages | 605 / 142 | 605 / 142 | records_scale | p11_scale_14b.log L8 | MATCH |
| phi3 stale (pre-reg classifier) | 0.31 [0.27–0.35] | 186/605 = 0.307, Wilson 0.272–0.345, page-boot 0.264–0.349 | records_scale | L8 | MATCH |
| Qwen-7B stale | 0.19 [0.16–0.23] | 117/605 = 0.193, Wilson 0.164–0.227 | records_scale | L11 | MATCH |
| Qwen-14B stale | 0.03 [0.02–0.05] | 19/605 = 0.031, Wilson 0.020–0.049 | records_scale | L14 | MATCH |
| Adequacy phi3 / 7B / 14B | 0.98 / (n.s.) / 0.91 | 732/750 = 0.976; 593/750 = 0.791; 685/750 = 0.913 | records_scale | L7, L10, L13 | MATCH (all ≥ 0.60) |
| phi3 dated | 0.31 → 0.20, p = 1e-7 | 121/605 = 0.200; fixed 107, broke 42; exact p = 1.0e-7 | records_scale | L9 | MATCH |
| Qwen-7B dated | 0.19 → 0.22, p = 0.28 | 131/605 = 0.217; 64 / 78; p = 0.28 | records_scale | L12 | MATCH |
| p11 run 1 | 609 cases, 144 pages; 0.29 (ad. 0.98), 0.21 (ad. 0.80) | 609 / 144; 0.291 (0.976); 0.207 (0.798) | records_scale_run1 | p11_scale.log L4–8 | MATCH |
| Diagnostic phi3 new / old / both / neither | 0.45 / 0.31 / 0.20 / 0.04 | 0.455 / 0.307 / 0.195 / 0.043 | records_scale | none (post hoc) | MATCH |
| Diagnostic Qwen-7B | 0.24 / 0.20 / 0.43 / 0.13 | 0.243 / 0.197 / 0.426 / 0.134 | records_scale | none | MATCH |
| Diagnostic Qwen-14B | 0.10 / 0.04 / 0.84 / 0.02 | 0.098 / 0.041 / 0.840 / 0.021 | records_scale | none | MATCH, scorer-dependent (§3) |
| Replace control old-only | 0.00 all | 0/750 for each generator | records_scale | none | MATCH |
| p12 phi3 CN injected-only | 0.45 [0.41–0.50] | 321/708 = 0.453; Wilson 0.417–0.490; page-boot 0.405–0.503 | p12/records | p12 log L5 | MATCH (the log's CI is page-boot) |
| p12 phi3 diff | +0.15 | +0.146 | p12 + p11 records | L5 | MATCH |
| p12 Qwen-7B | 0.24 vs 0.20, +0.04 | 0.236 vs 0.197, +0.039 | same | L6 | MATCH |
| p12 "both" | CN 0.16 / 0.40; real 0.20 / 0.43 | 0.160 / 0.398; 0.195 / 0.426 | same | L5–6 | MATCH |
| p12 n | 708 vs 605 | 708 per generator (376 minilm + 332 bge) vs 605 | same | L2–3 | MATCH |
| p12 fresh counterfactuals | 400/400 (fallback not logged) | 400/400; **fallback used 8/400** | qwen2.5_7b cache | L1 | MATCH; fallback count now known |
| p13 M1 hit@3 | 1.000, 3 indexes × 2 retrievers | 1.000 for all 6 (own retrieval) | rebuilt | p13 log L1–3, 8–10 | MATCH |
| p13 "both" excluded | 376 | 228 + 148 | records_scale | L4, L11 | MATCH |
| p13 M3a p | W1 0.70 / 0.105; MMD 0.12 / 0.09 (minilm / bge) | as stated in p13_results.json; **independent rerun differs for bge** (§9.3) | p13_results.json | L6–7, 13–14 | MATCH to file; claim fragile |
| p13 M3b AUROC | 0.465 / 0.502 | 0.465 / 0.502 **per generator** (phi3 / Qwen-7B), pooled over retrievers | p13_cases | L18, L22 | MATCH; §0.3 label ambiguous |
| p13 M2 TPR / FPR | 0.83 / 0.88; 0.85 / 0.84 | 155/186 = 0.833, 105/119 = 0.882; 234/275 = 0.851, 123/147 = 0.837 | p13_cases | L16, L20 | MATCH |
| p13 M4 TPR / FPR | 1.00 / 0.99; 0.02 / 0.01 | 186/186, 118/119 = 0.992; 5/275 = 0.018, 2/147 = 0.014 | p13_cases | L17, L21 | MATCH |
| p13 M4 agreement | 0.989 | 0.9892 / 0.9887 | p13_cases | L19, L23 | MATCH |
| p13 labels | `P9.commit` | 727/727 stored labels equal `P9.commit`; 727/727 outputs equal the p11 record | p13_cases | none | MATCH |
| FreshQA | 23 q, 10 cases, 6 pages; 4/10, 2/10, 0/10; 14B correct 7/10; dated 0/10 | same | records_fresh | p11_fresh_14b.log L1, 8–14 | MATCH |
| p6 | old-only chunk 26–42%; replace 0% | k=1 append: 0.36 / 0.42 (arXiv), 0.259 / 0.345 (Python docs); replace 0.000 | log only | p6_metamorphic_k1.log | MATCH (log only; no records) |
| p7 | 161 cases; stale 17% both; adequacy 0.70 / 0.78 | 28/161, 27/161; 135/192 = 0.70, 149/192 = 0.78 | log only | p7_*.log tails | MATCH (log only) |
| p8 | labels cut stale to 7–11% | 0.11, 0.11, 0.08, 0.07 | log only | p8_version_labels.log L2–6 | MATCH (log only) |

## 3. Scorer check and hand reading (90 records)

**My scorer against `P9.commit`.** The two agree on all but 109 of 5,275 records.
- **107 are the 14B, all in append.** Each answer is two values joined by a comma with no space,
  e.g. `'519,542'` (old 519, new 542) or `'357,022,357,114'`. My scorer read them as one
  thousands-grouped number. In every case the two parts are exactly the old and new values,
  so `P9.commit`'s "both" is right and my scorer was wrong.
- **Consequence.** The 14B "both" rate (0.84) depends on reading comma-joined output as a list.
  Under a strict thousands reading it would be 0.66. The prompt itself asks for "number(s),
  separated by commas", so this must be disclosed.
- **The other 2 are unit edge cases:** `'13 billion'` vs 13.0, and `'1,040,748'` (new 1.04 billion,
  old 748 million).

**Hand reading** (`verification_2026-10-13/v_hand.out`; 30 random append/none records per
generator, seed 20261013, each shown with its 3 context passages):

| Generator | My label ≠ `P9.commit` | Cases |
|---|---|---|
| phi3 | 0/30 | none |
| Qwen-7B | 1/30 | [10] `'788 million, 2,860 million'`, new 2.86 billion: really "both", scored "old" |
| Qwen-14B | 2/30 | [17] `'-665,-1140unknown'` (new −1.14 billion): "both", scored "old"; [29] `'3951,1410'` (millions): "both", scored "neither" |

Overall 3/90 = **3.3% disagreement**. Every error is a **unit conversion** (million ↔ billion)
that string matching cannot see, and every one hides a "both". The 14B "both" rate is
therefore, if anything, slightly *under*-stated. The four M4/oracle disagreements in p13 are
the same kind (`'1,100,-473'`, `'87.981,105.907'`).

**Truncation (`num_predict` = 24).**
- 62 of phi3's 605 append answers are ≥ 60 characters. Some are cut off mid-list, e.g.
  `'The net income of Blackstone Inc. is US$2.44 billion (2024) and US'`, scored "old" although
  the model was listing both.
- At most 10 phi3 old-only labels are long answers, so phi3's 0.31 could be overstated by
  ≤ 0.017.
- Qwen-7B has no answers ≥ 60 characters; the 14B has 2.

## 4. The data is real (live MediaWiki API, `v_live.py`, `v_live.out`, `v_live2.out`)

Five random pages that carry sampled questions (seed 5): Saint Vincent and the Grenadines,
Mexico, Kuwait, Sabre Corporation and Croatia.
- **Revisions.** Each stored `ts0`/`ts1` resolves to a live revision with exactly that
  timestamp (revids 1313179825/1378211546, 1314345991/1378233112, 1314044020/1379481431,
  1310864038/1379424313, 1313807473/1378475306).
- **Text.** The live section-0 wikitext is **byte-identical** to the stored `text0`/`text1` in
  10/10 revisions.
- **Questions.** All 13 sampled questions on these pages pass: the old value is in the old
  revision, the new value is in the new one, the field changed, and the new value is absent
  from the old field. Example: Mexico `gdp_nominal` "$1.693 trillion" → "$2.121 trillion".
- **Cosmetic defects.** `&nbsp;` survives rendering (e.g. "3.71&nbsp;billion"). Sabre's old net
  income `{{Color|red|(0.528)}}` (a loss) renders as "red (0.528)", so the minus sign is lost.

## 5. Literature, checked against the sources

arXiv HTML and abstract pages were fetched with curl and grepped. The extracted text is in
`verification_2026-10-13/lit/`. Quotes are verbatim.

**Kim, Pasini & Tonella, "When Knowledge Changes: Metamorphic Testing of RAG Systems with
Mutations", arXiv 2607.26843.** Authors from the `citation_author` metadata; arXiv Comments:
"ASE 2026".

| §0 statement | Verdict | Quote |
|---|---|---|
| default top-k = 1 | SUPPORTED | "In the study, the default top-k is set to 1." TK operator: "TK increases k from 1 to 3." |
| GPT-5-mini | SUPPORTED | "Default RAG generation, counterfactual generation, closed-book filtering, semantic-preservation judging, and answer-equivalence judging use OpenAI's GPT-5-mini (gpt-5-mini-2025-08-07)." |
| text-embedding-3-small | SUPPORTED | "We use text-embedding-3-small as the embedding model." |
| CN peaks at 27.6% (VQAonBD, pre-chunk) | SUPPORTED | "CN consistently produces the highest violation rates across all datasets and both scopes, reaching 27.6% (pre) and 27.0% (post) on VQAonBD" |
| exact numeric match on financial datasets | SUPPORTED | "The four financial datasets use exact numeric comparison for answer equivalence, reducing their reliance on the judge." |
| LLM judge checks semantic preservation | SUPPORTED | "(1) a semantic preservation judge that verifies, for each mutant, whether the factual evidence required to answer the query remains intact after the perturbation" (an answer-equivalence judge is also used on RepLiQA) |
| CN = LLM-written counterfactual; answer should follow it | SUPPORTED | "the operator prompts a separate LLM to generate a plausible counterfactual statement given the original fact with the user query." "We expect y′≠y, specifically where y′ aligns with the injected fact." |
| CN framed as stale data (v1 vs v2) | SUPPORTED | Knowledge Conflict Fault (which CN targets): "This reflects the common real-world challenge of stale data, where a knowledge base contains both outdated legacy documents and recent updates (e.g., v1 vs. v2 manuals)." Also CN: "the retriever may return the contradiction without the original fact, simulating stale data scenarios." |
| expected behaviour is ambiguous | SUPPORTED (our reading of two sentences) | Fault definition: "A robust system must exhibit explicit conflict awareness, signalling or resolving the discrepancy." Variance relation: "y′ aligns with the injected fact." **By their own fault definition, an answer that lists both values ("signalling") may count as robust.** This matters for how the 14B's "both" is framed. |

**Stale-Document Poisoning**: Ahmed, Lukas Galke Poech and Röttger, arXiv 2609.31342,
2026-09-25.
- **SUPPORTED** (abstract): "317 verified knowledge reversals"; "Across 12 models"; "poisoning
  ranges from 17-91%"; "a fixed recency-aware hybrid re-ranker reduces poisoning by
  4.6-10.0 points".
- **"no re-embedding": SUPPORTED.** Zero occurrences of re-embed or embedding-maintenance
  topics.
- **"no geometry": CONTRADICTED.** "Appendix M Geometry Decomposition: The poison-induced
  decision-token shift is decomposed into a radial component along an unaided-correctness
  direction and an orthogonal remainder." This is internal-state geometry, not retrieval-
  embedding geometry. The handoff must say "no *embedding-space* geometry".
- **"No retrieval-vs-answer metric comparison": OVERSTATED.** Appendix N reports
  up-to-date-document selection (retrieval) and downstream poisoning side by side: "The
  re-ranker selects the up-to-date document more often than dense retrieval in 8 of 12
  non-clean cells". Note also that "Both dense systems use sentence-transformers/all-MiniLM-L6-v2",
  the same retriever we use.
- Their mixed index has "one up-to-date and two outdated documents per item", but "Plain dense
  retrieval returns the top semantic match". Their generator therefore sees one document,
  not both versions together as in our k=3 setting.

**HoH**: Ouyang, Pan, Cheng, Yan, Luo, Lin and Liu, arXiv 2503.04800.
- Venue: the ACL Anthology page `2025.acl-long.301` was opened, and its title matches. ACL 2025
  is **SUPPORTED**.
- The claim "outdated KB entries reduce RAG answer accuracy" is **SUPPORTED**: "outdated
  information significantly degrades RAG performance in two critical ways: (1) it
  substantially reduces response accuracy by distracting models from correct information".
  Its scope is "the impact of outdated information coexisting in retrieval sources", which is
  the same setting as our RQ1 and confirms that RQ1's harm is not novel.

**FreshQA / FreshLLMs**: Vu et al., arXiv 2310.03214 (title and authors from metadata).
- **SUPPORTED** as a dynamic QA benchmark: "we introduce FreshQA, a novel dynamic QA benchmark".
- The licence claim (Apache-2.0) is **SUPPORTED**: the `LICENSE` at
  github.com/freshllms/freshqa starts "Apache License Version 2.0".

**Podkopaev & Ramdas**, "Tracking the risk of a deployed model and detecting harmful
distribution shifts", arXiv 2110.06177.
- Comments: "Accepted as a conference paper at ICLR 2022". **SUPPORTED** as cited in memory
  ("harmful vs benign drift").
- I read only the abstract and metadata, not the body.

**LegalSearch-R1**: arXiv 2605.25920, "Can LLMs Time Travel? Enhancing Temporal Consistency in
Legal Agentic Search through Reinforcement Learning"; Comments: "Accepted by EMNLP NLLP 2026".
- The memory note is **SUPPORTED**: "search agents rarely incorporate temporal constraints into
  queries"; "surpasses baselines by 57.7% to 80.3% on temporal consistency"; "trained on
  temporally-indexed data spanning multiple amendment periods".

**Revoked but Still Authoritative**: Shen, Toyoda and Leung, arXiv 2609.08258.
- **SUPPORTED, with a qualifier**: "we measure five such systems"; "no system enforces
  revocation by default: the revoked fact is returned wherever the revocation label is
  visible to the retrieval layer, outranks its replacement".
- "Outranks" holds *where the revocation label is visible to retrieval*, so cite it with that
  condition.

**Not checked:** the "team doc". I found no team document on disk, and the Google Drive
connector is not authorised in this session.

## 6. Pre-registration order

Times are file mtimes and the chain log. Note: on macOS the birth time equals the last write
time for these files, because editors rewrite them.

| Event | Time (2026-10-13 unless noted) |
|---|---|
| p11_wiki_frozen created | 10-11 23:41 (after p11's "pre-registered 2026-10-11" date) |
| p11 run 1 records / log | 10-12 20:57 |
| 14B run stopped (sample-order bug) | 14:39 (log holds only the header line) |
| `p11_field_passages.py` last written | 15:51:50 |
| Chain start | 15:58:23 |
| p11 rerun results printed (incl. 14B) | 16:52:29 |
| **`p9_wiki_infobox.py` (holds `P9.commit`) last written** | **16:54:33** |
| `p12_counterfactual.py` (amendment) last written | 16:55:04 |
| p12 result | 17:44:09 |
| p13 first run (already labels with `P9.commit`: "excluded: 228") crashes | 18:01:40 |
| `p13_monitors.py` last written (crash fix) | 18:01:58 |
| p13 result | 18:15:35 |

**Findings.**
- **p12 and p13.** Both amendments precede their own first results:
  - p12: script 16:55, result 17:44.
  - p13: the crash run already used `P9.commit`.
- **But the p12 amendment was not blind.** `P9.commit`'s file and the p12 amendment were last
  written *after* the p11 rerun printed its results, including the 14B result. The amendment
  was therefore written after the p11 rerun outputs were available. Disclose it as "amended
  before any p12 result, after the p11 rerun results".
- **Orphan 14B answers.** The 14B generation cache holds 1,528 answers. 1,274 match the final
  run's scale prompts. Of the other 254, about 32 are FreshQA prompts, and **about 220 are not
  explained by the final run**, most likely the stopped 14:39 run on the unstable sample. Its
  log shows that no result was printed. Whether those outputs were read cannot be verified.
- **Limits.**
  - The pretests and the handoff are **uncommitted**, so there is no git timestamp.
  - Docstring dates are self-reported.
  - An mtime shows only the *last* write, so earlier edits are invisible.
  - Nothing here proves a docstring was written before data were seen. Only the ordering of
    last writes against result files is verifiable.

**Recommendation for p14 onwards.** Before each run:
1. commit the script, or at least record `shasum -a 256 script.py` into the chain log with a
   timestamp, and write the hash into the result log's first line;
2. push the commit, or e-mail the hash to Suvidh and Aryan, so a third party holds the
   timestamp;
3. write the analysis code (thresholds and scorer) into the same committed file, never into a
   module edited later.

## 7. Controls and leakage

| Check | Result |
|---|---|
| Replace old-only = 0.00 | 0/750 for phi3, Qwen-7B and Qwen-14B: **PASS** |
| Adequacy ≥ 0.60 | 0.976 / 0.791 / 0.913: **PASS** |
| Question text contains no answer | 0/400 sampled questions contain an old or new value: **PASS** |
| Years excluded | 0 raw 19xx/20xx tokens in the old/new values. Three comma-formatted counts look like years ("2,000", "1,900", "1,934" employees); they are real counts: **PASS** |
| Both values in context | 605/605 append cases contain both values literally in the top-3 text, every one via the field passage. Replace: 750/750 contain the new value: **PASS** |
| new ∩ old values | 0: **PASS** |
| p12 ×1.13 fallback | **8/400.** Some are implausible: "gdp ppp per capita rank: 119.78th", "116.39rd", population density "126.56". One (Oman) still contains the current value 315,331 |
| p12 years in injected values | **131/400** counterfactual value sets include a year, e.g. "(2025)" (see §9.4) |

## 8. What this session could not verify

1. Docstring pre-registration dates (self-reported; see §6).
2. Whether the ~220 orphan 14B outputs from the stopped run were looked at.
3. p6, p7 and p8: checked against logs only. Their records were not in the requested list and
   were not recomputed.
4. The M3a p-values in `p13_results.json` exactly. My own permutation tests use different
   random projections and subsamples; results in §9.3.
5. The "team doc" cited in the task: not found on disk; Google Drive not authorised.
6. Podkopaev & Ramdas, LegalSearch-R1 and Revoked: abstract and metadata only, not the full
   text.
7. Whether Ollama model digests changed between runs. Only today's IDs were recorded.

## 9. Problems found (not fixed; for the user's decision)

1. **14B "both" depends on the parser.**
   - 107 of its 508 "both" answers are comma-joined with no space ("519,542").
   - Hand reading supports `P9.commit`, but the number must be reported with this caveat.
   - p14 (single-answer prompt) removes the ambiguity.
2. **Unit conversions are invisible to the scorer.** 3/90 in the hand check, plus the M4
   disagreements; they bias toward too few "both". Disclose; consider the human check
   (step 13) as the remedy.
3. **The M3a "no significant shift" claim is fragile for bge.**
   - Independent reruns on three random 1,000-subsamples: bge sliced-W1 p = 0.014 / 0.088 / 0.208,
     MMD p = 0.064 / 0.088 / 0.090.
   - minilm stays null (W1 0.66–0.78, MMD 0.39–0.67).
   - Suggested wording: "the batch-level shift is small and borderline for bge (p 0.01–0.21
     depending on subsample); a detected shift would in any case not identify which answers
     are stale".
4. **p12 year leak.**
   - 11 included CN cases (2 minilm, 9 bge) had the counterfactual passage *outside* the
     top-3; they qualified only because a year matched.
   - 11 answers, all phi3, were labelled "both" only because they echoed "(2025)"; they are
     really "current". This lowers phi3's CN "both" rate, not injected-only.
   - Sensitivity (post hoc): excluding those cases gives phi3 0.461, Qwen-7B 0.240; the gaps
     stay +0.15 and +0.04, so the **conclusion is unchanged**. The code should exclude years
     from counterfactual values as p9 does.
5. **M2 is a weak baseline.**
   - The faithfulness judge says "not supported" for about 84% of answers, right or wrong
     (FPR 0.85 / 0.84).
   - Reviewers may call it a strawman. Either strengthen it (RAGAS proper, or a better judge
     prompt) or report it as "a local judge with this prompt".
6. **§0.3 wording fixes.**
   - "M3b AUROC 0.465 / 0.502" is per generator, not per retriever. Per retriever: phi3
     0.509 / 0.496; Qwen-7B 0.454 / 0.455.
   - §0.4 item 4: the crashing call passed *one* argument; the fix passes `(X, Y)`.
7. **§5 literature table.** Stale-Document Poisoning: "no geometry" → "no embedding-space
   geometry"; "no retrieval-vs-answer metric comparison" → "reports up-to-date-document
   selection alongside poisoning for a re-ranker, at top-1, with MiniLM-L6-v2; no
   both-versions-in-context generator study".
8. **Pre-registration disclosure** (§6): the scorer amendment came after the p11 rerun
   results; about 220 orphan 14B outputs exist.
9. **Truncation at 24 tokens** can cut phi3's lists (≤ 0.017 effect on phi3 old-only).
10. **Cosmetic:** `&nbsp;` left in passages; a negative sign is lost in the "red (x)" template.
