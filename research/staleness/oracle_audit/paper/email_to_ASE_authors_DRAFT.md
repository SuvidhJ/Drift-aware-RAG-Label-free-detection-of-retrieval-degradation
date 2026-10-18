DRAFT, not sent. To be reviewed and sent by the authors (Aditi, Suvidh, Aryan) before submission.

To: jinhan.kim@usi.ch, samuele.pasini@usi.ch, paolo.tonella@usi.ch
Subject: Question about the counterfactual relation in "When Knowledge Changes" (ASE 2026)

Dear Dr Kim, Dr Pasini and Prof. Tonella,

We have been working with your ASE 2026 paper "When Knowledge Changes: Metamorphic Testing of RAG
Systems with Mutations", using the released artefact (commit 597fd6582b) and the outputs on Figshare.
Thank you for making both available. They allowed us to study the counterfactual-noise (CN) operator
in some detail, and we would like to share what we found before we submit a manuscript that discusses
it. If we have misread anything, we would be grateful to hear it.

Our main question concerns the variance relation. Section 4.5 of the paper says that y' should align
with the injected fact. In the numeric branch of step3_mutation_campaign.py (lines 746 to 748), an
altering mutant is marked SATISFIED whenever the answer differs from the baseline answer, and we could
not find a place where the answer is compared with the injected value. We also noticed that the
RepLiQA judge (line 814) receives the original context rather than the counterfactual.

In the released outputs, 460 of the 461 CN violations in the altering branch are answers that match
the ground truth, which is at least 77% of all 594 CN violations reported in the paper. We also found
that 694 of the 2,671 generated counterfactuals (about 26%) are refusals by the generating model, for
example "Sorry, I can't help create misleading counterfactuals...". Most of these were judged
semantics-preserving.

Finally, for non-preserving mutants, step4_fault_repair.py (line 395) sets
fault_repaired = (gt_correct != "correct"). As we read it, a fault counts as repaired once the answer
departs from the original correct answer. This seems to differ from the example in Section 6.3: the
InspirePay case (RepLiQA, post-chunk, GPT-5.4) is described there as a fault that persisted after
repair, but the released outputs record it with is_repaired = True.

We are happy to send our scripts and a line-by-line trace. If you see any of this differently, or if a
later version of the artefact changes these points, please let us know so that we describe your work
accurately. We plan to submit in about two weeks and would welcome your comments before then.

Kind regards,
Aditi Saxena, Suvidh Jain and Aryan Garg
[affiliation, contact]
