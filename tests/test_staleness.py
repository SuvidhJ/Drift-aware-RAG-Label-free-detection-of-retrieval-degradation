"""The definitions behind the staleness-cost and answer-level results.

These pin the parts that, if they drifted, would change the paper's numbers without
raising an error: what counts as a fact edit, what the question is allowed to contain,
which answers are too short to trust, and how gold ranks are computed.
"""
from __future__ import annotations

import numpy as np
import pytest

from research.staleness import facts, measure, queries
from research.staleness.answer_level import control_questions

S1 = "Our method reaches an accuracy of 84.3 percent on the benchmark suite we study."
S1_REV = "Our method reaches an accuracy of 86.1 percent on the benchmark suite we study."
S1_REWORD = "Our new method reaches a top accuracy of 86.1 percent on the benchmark suite we study."
S2 = "The remaining experiments use exactly the same training protocol as before throughout."
S3 = "We release all the code and the trained checkpoints publicly for the community to use."


def _doc(t0: str, t1: str) -> dict:
    return {"text0": t0, "text1": t1}


# ------------------------------------------------------------------- sentences
def test_short_fragments_are_not_sentences():
    assert queries.sentences("Short one. " + S2) == [S2]


# ------------------------------------------------------------------ fact edits
def test_strict_edit_is_number_only():
    docs = [_doc(f"{S1} {S2}", f"{S1_REV} {S2}")]
    assert facts.strict_edits(docs) == [(0, S1_REV, S1)]


def test_strict_excludes_reworded_sentence():
    docs = [_doc(f"{S1} {S2}", f"{S1_REWORD} {S2}")]
    assert facts.strict_edits(docs) == []


def test_aligned_admits_reworded_sentence_with_new_number():
    docs = [_doc(f"{S1} {S2}", f"{S1_REWORD} {S2}")]
    assert facts.aligned_edits(docs) == [(0, S1_REWORD, S1)]


def test_aligned_ignores_wording_change_without_number_change():
    t = "Our method reaches an accuracy of 84.3 percent on the suite of benchmarks we study."
    assert facts.aligned_edits([_doc(f"{S1} {S2}", f"{t} {S2}")]) == []


def test_unrevised_document_has_no_fact_edits():
    docs = [_doc(f"{S1} {S2}", f"{S1} {S2}")]
    assert facts.strict_edits(docs) == [] and facts.aligned_edits(docs) == []


# ------------------------------------------------------------- fact questions
def test_question_does_not_contain_its_answer():
    qs, gold, answers = facts.fact_questions([_doc(f"{S1} {S2}", f"{S1_REV} {S2}")])
    assert gold == [0] and answers == [{"86.1"}]
    assert "86.1" not in qs[0] and "84.3" not in qs[0]


def test_short_answers_are_excluded():
    """'2' -> '3' appears in old documents by coincidence, so it is not an answer."""
    t0 = "The model uses 2 attention heads in every block of the network we describe here."
    t1 = "The model uses 3 attention heads in every block of the network we describe here."
    assert facts.fact_questions([_doc(f"{t0} {S2}", f"{t1} {S2}")]) == ([], [], [])


def test_control_takes_numbers_from_retained_sentences_only():
    docs = [_doc(f"{S1} {S2}", f"{S1} {S3}")]          # S1 retained, S2 -> S3
    qs, gold, answers = control_questions(docs)
    assert gold == [0] and answers == [{"84.3"}]


# ----------------------------------------------------------------- query pools
def test_new_and_retained_pools_partition_the_revised_text():
    docs = [_doc(f"{S1} {S2}", f"{S1} {S3}")]
    new, _ = queries.new_text(docs)
    kept, _ = queries.retained_text(docs)
    assert all(S3 in q or q.startswith(S3[:20]) for q in new)
    assert all(S1 in q for q in kept)


def test_pools_are_deterministic():
    docs = [_doc(f"{S1} {S2} {S3}", f"{S1_REV} {S2} {S3}")] * 5
    assert queries.new_text(docs)[0] == queries.new_text(docs)[0]


# ------------------------------------------------------------------ the metric
def test_gold_rank_is_exact():
    D = np.eye(4)
    Q = np.array([[1.0, 0.5, 0, 0], [0.2, 0.9, 0.95, 0]])
    ranks = measure.gold_ranks(Q, D, np.array([0, 1]))
    assert list(ranks) == [1, 2]


def test_metrics_from_ranks():
    r = np.array([1, 2, 11])
    assert list(measure.metric(r, "hit@1")) == [1, 0, 0]
    assert list(measure.metric(r, "hit@10")) == [1, 1, 0]
    assert measure.metric(r, "mrr")[1] == pytest.approx(0.5)


def test_pool_weights_sum_to_one():
    docs = [_doc(f"{S1} {S2}", f"{S1_REV} {S2}"), _doc(S3, S3)]
    assert sum(measure.pool_weights(docs).values()) == pytest.approx(1.0)


def test_formatting_change_is_not_a_fact_edit():
    """'84.3%' -> '84.3' is a dropped percent sign, not a changed fact."""
    t0 = "Our method reaches an accuracy of 84.3% on the benchmark suite we study here today."
    t1 = "Our method reaches an accuracy of 84.3 on the benchmark suite we study here today."
    assert facts.fact_questions([_doc(f"{t0} {S2}", f"{t1} {S2}")]) == ([], [], [])


def test_numbers_compare_by_value():
    assert facts.norm("20,000") == facts.norm("20000")
    assert facts.norm("17.67%") == facts.norm("17.67")
    assert "13.4" not in facts.values("a score of 113.45 overall")


def test_magnitude_suffix_is_part_of_the_value():
    assert facts.norm("450K") == facts.norm("450,000") == "450000"
    assert facts.norm("7B") == "7000000000"
    assert facts.norm("1.20") == facts.norm("1.2")
    assert facts.NUM.findall("weighs 5 kg and uses 5 MB") == ["5", "5"]


def test_spoken_values_match_digits():
    assert facts.spoken_values("about 1.5 million images") == {facts.norm("1.5M")}


def test_version_strings_are_kept_verbatim():
    assert facts.norm("3.8.0") == "3.8.0"
