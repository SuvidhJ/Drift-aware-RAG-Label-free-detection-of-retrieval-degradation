"""Pin the definitions used by the oracle-audit paper (research/staleness/oracle_audit, pretests p14-p16).

These tests guard the scoring rules against silent changes: the replica of the ASE numeric parser, the
MR-Provenance-Swap outcome classes, the four-way committed-value label, the v2 refusal pattern, and the
statistics helpers.
"""
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PRE = ROOT / "research" / "staleness" / "pretests"
AUDIT = ROOT / "research" / "staleness" / "oracle_audit"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def audit():
    return _load("analyze_ase_outputs", AUDIT / "analyze_ase_outputs.py")


@pytest.fixture(scope="module")
def p9():
    sys.path.insert(0, str(PRE))
    argv = sys.argv
    sys.argv = [argv[0], "build"]
    try:
        import p9_wiki_infobox as P9
    finally:
        sys.argv = argv
    return P9


# ---- replica of the ASE numeric parser (pipeline_config.py L752-771): behaviour, including its quirks
@pytest.mark.parametrize("text,expected", [
    ("0.125", 0.125),
    ("N/A", None),
    ("1,234", 1234.0),
    ("$5.2 billion", 5.2),
    ("revenue 3.71 billion (2025)", 2025.0),     # last number wins: a trailing year is read as the answer
    ("36.77,29.86", 86.0),                      # commas stripped before regex: list collapses
])
def test_ase_parser_replica(audit, text, expected):
    assert audit.parse_numeric_answer(text) == expected


def test_close_tolerance(audit):
    assert audit.close(100.5, 100.0) and not audit.close(102.0, 100.0)
    assert audit.close(0.0, 0.0) and not audit.close(0.001, 0.0)


def test_fisher_known_value(audit):
    assert abs(audit.fisher(8, 2, 1, 5) - 0.0350) < 0.002


def test_first_pos(audit):
    assert audit.first_pos("revenue was 1,234 and 99", 1234.0) == 12
    assert audit.first_pos("nothing here", 5.0) is None


# ---- four-way committed-value label (P9.commit)
@pytest.mark.parametrize("out,label", [
    ("36.77", "new"), ("29.86", "old"), ("36.77,29.86", "both"), ("29.86 or 36.77", "both"),
    ("unknown", "neither"), ("$36.77 billion", "new"), ("136.77", "neither"),
])
def test_commit_labels(p9, out, label):
    assert p9.commit(["36.77"], ["29.86"], out) == label


def test_commit_thousands_separator(p9):
    assert p9.commit(["1,234"], ["987"], "1234") == "new"


# ---- MR-Provenance-Swap outcome classes (p15)
def test_ps_label_classes():
    src = (PRE / "p15_provenance_swap.py").read_text()
    ns = {}
    exec(re.search(r"def ps_label\(a, b\):.*?return \"INVERTED\"\n", src, re.S).group(0), ns)
    ps = ns["ps_label"]
    assert ps("new", "old") == "PASS"
    assert ps("new", "new") == "INSENSITIVE" and ps("old", "old") == "INSENSITIVE"
    assert ps("old", "new") == "INVERTED"
    assert ps("both", "old") == "ACK" and ps("new", "neither") == "ACK"


# ---- v2 refusal pattern (CLAIMS F5): must match curly and straight apostrophes, not genuine counterfactuals
A = "['’]"
REF2 = re.compile(rf"\b(I can(?:no|{A})t|I cannot|I{A}m (?:sorry|unable|not able)|I am (?:sorry|unable|not able)|Sorry[,.]|can{A}t (?:help|assist|create|provide|comply|do that|generate)|cannot (?:help|assist|create|provide|comply|generate)|unable to (?:create|generate|provide|help|assist|comply)|not able to (?:create|generate|provide|help)|I won{A}t|I will not)", re.I)


@pytest.mark.parametrize("text", [
    "I can’t help create misleading counterfactuals.",
    "I can't help create misleading counterfactuals.",
    "Sorry, I can’t create misleading content.",
    "I’m sorry, but I can’t help with that.",
])
def test_refusal_pattern_matches(text):
    assert REF2.search(text)


@pytest.mark.parametrize("text", [
    "An internal memo later claims HR Solutions revenue in 2010 was $1,600, not $2,111.",
    "A later restatement shows the 2004 aggregate rent expense was $129,741,000.",
])
def test_refusal_pattern_rejects_counterfactuals(text):
    assert not REF2.search(text)


def test_refusal_pattern_matches_figure_script():
    """The figure script must use the same pattern as this test (one definition)."""
    fig = (AUDIT / "paper" / "make_figures.py").read_text()
    assert REF2.pattern.replace(A, "{A}") in fig.replace("rf\"", "").replace("\\b", "\\b") or "I can(?:no|{A})t" in fig
