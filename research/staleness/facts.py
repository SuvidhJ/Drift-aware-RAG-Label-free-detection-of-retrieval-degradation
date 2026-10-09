"""Fact edits: revisions that change a number a reader might ask for.

Defined from text alone -- no embeddings anywhere -- so that nothing measured against
them can be circular with the retriever being evaluated.

``strict``   the revised sentence is identical to an old one once numbers are masked
             (only the number changed).  Pre-registered; rare (1.4-2.2% of revisions).
``aligned``  the revised sentence's closest old counterpart (shingle Jaccard >= 0.5,
             i.e. the same sentence lightly reworded) carries a different set of
             numbers.  The working definition: 9.7-24.7% of revisions.

A fact question is the revised sentence with its numbers removed, so it does not
contain its own answer.  Its answer is the set of DISTINCTIVE numbers (>= 3 chars,
e.g. 84.3, 2023, 1.5%) that are new in the revised sentence; short integers such as
'2' are excluded because they appear in old documents by coincidence.
"""
from __future__ import annotations

import collections
import re

from drift.freshness import canonical, jaccard, shingles

from . import queries

# A number, with an optional magnitude suffix ('450K', '7B', '1.5M') that is part of
# its value. The word boundary keeps '5 kg' and '5 MB' from reading as thousands/millions.
NUM = re.compile(r"\d+(?:[.,]\d+)*(?:\s?(?:[KMB]|k)\b)?%?")
_SUFFIX = {"K": 10**3, "k": 10**3, "M": 10**6, "B": 10**9}
_WORDS = re.compile(r"(\d+(?:\.\d+)?)\s*(thousand|million|billion)\b", re.I)
_WORD_MULT = {"thousand": 10**3, "million": 10**6, "billion": 10**9}
MIN_ANSWER = 3


def norm(x: str) -> str:
    """A number's value-identity: '17.67%' == '17.67', '20,000' == '20000',
    '450K' == '450,000', '1.20' == '1.2'.

    Comparing spellings instead of values counted a dropped percent sign as a changed
    fact, and dropping the magnitude suffix read '450K -> 500K' as '450 -> 500'.
    Version strings such as '3.8.0' are not single values and are kept verbatim.
    """
    from decimal import Decimal, InvalidOperation

    t = x.strip().rstrip("%").replace(",", "")
    mult = 1
    if t and t[-1] in _SUFFIX:
        mult = _SUFFIX[t[-1]]
        t = t[:-1].strip()
    try:
        return format((Decimal(t) * mult).normalize(), "f")
    except InvalidOperation:
        return t


def spoken_values(text: str) -> set[str]:
    """Values written out in words, e.g. '1.5 million' -- generators answer this way."""
    return {norm(str(__import__("decimal").Decimal(n) * _WORD_MULT[w.lower()]))
            for n, w in _WORDS.findall(text)}


def values(text: str) -> set[str]:
    return {norm(x) for x in NUM.findall(text)}


def nums(s: str) -> collections.Counter:
    return collections.Counter(norm(x) for x in NUM.findall(s))


def mask(s: str) -> str:
    return re.sub(r"\s+", " ", NUM.sub("#", s)).strip()


def strip_nums(s: str) -> str:
    return re.sub(r"\s+", " ", NUM.sub("", s)).strip()


def revised(docs: list[dict]) -> list[int]:
    return [i for i, d in enumerate(docs) if canonical(d["text0"]) != canonical(d["text1"])]


def strict_edits(docs: list[dict]) -> list[tuple[int, str, str]]:
    out = []
    for i in revised(docs):
        old = queries.sentences(docs[i]["text0"])
        oset, by_mask = set(old), {}
        for s in old:
            by_mask.setdefault(mask(s), s)
        for s in queries.sentences(docs[i]["text1"]):
            if s in oset or not NUM.search(s):
                continue
            o = by_mask.get(mask(s))
            if o is not None and o != s:
                out.append((i, s, o))
    return out


def aligned_edits(docs: list[dict], min_jaccard: float = 0.5) -> list[tuple[int, str, str]]:
    out = []
    for i in revised(docs):
        old = queries.sentences(docs[i]["text0"])
        oset = set(old)
        osh = [(o, shingles(canonical(o), 5)) for o in old]
        for s in queries.sentences(docs[i]["text1"]):
            if s in oset or not NUM.search(s) or not osh:
                continue
            ss = shingles(canonical(s), 5)
            best = max(osh, key=lambda t: jaccard(ss, t[1]))
            if jaccard(ss, best[1]) >= min_jaccard and nums(s) != nums(best[0]):
                out.append((i, s, best[0]))
    return out


def fact_questions(docs: list[dict], definition: str = "aligned"):
    """(question texts, gold doc ids, answer sets) for edits with a distinctive answer."""
    edits = aligned_edits(docs) if definition == "aligned" else strict_edits(docs)
    qs, gold, answers = [], [], []
    for i, s, o in edits:
        old_v = values(o)
        new = {x for x in NUM.findall(s) if norm(x) not in old_v and len(x) >= MIN_ANSWER}
        if new:
            qs.append(strip_nums(s))
            gold.append(i)
            answers.append(new)
    return qs, gold, answers
