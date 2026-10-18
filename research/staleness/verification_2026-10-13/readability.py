"""Readability of the manuscript, per section: Flesch Reading Ease (higher = easier) and
Flesch-Kincaid grade (US school grade needed to follow the text).

    python3 research/staleness/verification_2026-10-13/readability.py research/staleness/oracle_audit/paper/main.tex
"""
import re
import sys


def clean(s):
    s = re.sub(r'\\begin\{(table|figure|lstlisting)\*?\}.*?\\end\{\1\*?\}', ' ', s, flags=re.S)
    s = re.sub(r'%.*', '', s)
    s = re.sub(r'\$[^$]*\$', 'X', s)
    s = re.sub(r'\\(cite|citet|citep)\{[^}]*\}', 'Ref', s)
    s = re.sub(r'\\(ref|label)\{[^}]*\}', '1', s)
    s = re.sub(r'\\(emph|textsc|texttt|path)\{([^}]*)\}', r'\2', s)
    s = s.replace(r'\cn{}', 'CN').replace(r'\cn', 'CN').replace(r'\mrps{}', 'MR-PS').replace(r'\mrps', 'MR-PS')
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', s)
    s = s.replace('~', ' ').replace('{', '').replace('}', '').replace("``", '"').replace("''", '"')
    return re.sub(r'\s+', ' ', s.replace('\\%', '%'))


def syllables(w):
    w = w.lower()
    if len(w) <= 3:
        return 1
    w = re.sub(r'(?:[^laeiouy]es|ed|[^laeiouy]e)$', '', w)
    w = re.sub(r'^y', '', w)
    return max(1, len(re.findall(r'[aeiouy]{1,2}', w)))


def score(t):
    t = re.sub(r'\b(e\.g|i\.e|et al|vs|Fig|cf)\.', lambda m: m.group(0).replace('.', ''), t)
    t = re.sub(r'(\d)\.(\d)', r'\1\2', t)
    sents = [s for s in re.split(r'(?<=[.!?])\s+(?=[A-Z"(])', t) if len(s.split()) >= 3]
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", t)
    syl = sum(syllables(w) for w in words)
    W, S = len(words), max(1, len(sents))
    return dict(words=W, sent_len=round(W / S, 1),
                fre=round(206.835 - 1.015 * W / S - 84.6 * syl / W, 1),
                grade=round(0.39 * W / S + 11.8 * syl / W - 15.59, 1),
                long=round(100 * sum(syllables(w) >= 3 for w in words) / W, 1))


def main(path):
    src = open(path).read()
    src = src[src.index(r'\begin{abstract}'):src.index(r'\section*{Data availability}')]
    chunks = re.split(r'\\section\{', src)
    rows = [('Abstract', chunks[0].split(r'\end{abstract}')[0])]
    rows += [(c.split('}')[0][:42], c.split('}', 1)[1]) for c in chunks[1:]]
    print(f"{'section':<44}{'words':>6}{'sentLen':>8}{'FRE':>7}{'grade':>7}{'%3+syl':>8}")
    for name, txt in rows + [('WHOLE PAPER', src)]:
        r = score(clean(txt))
        print(f"{name:<44}{r['words']:>6}{r['sent_len']:>8}{r['fre']:>7}{r['grade']:>7}{r['long']:>8}")


if __name__ == '__main__':
    main(sys.argv[1])
