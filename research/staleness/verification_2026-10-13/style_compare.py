import re, sys, statistics as st
from collections import Counter

def body_from_tex(path):
    s = open(path).read()
    s = s[s.index(r'\begin{abstract}'):s.index(r'\section*{Data availability}')]
    s = re.sub(r'\\begin\{(table|figure|lstlisting)\*?\}.*?\\end\{\1\*?\}', ' ', s, flags=re.S)
    s = re.sub(r'%.*', '', s)
    s = re.sub(r'\$[^$]*\$', 'X', s)
    s = re.sub(r'\\(cite|citet|citep)\{[^}]*\}', 'Ref', s)
    s = re.sub(r'\\(ref|label)\{[^}]*\}', '1', s)
    s = re.sub(r'\\(emph|textsc|texttt|path|item)\{([^}]*)\}', r'\2', s)
    s = s.replace(r'\cn{}', 'CN').replace(r'\cn', 'CN').replace(r'\mrps{}', 'MR-PS').replace(r'\mrps', 'MR-PS')
    s = re.sub(r'\\[a-zA-Z]+\*?(\[[^\]]*\])?', ' ', s)
    s = s.replace('~', ' ').replace('{', '').replace('}', '').replace("``", '"').replace("''", '"').replace('\\%','%').replace('---',' - ').replace('--','-')
    return s

def body_from_txt(path):
    t = open(path).read()
    t = t[t.index('Abstract'):]
    cut = t.rfind('REFERENCES')
    t = t[:cut] if cut > 0 else t
    t = re.sub(r'-\s*\n\s*', '', t)            # de-hyphenate line breaks
    t = re.sub(r'\s+', ' ', t)
    return t

def stats(t):
    t = re.sub(r'\s+', ' ', t)
    t2 = re.sub(r'\b(e\.g|i\.e|et al|vs|Fig|Sec|cf|Eq)\.', lambda m: m.group(0).replace('.', ''), t)
    t2 = re.sub(r'(\d)\.(\d)', r'\1\2', t2)
    sents = [x.strip() for x in re.split(r'(?<=[.!?])\s+(?=[A-Z"(])', t2) if len(x.split()) >= 3]
    L = [len(x.split()) for x in sents]
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", t)
    W = len(words); lw = [w.lower() for w in words]
    c = Counter(lw)
    per = lambda n: round(1000 * n / W, 1)
    def cnt(lst): return sum(c[w] for w in lst)
    passive = len(re.findall(r'\b(is|are|was|were|been|be|being)\s+(\w+ly\s+)?\w+(ed|en)\b', t, re.I))
    out = {
        'words': W, 'sentences': len(sents),
        'sent len mean': round(st.mean(L), 1), 'sent len median': st.median(L), 'sent len SD': round(st.pstdev(L), 1),
        '% sents <=12 words': round(100 * sum(l <= 12 for l in L) / len(L)), '% sents >=30 words': round(100 * sum(l >= 30 for l in L) / len(L)),
        'we/our/us per 1k': per(cnt(['we', 'our', 'us'])),
        'passives per 100 sents': round(100 * passive / len(sents)),
        'hedges per 1k (may/might/could/seems/likely/perhaps/possibly/suggest*)': per(cnt(['may','might','could','seems','seem','likely','perhaps','possibly','suggest','suggests','arguably'])),
        'boosters per 1k (clearly/very/much/important/essential/indeed/simply/really)': per(cnt(['clearly','very','much','important','essential','indeed','simply','really','obviously'])),
        'connectives per 1k (however/therefore/thus/hence/but/although/because/so)': per(cnt(['however','therefore','thus','hence','but','although','because','so'])),
        'semicolons per 1k': per(t.count(';')), 'colons per 1k': per(t.count(':')), 'parentheses per 1k': per(t.count('(')),
        'e.g./i.e. per 1k': per(len(re.findall(r'\b(e\.g\.|i\.e\.)', t))),
        'sentence starts The/This/These %': round(100 * sum(x.split()[0] in ('The','This','These') for x in sents) / len(sents)),
        'sentence starts We %': round(100 * sum(x.split()[0] == 'We' for x in sents) / len(sents)),
        'type-token ratio (first 3000 words)': round(len(set(lw[:3000])) / 3000, 3),
        '"rather than"/"instead of" per 1k': per(len(re.findall(r'\b(rather than|instead of)\b', t, re.I))),
        '"not only"/"in other words"/"in particular" per 1k': per(len(re.findall(r'\b(not only|in other words|in particular)\b', t, re.I))),
    }
    return out, sents

if __name__ == '__main__':
    ours, s1 = stats(body_from_tex(sys.argv[1]))
    theirs, s2 = stats(body_from_txt(sys.argv[2]))
    w = max(len(k) for k in ours)
    print(f"{'feature':<{w}}  {'ours (v3)':>10}  {'Fraser&Arcuri 2012':>18}")
    for k in ours: print(f"{k:<{w}}  {ours[k]!s:>10}  {theirs[k]!s:>18}")
