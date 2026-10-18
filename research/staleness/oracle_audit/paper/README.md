# Paper: LaTeX source (Elsevier elsarticle; target JSS / IST)

- `main.tex`: manuscript (draft v5, 2026-10-18; structure from the ACM SIGSOFT Empirical Standards, plain-language prose (Flesch Reading Ease 61.0, grade 9.0; see verification_2026-10-13/readability.py); numbers unchanged from v1 of 2026-10-14). Red `[PENDING: …]`
  markers show what is still open: the human validation, affiliations, CRediT and the
  AI-declaration review.
- `ieee/main_ieee.tex`: the same manuscript in IEEE journal format (IEEEtran, two columns, numeric
  citations via IEEEtranN; 9 pages). It is GENERATED from `main.tex`: edit `main.tex` only, then run
  `python3 ieee/make_ieee.py` and recompile. Under XeTeX/Tectonic it loads TeX Gyre fonts; on
  Overleaf (pdfLaTeX) it uses IEEEtran's default fonts.
- `refs.bib`: references, each verified against arXiv metadata or Crossref DOIs on 2026-10-14.
- **Compile**, either way:
  - `tectonic -X compile main.tex` (the release binary from
    github.com/tectonic-typesetting/tectonic needs no admin rights); or
  - upload `main.tex` and `refs.bib` to Overleaf (elsarticle is built in).
- Current build: 22 pages (IEEE version: 9 pages); 0 unresolved citations or references.
- **Before submission:**
  - switch `\documentclass[preprint,12pt,authoryear]` to the journal's required options;
  - add the highlights file Elsevier requires (3–5 bullets, ≤ 85 characters each);
  - check the generative-AI declaration wording.
