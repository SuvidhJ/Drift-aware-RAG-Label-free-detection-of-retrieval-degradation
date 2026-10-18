"""Figures for the paper, computed directly from raw records (no hand-typed numbers).

    ~/SemProject/ase-replication/.venv/bin/python research/staleness/oracle_audit/paper/make_figures.py

Inputs:  embeddings/cache/staleness/p15/records.jsonl (MR-PS run);
         ~/SemProject/ase-replication/output (ASE authors' released outputs) for the refusal figure.
Outputs: fig_rates.pdf, fig_classes.pdf, fig_refusals.pdf (this folder).
Palette: validated reference categorical slots 1-4 (dataviz skill; CVD-safe; contrast relief = direct
         value labels + tables in the paper).
"""
import glob
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
P15 = REPO / "embeddings/cache/staleness/p15/records.jsonl"
ASE = Path.home() / "SemProject/ase-replication/output"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
MODELS = [("phi3:mini", "phi3-mini"), ("qwen2.5:7b-instruct", "Qwen2.5-7B"), ("qwen2.5:14b-instruct", "Qwen2.5-14B")]
plt.rcParams.update({"font.family": "serif", "font.size": 8, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False})


def ps_label(a, b):
    if a in ("both", "neither") or b in ("both", "neither"):
        return "ACK"
    if a == "new" and b == "old":
        return "PASS"
    return "INSENSITIVE" if a == b else "INVERTED"


def p15():
    recs = [json.loads(l) for l in open(P15) if l.strip()]
    out = {}
    for g, _ in MODELS:
        R = {(r["q"], r["src"], r["dates"], r["order"]): r["label"] for r in recs if r["gen"] == g}
        qs = sorted({k[0] for k in R})
        cls = Counter(ps_label(R[(q, "REAL", "CORRECT", o)], R[(q, "REAL", "SWAPPED", o)]) for q in qs for o in ("M-LAST", "M-FIRST"))
        n = sum(cls.values())
        pub = sum(R[(q, "REAL", "NONE", "M-LAST")] == "old" for q in qs) / len(qs)
        out[g] = {"pub": pub, "ps": (cls["INSENSITIVE"] + cls["INVERTED"]) / n, "cls": {k: v / n for k, v in cls.items()}, "n": n}
    return out


def style(ax):
    ax.yaxis.grid(True, color=GRID, linewidth=0.6); ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_rates(d):
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    x = range(len(MODELS)); w = 0.36
    for i, (key, lab, col) in enumerate((("pub", "Published CN relation", BLUE), ("ps", "MR-Provenance-Swap", ORANGE))):
        vals = [d[g][key] for g, _ in MODELS]
        xs = [j + (i - 0.5) * (w + 0.02) for j in x]
        ax.bar(xs, vals, width=w, color=col, label=lab, edgecolor=SURF, linewidth=1)
        for xv, v in zip(xs, vals):
            ax.text(xv, v + 0.012, f"{v:.1%}" if v < 0.10 else f"{v:.0%}", ha="center", va="bottom", fontsize=7, color=INK)
    ax.set_xticks(list(x)); ax.set_xticklabels([m for _, m in MODELS])
    ax.set_ylabel("Fault rate (real updates)"); ax.set_ylim(0, 0.62)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style(ax); ax.legend(frameon=False, fontsize=7, loc="upper right")
    fig.tight_layout(); fig.savefig(HERE / "fig_rates.pdf"); fig.savefig(HERE / "fig_rates.png", dpi=200); plt.close(fig)


def fig_classes(d):
    fig, ax = plt.subplots(figsize=(3.5, 2.2))
    order = [("PASS", "Follows the later date", BLUE), ("INSENSITIVE", "Ignores dates", ORANGE),
             ("INVERTED", "Follows the earlier date", YELLOW), ("ACK", "Lists both / abstains", AQUA)]
    ys = range(len(MODELS))[::-1]
    for y, (g, lab) in zip(ys, MODELS):
        left = 0
        for k, _, col in order:
            v = d[g]["cls"].get(k, 0)
            ax.barh(y, v, left=left, color=col, height=0.55, edgecolor=SURF, linewidth=1.2)
            if v >= 0.08:
                ax.text(left + v / 2, y, f"{v:.0%}", ha="center", va="center", fontsize=7, color=INK)
            left += v
    ax.set_yticks(list(ys)); ax.set_yticklabels([m for _, m in MODELS])
    ax.set_xlim(0, 1); ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of date-swap pairs (real updates)")
    ax.tick_params(length=0); ax.spines["left"].set_visible(False)
    handles = [matplotlib.patches.Patch(color=c, label=l) for _, l, c in order]
    fig.legend(handles=handles, frameon=False, fontsize=6.5, ncol=2, loc="upper center", bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.80)); fig.savefig(HERE / "fig_classes.pdf"); fig.savefig(HERE / "fig_classes.png", dpi=200); plt.close(fig)


def fig_refusals():
    A = "['’]"
    REF2 = re.compile(rf"\b(I can(?:no|{A})t|I cannot|I{A}m (?:sorry|unable|not able)|I am (?:sorry|unable|not able)|Sorry[,.]|can{A}t (?:help|assist|create|provide|comply|do that|generate)|cannot (?:help|assist|create|provide|comply|generate)|unable to (?:create|generate|provide|help|assist|comply)|not able to (?:create|generate|provide|help)|I won{A}t|I will not)", re.I)
    dsets = [("repliqa", "RepLiQA"), ("ConvFinQA", "ConvFinQA"), ("FinQA", "FinQA"), ("TAT-DQA", "TAT-DQA"), ("VQAonBD", "VQAonBD")]
    rate = {}
    for ds, _ in dsets:
        for sc in ("pre_chunk", "post_chunk"):
            n = k = 0
            for f in glob.glob(str(ASE / ds / f"{sc}_context*" / "step3_mutation_results*.json")):
                for m in json.load(open(f)):
                    if m["mutation_operator"].startswith("counterfactual_noise"):
                        cf = " ".join(str(l.get("generated_counterfactual", "")) for l in m.get("mutation_log", []))
                        n += 1; k += bool(REF2.search(cf))
            rate[(ds, sc)] = k / n
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    w = 0.36
    for i, (sc, lab, col) in enumerate((("pre_chunk", "Pre-chunk", BLUE), ("post_chunk", "Post-chunk", ORANGE))):
        xs = [j + (i - 0.5) * (w + 0.02) for j in range(len(dsets))]
        vals = [rate[(ds, sc)] for ds, _ in dsets]
        ax.bar(xs, vals, width=w, color=col, label=lab, edgecolor=SURF, linewidth=1)
        for xv, v in zip(xs, vals):
            ax.text(xv, v + 0.01, f"{v:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(range(len(dsets))); ax.set_xticklabels([l for _, l in dsets], fontsize=6.5)
    ax.set_ylabel("CN mutants that are refusals"); ax.set_ylim(0, 0.58)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style(ax); fig.legend(frameon=False, fontsize=7, ncol=2, loc="upper center", bbox_to_anchor=(0.55, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.90)); fig.savefig(HERE / "fig_refusals.pdf"); fig.savefig(HERE / "fig_refusals.png", dpi=200); plt.close(fig)
    return rate


if __name__ == "__main__":
    import matplotlib.ticker  # noqa: F401
    import matplotlib.patches  # noqa: F401
    d = p15()
    fig_rates(d); fig_classes(d); r = fig_refusals()
    for g, lab in MODELS:
        print(f"{lab}: published {d[g]['pub']:.3f}, MR-PS {d[g]['ps']:.3f}, classes {{{', '.join(f'{k}: {v:.3f}' for k, v in d[g]['cls'].items())}}}")
    print("refusal rates:", {f"{k[0]}/{k[1][:4]}": round(v, 3) for k, v in r.items()})
