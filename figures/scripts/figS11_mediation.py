"""
Supplementary Figure — Exploratory COX8A cis-eQTM mediation
(Demoted from main Fig 6: hypothesis-generating only. The cis-eQTM screen has NO
FDR-significant pair; the total effect for COX8A is not significant; and the
proportion-mediated estimate is unstable. Shown as an illustrative example of a
possible transcriptome–methylome coupling, not an established mechanism.)

Panels:
  A: COX8A mediation path diagram (X = group HC→RC, M = CpG, Y = COX8A expression)
  B: ACME forest — all 6 gene×CpG pairs tested

No suptitle (caption added in manuscript).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings("ignore")

BASE = "PROJECT_ROOT/research/LongCOVID/Results"
NEW  = BASE + "_updated_260714"
OUT  = f"{NEW}/manuscript/figures"
MED_FILE = f"{BASE}/5.Integration/2.Mediation/data/mediation_results.tsv"  # mediation not re-run (exploratory, unchanged)
N_BOOT = 1000

TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS, ANNOT_FS, PANEL_FS = 12, 11, 9.5, 9.5, 9, 16

plt.rcParams.update({
    "font.family": "Arial", "font.size": 11, "axes.linewidth": 0.9,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.9, "ytick.major.width": 0.9,
    "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})

SIG_COL, NSIG_COL, MED_COL, NEG_COL = "#E63946", "#AAAAAA", "#2A9D8F", "#457B9D"


def add_panel_label(ax, lab):
    ax.annotate(lab, xy=(0, 1), xycoords="axes fraction",
                xytext=(-30, 14), textcoords="offset points",
                fontsize=PANEL_FS, fontweight="bold", va="bottom", ha="left")


def panel_A(ax, med_df):
    row = med_df[med_df["cpg"] == "chr11_64304843"].iloc[0]
    a, b, ade = row["alpha_XtoM"], row["gamma_MtoY"], row["ade"]
    acme, lo, hi = row["acme"], row["acme_ci_lo"], row["acme_ci_hi"]

    ax.set_xlim(0, 10); ax.set_ylim(0, 6.4); ax.axis("off")
    X, Y, M = (1.9, 4.5), (8.1, 4.5), (5.0, 1.55)
    BW, BH = 2.7, 1.25
    NODE_LIGHT = {"#333": "#ECECEC", MED_COL: "#D0F0ED", NEG_COL: "#D0E4F0"}

    def node(cx, cy, label, sub, col, fs):
        ax.add_patch(mpatches.FancyBboxPatch(
            (cx-BW/2, cy-BH/2), BW, BH, boxstyle="round,pad=0.10",
            linewidth=1.6, edgecolor=col, facecolor=NODE_LIGHT.get(col, "#ECECEC")))
        ax.text(cx, cy+0.18, label, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=col)
        ax.text(cx, cy-0.30, sub, ha="center", va="center", fontsize=ANNOT_FS, color="#555")

    node(*X, "Recovery group", "HC=0 → RC=1  (X)", NEG_COL, LABEL_FS)
    node(*Y, "COX8A", "expression, logCPM  (Y)", "#333", LABEL_FS + 2)
    node(*M, "chr11_64304843", "CpG methylation (M)", MED_COL, LABEL_FS)

    def arrow(x1, y1, x2, y2, col):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, mutation_scale=16))

    arrow(X[0]+BW/2, X[1], Y[0]-BW/2, Y[1], "#666")
    ax.text(5.0, X[1]+0.55, f"ADE (direct) = {ade:+.2f}", ha="center",
            fontsize=ANNOT_FS+1, color="#555", style="italic")
    arrow(X[0]+0.2, X[1]-BH/2, M[0]-BW/2+0.15, M[1]+BH/2, MED_COL)
    ax.text(3.02, 2.95, f"a = {a:+.2f}\n(group→CpG)", ha="center", va="center",
            fontsize=ANNOT_FS+1, color=MED_COL, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=MED_COL, lw=0.8))
    arrow(M[0]+BW/2-0.15, M[1]+BH/2, Y[0]-0.2, Y[1]-BH/2, MED_COL)
    ax.text(6.98, 2.95, f"b = {b:+.3f}\n(CpG→COX8A)", ha="center", va="center",
            fontsize=ANNOT_FS+1, color=MED_COL, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=MED_COL, lw=0.8))

    # honest caveat box (no proportion-mediated headline)
    ax.text(5.0, 0.34,
            f"ACME (indirect) = {acme:.4f}   95% CI [{lo:.3f}, {hi:.3f}]\n"
            f"Exploratory: cis-eQTM FDR n.s.; total effect n.s.; "
            f"proportion-mediated unstable",
            ha="center", va="center", fontsize=ANNOT_FS, color="#555",
            bbox=dict(boxstyle="round,pad=0.35", fc="#f4f4f4", ec="#BBB", lw=1.0))

    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")


def panel_B(ax, med_df):
    med_df = med_df.copy().sort_values("acme")
    for i, row in enumerate(med_df.itertuples()):
        col = SIG_COL if row.acme_sig else NSIG_COL
        ax.plot([row.acme_ci_lo, row.acme_ci_hi], [i, i], color=col, lw=1.6, zorder=3)
        ax.scatter(row.acme, i, color=col, s=85 if row.acme_sig else 45, zorder=4,
                   edgecolors="white", linewidths=0.8)
        if row.acme_sig:
            ax.text(row.acme_ci_hi + 0.004, i, f"ACME={row.acme:.4f} (nominal)",
                    fontsize=ANNOT_FS - 0.5, color=col, va="center", fontweight="bold")
    ax.axvline(0, color="#888", lw=0.9, ls="--")
    ax.set_yticks(range(len(med_df)))
    ax.set_yticklabels(med_df["cpg_label"].tolist(), fontsize=TICK_FS)
    ax.set_xlabel("ACME (indirect effect)", fontsize=LABEL_FS)
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(-0.11, 0.085)
    handles = [mpatches.Patch(color=SIG_COL, label="ACME CI excludes 0 (nominal)"),
               mpatches.Patch(color=NSIG_COL, label="Not significant")]
    ax.legend(handles=handles, fontsize=LEGEND_FS, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)


def main():
    med_df = pd.read_csv(MED_FILE, sep="\t")
    fig = plt.figure(figsize=(15, 5.6))
    gs = GridSpec(1, 2, figure=fig, left=0.05, right=0.985, top=0.9, bottom=0.16,
                  wspace=0.28)
    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    panel_A(ax_A, med_df)
    panel_B(ax_B, med_df)
    for ax, lab in [(ax_A, "A"), (ax_B, "B")]:
        add_panel_label(ax, lab)
    out_png = f"{OUT}/figS11_mediation.png"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/figS11_mediation.pdf", bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
