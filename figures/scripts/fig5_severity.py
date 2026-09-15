"""
Fig 5 — Acute severity bifurcates the blood transcriptome via cell composition
        but converges completely at convalescence
Panels:
  A: Persistent-UP gene-set score (MM vs SC) across timepoints
  B: Forest plot — Cohen's d (MM − SC) per timepoint, Persistent UP + Delayed UP
  C: Cell composition at T1 (MM vs SC) — top cell types by |d| (CIBERSORTx LM22)
  D: Cell-composition correction at T1 — raw vs adjusted Cohen's d

No figure-level suptitle (title/caption added in manuscript).
Note: SC group is small (n = 5–21 across timepoints); later timepoints noisier.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE = "PROJECT_ROOT/research/LongCOVID/Results"
NEW  = BASE + "_updated_260714"
OUT  = f"{NEW}/manuscript/figures"

MWU_FILE  = f"{NEW}/5.Integration/3.Severity/data/severity_mwu_clean.tsv"           # clean cohort
# Panel C/D use T1 (acute) samples only; RC exclusions don't affect AC T1 → original files valid.
CELL_FILE = f"{BASE}/5.Integration/3.Severity/data/t1_cellcomp_comparison.tsv"
ADJ_FILE  = f"{NEW}/5.Integration/3.Severity/data/t1_score_adjusted.tsv"            # ANCOVA re-run on filtered clean scores (2026-07-30)

MM_COL = "#457B9D"    # blue — MM (mild-moderate)
SC_COL = "#E63946"    # red  — SC (severe-critical)
DELAYED_COL = "#F4A261"
RAW_D_COL = "#E4B429"  # gold/amber — "Raw d" in Panel C (kept distinct from the
                       # red SC/severity colour so the two don't get conflated)
FDR_TH = 0.05

# ── font sizes ─────────────────────────────────────────────────────────────
TITLE_FS  = 12
LABEL_FS  = 11
TICK_FS   = 9.5
LEGEND_FS = 9.5
ANNOT_FS  = 9
PANEL_FS  = 16

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

TP_ORDER  = ["T0", "T1", "T2", "T3", "RC"]
TP_LABELS = {"T0": "T0\n(D0)", "T1": "T1\n(D7)", "T2": "T2\n(D14)",
             "T3": "T3\n(D21)", "RC": "RC"}
TP_XPOS   = [0, 1, 2, 3, 4]


def add_panel_label(ax, lab):
    ax.annotate(lab, xy=(0, 1), xycoords="axes fraction",
                xytext=(-42, 16), textcoords="offset points",
                fontsize=PANEL_FS, fontweight="bold", va="bottom", ha="left")


def panel_A_trajectories(axes, mwu_df):
    """3 UP gene-set score trajectories (MM vs SC) side by side — shows the T1
    bifurcation and complete RC convergence is shared across Persistent / Delayed
    / Acute-specific sets (not persistent-specific)."""
    sets = [("score_persistent_up", "Persistent UP"),
            ("score_delayed_up",    "Delayed UP"),
            ("score_acute_only_up", "Acute-specific UP")]
    xlabels = [TP_LABELS[tp] for tp in TP_ORDER]
    for ax, (sc, name) in zip(axes, sets):
        sub = mwu_df[mwu_df["score"] == sc].copy()
        sub["o"] = sub["timepoint"].map({tp: i for i, tp in enumerate(TP_ORDER)})
        sub = sub.sort_values("o")
        ax.plot(TP_XPOS, sub["mean_MM"].values, "-o", color=MM_COL, ms=6, lw=2,
                label="MM", zorder=3)
        ax.plot(TP_XPOS, sub["mean_SC"].values, "-s", color=SC_COL, ms=6, lw=2,
                label="SC", zorder=3)
        ax.axvspan(0.8, 1.2, color="#E63946", alpha=0.07, zorder=0)
        # significance marker at EVERY timepoint (* = FDR<0.05, else n.s.)
        for _, row in sub.iterrows():
            ti = TP_ORDER.index(row["timepoint"])
            ytop = max(row["mean_MM"], row["mean_SC"])
            if row["fdr"] < FDR_TH:
                ax.text(ti, ytop + 0.10, "*", ha="center", va="bottom",
                        fontsize=14, color="#333", fontweight="bold")
            else:
                ax.text(ti, ytop + 0.10, "n.s.", ha="center", va="bottom",
                        fontsize=ANNOT_FS, color="#333")
        ax.axhline(0, color="#CCC", lw=0.7, ls="--", zorder=1)
        ax.set_xticks(TP_XPOS)
        ax.set_xticklabels(xlabels, fontsize=TICK_FS - 2)
        ax.set_title(name, fontsize=ANNOT_FS + 1, fontweight="bold")  # subplot id
        ax.tick_params(labelsize=TICK_FS - 1)

    axes[0].set_ylim(top=axes[0].get_ylim()[1] + 0.28)   # headroom for markers
    axes[0].set_ylabel("Gene-set score (z)\nMM vs SC", fontsize=LABEL_FS)
    axes[0].legend(fontsize=LEGEND_FS, frameon=False, loc="upper right")


def panel_B_heatmap(ax, mwu_df):
    """Cohen's d (MM-SC) heatmap: 6 gene sets x 5 timepoints. * = FDR<0.05.
    Shows the T1 severity bifurcation is shared across ALL sets (not persistent-
    specific) and that all sets converge (n.s.) by RC."""
    score_order = ["score_persistent_up", "score_delayed_up", "score_acute_only_up",
                   "score_persistent_down", "score_delayed_down", "score_acute_only_down"]
    row_labels  = ["Persistent UP", "Delayed UP", "Acute-specific UP",
                   "Persistent DOWN", "Delayed DOWN", "Acute-specific DOWN"]
    D = np.full((len(score_order), len(TP_ORDER)), np.nan)
    F = np.full_like(D, np.nan)
    for i, sc in enumerate(score_order):
        sub = mwu_df[mwu_df["score"] == sc]
        for j, tp in enumerate(TP_ORDER):
            r = sub[sub["timepoint"] == tp]
            if not r.empty:
                D[i, j] = r["cohen_d"].values[0]; F[i, j] = r["fdr"].values[0]

    im = ax.imshow(D, cmap="RdBu_r", vmin=-1.1, vmax=1.1, aspect="auto")
    for i in range(len(score_order)):
        for j in range(len(TP_ORDER)):
            if np.isnan(D[i, j]):
                continue
            star = "*" if F[i, j] < FDR_TH else ""
            txtcol = "white" if abs(D[i, j]) > 0.6 else "#333"
            ax.text(j, i, f"{D[i,j]:.2f}{star}", ha="center", va="center",
                    fontsize=ANNOT_FS, color=txtcol,
                    fontweight="bold" if star else "normal")
    ax.set_xticks(range(len(TP_ORDER)))
    ax.set_xticklabels([TP_LABELS[tp] for tp in TP_ORDER], fontsize=TICK_FS)
    ax.set_yticks(range(len(score_order)))
    ax.set_yticklabels(row_labels, fontsize=TICK_FS)
    ax.tick_params(length=0)
    # highlight T1 column
    ax.add_patch(plt.Rectangle((0.5, -0.5), 1, len(score_order), fill=False,
                               edgecolor="#222", lw=1.8))
    cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Cohen's d (MM − SC)", fontsize=ANNOT_FS)
    cb.ax.tick_params(labelsize=TICK_FS - 1)


def panel_C_cellcomp(ax, cell_df):
    sig = cell_df[cell_df["pval"] < 0.05].copy()
    sig = sig.sort_values("cohen_d", key=abs, ascending=False).head(6)
    sig["label"] = (sig["cell_type"].str.replace("_", " ")
                    .str.replace("T cells", "T"))
    sig = sig.iloc[::-1]   # largest |d| on top

    y = np.arange(len(sig))
    h = 0.38
    ax.barh(y + h/2, sig["mean_MM"], color=MM_COL, alpha=0.85, height=h, label="MM")
    ax.barh(y - h/2, sig["mean_SC"], color=SC_COL, alpha=0.85, height=h, label="SC")

    for i, row in enumerate(sig.itertuples()):
        x = max(row.mean_MM, row.mean_SC) + 0.006
        mark = "**" if row.fdr < FDR_TH else "*"
        ax.text(x, i, f"{mark}  d={row.cohen_d:+.2f}", va="center",
                fontsize=LABEL_FS + 1, color="#444")

    ax.set_yticks(y)
    ax.set_yticklabels(sig["label"].tolist(), fontsize=LABEL_FS + 1.5)
    ax.set_xlabel("CIBERSORTx cell fraction", fontsize=LABEL_FS)
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="upper right")
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(0, sig[["mean_MM", "mean_SC"]].values.max() * 1.30)


def panel_D_adjusted(ax, adj_df):
    # grouped by gene-set family (Persistent / Delayed / Acute-specific) with UP then
    # DOWN adjacent within each family, so each up/down pair reads together.
    # Each bar-pair is ticked UP / DOWN; the full family word is written once,
    # centred under the pair (avoids repeating long words like "Acute-specific").
    sets_to_show = [("score_persistent_up",   "UP",   "Persistent"),
                    ("score_persistent_down", "DOWN", "Persistent"),
                    ("score_delayed_up",      "UP",   "Delayed"),
                    ("score_delayed_down",    "DOWN", "Delayed"),
                    ("score_acute_only_up",   "UP",   "Acute-specific"),
                    ("score_acute_only_down", "DOWN", "Acute-specific")]
    raw_vals, adj_vals, tick_labels, families = [], [], [], []
    for sc, ud, fam in sets_to_show:
        row = adj_df[adj_df["score"] == sc]
        if row.empty:
            continue
        row = row.iloc[0]
        raw_vals.append(row["d_raw"]); adj_vals.append(row["d_adj"])
        tick_labels.append(ud); families.append(fam)

    # x positions: adjacent within a family, extra gap between families
    GROUP_GAP = 0.8
    xpos, p = [], 0.0
    for i in range(len(tick_labels)):
        xpos.append(p)
        p += 1.0
        if i % 2 == 1:            # after each DOWN → end of a family pair
            p += GROUP_GAP
    xpos = np.array(xpos); w = 0.38

    ax.bar(xpos - w/2, raw_vals, width=w, color=RAW_D_COL, alpha=0.85, label="Raw d")
    ax.bar(xpos + w/2, adj_vals, width=w, color="#9AA3AB", alpha=0.9,
           label="Adjusted d (cell-composition corrected)")

    for xi, rv, av in zip(xpos, raw_vals, adj_vals):
        ax.annotate("", xy=(xi + w/2, av), xytext=(xi - w/2, rv),
                    arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.2))
        pct = (rv - av) / abs(rv) * 100
        ax.text(xi, max(rv, av) + 0.03, f"−{pct:.0f}%", ha="center",
                fontsize=ANNOT_FS + 0.5, color="#333")

    ax.axhline(0, color="#888", lw=0.9)
    ax.set_xticks(xpos)
    ax.set_xticklabels(tick_labels, fontsize=TICK_FS + 1, rotation=0, ha="center")

    # family bracket + label once, centred under each up/down pair
    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for i in range(0, len(families), 2):
        c = (xpos[i] + xpos[i + 1]) / 2
        ax.plot([xpos[i] - 0.42, xpos[i + 1] + 0.42], [-0.135, -0.135],
                transform=tr, color="#555", lw=1.0, clip_on=False)
        ax.text(c, -0.205, families[i], transform=tr, ha="center", va="top",
                fontsize=TICK_FS + 1.5, fontweight="bold", color="#222",
                clip_on=False)

    ax.set_ylabel("Cohen's d (MM − SC) at T1", fontsize=LABEL_FS)
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="upper right")
    ax.tick_params(axis="y", labelsize=TICK_FS + 1)
    ax.set_xlim(xpos[0] - 0.7, xpos[-1] + 0.7)
    ax.set_ylim(min(0, min(adj_vals)) - 0.12, max(raw_vals) * 1.30)


def main():
    mwu_df  = pd.read_csv(MWU_FILE,  sep="\t")
    cell_df = pd.read_csv(CELL_FILE, sep="\t")
    adj_df  = pd.read_csv(ADJ_FILE,  sep="\t")

    fig = plt.figure(figsize=(15, 9.5))
    # top row: A = 3 UP-set trajectories (MM vs SC, shared y)
    gs_top = GridSpec(1, 3, figure=fig, left=0.08, right=0.97, top=0.94,
                      bottom=0.58, wspace=0.18)
    # bottom row: B cell composition / C cell-composition adjustment
    gs_bot = GridSpec(1, 2, figure=fig, left=0.10, right=0.95, top=0.44,
                      bottom=0.10, wspace=0.32)

    a1 = fig.add_subplot(gs_top[0, 0])
    a2 = fig.add_subplot(gs_top[0, 1], sharey=a1)
    a3 = fig.add_subplot(gs_top[0, 2], sharey=a1)
    ax_B = fig.add_subplot(gs_bot[0, 0])
    ax_C = fig.add_subplot(gs_bot[0, 1])

    panel_A_trajectories([a1, a2, a3], mwu_df)
    panel_C_cellcomp(ax_B, cell_df)      # cell composition → Panel B
    panel_D_adjusted(ax_C, adj_df)       # cell-composition adjustment → Panel C

    add_panel_label(a1, "A")
    for ax, lab in [(ax_B, "B"), (ax_C, "C")]:
        add_panel_label(ax, lab)

    out_png = f"{OUT}/fig5_severity.png"
    out_pdf = f"{OUT}/fig5_severity.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
