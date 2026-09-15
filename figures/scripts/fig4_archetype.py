"""
Fig 4 — 5-Archetype classification
Panels:
  A: Recovery Index (RI) histogram — coloured by archetype, dashed boundaries
  B: Scatter RI × T0 Δz — archetype colour + boundary lines; data-driven
     top-Escalated genes labelled with adjustText (no overlap)
  C: Donut — archetype proportions
  D: 5 small trajectory plots, one per archetype (mean ± SEM)

No figure-level suptitle (title/caption added in manuscript).
Counts (archetype_summary_v2): Escalated 15 / Plateau 139 / Partial 89 /
Recovered 11 / Overshoot 9  (total 263).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from matplotlib.gridspec import GridSpec
from adjustText import adjust_text
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE = "PROJECT_ROOT/research/LongCOVID/Results"
NEW  = BASE + "_updated_260714"
OUT  = f"{NEW}/manuscript/figures"

ARC_FILE = f"{NEW}/4.Trajectory/3.Archetypes/data/gene_archetypes_filtered.tsv"  # canonical, IG/pseudo removed (2026-07-30)
SUM_FILE = f"{BASE}/4.Trajectory_ver2/3.Archetypes/data/archetype_summary_v2.tsv"  # unused (donut panel not drawn)

# ── colour palette (archetype) ─────────────────────────────────────────────
ARC_COLORS = {
    "Escalated":  "#9B2226",
    "Plateau":    "#E63946",
    "Partial":    "#F4A261",
    "Recovered":  "#A8DADC",
    "Overshoot":  "#457B9D",
}
ARC_ORDER = ["Escalated", "Plateau", "Partial", "Recovered", "Overshoot"]

# data stores the archetype as "Plateau"; render it as "Sustained" (RI 0.7–1.5
# group = acute-level change retained, no recovery). Keep keys for data lookups.
DISPLAY_NAME = {
    "Escalated": "Escalated",
    "Plateau":   "Sustained",
    "Partial":   "Partial",
    "Recovered": "Resolving",
    "Overshoot": "Overshoot",
}

RI_BOUNDS = {
    "Escalated":  (1.5, np.inf),
    "Plateau":    (0.7, 1.5),
    "Partial":    (0.2, 0.7),
    "Recovered":  (-0.5, 0.2),
    "Overshoot":  (-np.inf, -0.5),
}

# genes to highlight in Panel B (archetype/RI taken live from data)
# data-driven: top Escalated genes by convalescent deviation |Δ z at RC|
#   (named, non-globin/lncRNA/pseudogene) — the genes that WORSEN most by RC
DATADRIVEN_GENES = ["H2AC18", "ANKRD36C", "CIBAR1", "CDKN1C"]

# ── font sizes (enlarged for readability) ──────────────────────────────────
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


def parse_gene(gid):
    parts = str(gid).split("_", 1)
    return parts[1] if len(parts) > 1 else gid


def add_panel_label(ax, lab):
    ax.annotate(lab, xy=(0, 1), xycoords="axes fraction",
                xytext=(-38, 16), textcoords="offset points",
                fontsize=PANEL_FS, fontweight="bold", va="bottom", ha="left")


def panel_A_histogram(ax, arc_df):
    ri = arc_df["RI"].clip(-3, 4)
    bins = np.linspace(-3, 4, 50)

    for arch in ARC_ORDER[::-1]:
        lo, hi = RI_BOUNDS[arch]
        subset = ri[(ri >= lo) & (ri < hi)]
        ax.hist(subset, bins=bins, color=ARC_COLORS[arch],
                alpha=0.85, edgecolor="white", linewidth=0.3, label=arch)

    for thresh in [1.5, 0.7, 0.2, -0.5]:
        ax.axvline(thresh, color="#444", lw=0.8, ls="--", alpha=0.6)

    ax.set_xlabel("Recovery Index (RI)", fontsize=LABEL_FS)
    ax.set_ylabel("Number of genes", fontsize=LABEL_FS)
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)   # headroom so legend clears bars
    # legend carries proportion %(n) per archetype (replaces the removed donut panel)
    counts = arc_df["archetype_name"].value_counts()
    total = len(arc_df)
    legend_handles = [
        mpatches.Patch(color=ARC_COLORS[a],
                       label=f"{DISPLAY_NAME[a]} — {100*counts.get(a, 0)/total:.1f}% (n={counts.get(a, 0)})")
        for a in ARC_ORDER]
    ax.legend(handles=legend_handles, fontsize=LEGEND_FS - 0.5, frameon=False,
              ncol=1, loc="upper right", borderaxespad=0.4)
    ax.tick_params(labelsize=TICK_FS)


def panel_B_scatter(ax, arc_df):
    arc_df = arc_df.copy()
    arc_df["gene"]     = arc_df["gene_id"].apply(parse_gene)
    arc_df["delta_T0"] = arc_df["mu_T0"] - arc_df["mu_HC"]
    arc_df["RI_plot"]  = arc_df["RI"].clip(-3, 4)

    for arch in ARC_ORDER:
        sub = arc_df[arc_df["archetype_name"] == arch]
        ax.scatter(sub["delta_T0"], sub["RI_plot"],
                   c=ARC_COLORS[arch], s=14, alpha=0.6,
                   linewidths=0, label=DISPLAY_NAME[arch], rasterized=True, zorder=2)

    for thresh in [1.5, 0.7, 0.2, -0.5]:
        ax.axhline(thresh, color="#444", lw=0.7, ls="--", alpha=0.5)

    texts = []
    # data-driven top Escalated — dark diamond marker + dark label box
    DD = "#1D3557"
    for _, row in arc_df[arc_df["gene"].isin(DATADRIVEN_GENES)].drop_duplicates("gene").iterrows():
        ax.scatter(row["delta_T0"], row["RI_plot"], s=60, facecolor="white",
                   edgecolor=DD, linewidths=1.6, zorder=4, marker="D")
        texts.append(ax.text(
            row["delta_T0"], row["RI_plot"], row["gene"],
            fontsize=ANNOT_FS, color=DD, fontweight="bold", zorder=5,
            bbox=dict(boxstyle="round,pad=0.2", fc="#EAF0F6", ec=DD, lw=0.8, alpha=0.95)))

    adjust_text(texts, ax=ax,
                arrowprops=dict(arrowstyle="-", color="#666", lw=0.7),
                expand=(1.6, 2.0), force_text=(0.5, 0.8))

    # legend for the highlighted data-driven top-Escalated genes
    leg = [mlines.Line2D([], [], marker="D", ls="", mfc="white", mec=DD,
                         mew=1.4, ms=8, label="Top Escalated (data-driven)")]
    ax.legend(handles=leg, fontsize=LEGEND_FS - 0.5, frameon=False, loc="lower right")

    ax.set_xlabel("Δ z-score at T0 (T0 − HC)", fontsize=LABEL_FS)
    ax.set_ylabel("Recovery Index (RI)", fontsize=LABEL_FS)
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")
    ax.tick_params(labelsize=TICK_FS)


def panel_C_donut(ax, sum_df):
    sizes  = [sum_df.loc[sum_df["archetype_name"] == a, "pct"].values[0] for a in ARC_ORDER]
    counts = [sum_df.loc[sum_df["archetype_name"] == a, "n"].values[0]   for a in ARC_ORDER]
    colors = [ARC_COLORS[a] for a in ARC_ORDER]

    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.5, edgecolor="white", linewidth=0.9),
    )
    # only the two large wedges get an inside label; small wedges go to legend
    for w, s, n in zip(wedges, sizes, counts):
        if s >= 8:
            angle = (w.theta1 + w.theta2) / 2
            x = 0.74 * np.cos(np.deg2rad(angle))
            y = 0.74 * np.sin(np.deg2rad(angle))
            ax.text(x, y, f"{s:.1f}%\n(n={n})", ha="center", va="center",
                    fontsize=ANNOT_FS, fontweight="bold", color="white")

    ax.text(0, 0, "263\ngenes", ha="center", va="center",
            fontsize=LABEL_FS + 1, fontweight="bold", color="#333")

    # legend carries the % (n) for every archetype → no cramped outside labels
    legend_handles = [
        mpatches.Patch(color=ARC_COLORS[a],
                       label=f"{DISPLAY_NAME[a]} — {s:.1f}% (n={n})")
        for a, s, n in zip(ARC_ORDER, sizes, counts)
    ]
    ax.legend(handles=legend_handles, fontsize=LEGEND_FS - 0.5, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, -0.26))
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")


def panel_D_trajectories(ax_container, arc_df):
    tp_cols = {"T0": "mu_T0", "T1": "mu_T1", "T2": "mu_T2",
               "T3": "mu_T3", "RC": "mu_RC"}
    tps  = list(tp_cols.keys())
    xpos = [0, 7, 14, 21, 65]

    for i, arch in enumerate(ARC_ORDER):
        ax = ax_container[i]
        sub = arc_df[arc_df["archetype_name"] == arch].copy()
        # orient each gene to its acute (T0) deviation direction so that up- and
        # down-regulated genes within one archetype don't cancel; all start +.
        sign = np.sign(sub["mu_T0"] - sub["mu_HC"]).replace(0, 1)
        dev  = {tp: (sub[tp_cols[tp]] - sub["mu_HC"]) * sign for tp in tps}
        vals = [dev[tp].mean() for tp in tps]
        sems = [dev[tp].sem() for tp in tps]

        col = ARC_COLORS[arch]
        ax.fill_between(xpos, [v - s for v, s in zip(vals, sems)],
                        [v + s for v, s in zip(vals, sems)], alpha=0.2, color=col)
        ax.plot(xpos, vals, "-o", color=col, ms=4, lw=1.8)
        ax.axhline(0, color="#888", lw=0.6, ls="--")

        ax.set_ylim(-0.75, 1.30)          # shared range → shapes directly comparable
        ax.set_title(f"{DISPLAY_NAME[arch]}\n(n={len(sub)})", fontsize=LABEL_FS + 2,
                     fontweight="bold", color=col, pad=3)
        ax.set_xticks(xpos)
        ax.set_xticklabels(["T0", "T1", "T2", "T3", "RC"], fontsize=TICK_FS - 1.5,
                           rotation=45)
        ax.tick_params(labelsize=TICK_FS - 1.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if i > 0:
            ax.set_yticklabels([])        # only leftmost keeps y tick labels

    ax_container[0].set_ylabel("Δ z-score, oriented\nto acute direction (HC = 0)",
                               fontsize=LABEL_FS)
    ax_container[2].set_xlabel("Timepoint", fontsize=LABEL_FS)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    arc_df = pd.read_csv(ARC_FILE, sep="\t")
    # clean-cohort file names the archetype column 'archetype'; scripts expect 'archetype_name'
    if "archetype_name" not in arc_df.columns and "archetype" in arc_df.columns:
        arc_df["archetype_name"] = arc_df["archetype"]
    arc_df = arc_df[arc_df["direction"].isin(["up", "down"])].copy()
    arc_df["gene"] = arc_df["gene_id"].apply(parse_gene)

    fig = plt.figure(figsize=(14, 10))
    # top row: A (RI histogram) + B (scatter); donut panel removed (redundant w/ A,
    # proportions now shown in A's legend)
    gs_top = GridSpec(1, 2, figure=fig,
                      left=0.07, right=0.97, top=0.94, bottom=0.56,
                      wspace=0.22)
    gs_bot = GridSpec(1, 5, figure=fig,
                      left=0.07, right=0.97, top=0.42, bottom=0.09,
                      wspace=0.40)

    ax_A = fig.add_subplot(gs_top[0, 0])
    ax_B = fig.add_subplot(gs_top[0, 1])
    mini_axes = [fig.add_subplot(gs_bot[0, i]) for i in range(5)]

    panel_A_histogram(ax_A, arc_df)
    panel_B_scatter(ax_B, arc_df)
    panel_D_trajectories(mini_axes, arc_df)

    for ax, lab in [(ax_A, "A"), (ax_B, "B")]:
        add_panel_label(ax, lab)

    # Panel C label + descriptive title above the mini-axes row (was D)
    fig.text(0.045, 0.475, "C", fontsize=PANEL_FS, fontweight="bold", va="bottom")

    out_png = f"{OUT}/fig4_archetype.png"
    out_pdf = f"{OUT}/fig4_archetype.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
