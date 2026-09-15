"""
Fig 2 — Discovery: persistent transcriptomics and predominantly delayed methylomic shifts
Panels:
  A: Volcano — Acute contrast (T0 vs HC)        [was "Gate 1"]
  B: Volcano — Convalescent contrast (RC vs HC) [was "Gate 2"], coloured by category
  C: Stacked bar — RNA vs Methylation category counts
  D: Pathway enrichment (GSEA, Convalescent contrast) — two-sided NES barplot
  E: Delayed CpG promoter context (simplified donut)

Terminology: the two differential-expression comparisons are named
  "Acute contrast"        = AC_T0 vs HC  (genes changed during acute infection)
  "Convalescent contrast" = RC vs HC      (genes still changed after recovery)
  Persistent = significant in BOTH ;  Delayed = significant only in Convalescent.

Both contrasts: FDR < 0.05 & |log2FC| > 1.0
  Acute        : 1,369 genes (1,156 up / 213 down)
  Convalescent :   345 genes (226 up / 119 down)
  Persistent   :   263 genes (182 up / 81 down)
  Delayed      :    82 genes
  Acute-specific   : 1,106 genes
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

# ── paths (CLEAN COHORT v3; outputs -> Results_updated_260714) ──────────────
BASE = "PROJECT_ROOT/research/LongCOVID/Results"
NEW  = BASE + "_updated_260714"
OUT  = f"{NEW}/manuscript/figures"

RNA_G1   = f"{BASE}/3.Discovery/1.RNA_DE_ver2/data/gate1_ac_t0_vs_hc.tsv"          # Acute gate unchanged
RNA_G2   = f"{NEW}/3.Discovery/1.RNA_DE/data/gate2_rc_vs_hc_clean.tsv"             # clean convalescent gate
RNA_DUAL = f"{NEW}/3.Discovery/1.RNA_DE/data/rna_markers_canonical_filtered.tsv"   # canonical, IG/pseudo removed (2026-07-30)
METH_DG  = f"{NEW}/3.Discovery/2.Methyl_DM/data/methyl_markers_cleancohort_clean.tsv"
GSEA_ALL = f"{BASE}/3.Discovery/1.RNA_DE_ver2/4.Functional/data/gsea/gsea_all_results.tsv"  # GSEA not re-run (stable)

# ── analysis constants (verified against data) ─────────────────────────────
FC_THR = 1.0      # |log2FC| threshold defining both contrasts
P_THR  = 0.05     # adjusted-p threshold
ACUTE_UP, ACUTE_DN = 995, 166           # Acute contrast counts (canonical, IG/pseudo removed)
PERSISTENT_N, DELAYED_N, ACUTE_ONLY_N = 208, 61, 953    # canonical, IG/pseudo removed (2026-07-30)
METH_PERSIST_N, METH_DELAYED_N = 4, 303                 # clean methyl

# ── style ──────────────────────────────────────────────────────────────────
PALETTE = {
    "persistent":  "#E63946",   # red
    "delayed":     "#F4A261",   # orange
    "acute_only":  "#A8DADC",   # light blue
    "not_sig":     "#CCCCCC",   # grey
    "up":          "#E63946",
    "down":        "#457B9D",
}

# ── font sizes (enlarged for readability) ──────────────────────────────────
TITLE_FS  = 12      # panel titles
LABEL_FS  = 11      # axis labels
TICK_FS   = 9.5     # tick labels
LEGEND_FS = 9.5     # legends
ANNOT_FS  = 9       # in-panel annotations
PANEL_FS  = 16      # standalone panel labels (A/B/C…)

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

XLAB_FC = r"$\log_2$ fold change"
YLAB_P  = r"$-\log_{10}$ adjusted $p$"


# ── helpers ────────────────────────────────────────────────────────────────
def parse_gene_name(gene_id):
    """ENSG123.4_GENE → GENE"""
    parts = str(gene_id).split("_", 1)
    return parts[1] if len(parts) > 1 else gene_id


def volcano_panel(ax, df, fc_col, p_col, cat_col, title,
                  xlim=None, offscale_note=None):
    df = df.copy()
    df["gene"]  = df.index.map(parse_gene_name)
    df["-logP"] = -np.log10(df[p_col].clip(lower=1e-300))
    df["logFC"] = df[fc_col]

    if cat_col in df.columns:
        color_map = {
            "persistent": PALETTE["persistent"],
            "delayed":    PALETTE["delayed"],
            "acute_only": PALETTE["acute_only"],
            "not_sig":    PALETTE["not_sig"],
        }
        colors = df[cat_col].map(color_map).fillna(PALETTE["not_sig"])
    else:
        sig = (df[p_col] < P_THR) & (df[fc_col].abs() > FC_THR)
        up  = sig & (df[fc_col] > 0)
        dn  = sig & (df[fc_col] < 0)
        colors = pd.Series(PALETTE["not_sig"], index=df.index)
        colors[up] = PALETTE["up"]
        colors[dn] = PALETTE["down"]

    # clip x for display so globin outliers don't compress the bulk
    x_plot = df["logFC"]
    if xlim is not None:
        x_plot = x_plot.clip(xlim[0], xlim[1])

    ax.scatter(x_plot, df["-logP"], c=colors, s=6, alpha=0.6,
               linewidths=0, rasterized=True)

    ax.axvline( FC_THR, color="#888", lw=0.6, ls="--")
    ax.axvline(-FC_THR, color="#888", lw=0.6, ls="--")
    ax.axhline(-np.log10(P_THR), color="#888", lw=0.6, ls="--")

    if xlim is not None:
        ax.set_xlim(xlim)
    if offscale_note:
        ax.annotate(offscale_note, xy=(0.985, 0.55), xycoords="axes fraction",
                    ha="right", va="center", fontsize=ANNOT_FS-1.5, color="#555",
                    style="italic",
                    arrowprops=None)

    ax.set_xlabel(XLAB_FC, fontsize=LABEL_FS)
    ax.set_ylabel(YLAB_P, fontsize=LABEL_FS)
    ax.set_title("", fontsize=TITLE_FS, fontweight="bold", pad=4)
    ax.tick_params(labelsize=TICK_FS)


def bar_comparison_panel(ax):
    """Panel C — RNA vs Methylation category counts (stacked)."""
    rna_cats  = {"Persistent": PERSISTENT_N,      "Delayed": DELAYED_N,      "Acute-specific": ACUTE_ONLY_N}
    meth_cats = {"Persistent": METH_PERSIST_N,    "Delayed": METH_DELAYED_N, "Acute-specific": 0}

    cats   = ["Persistent", "Delayed", "Acute-specific"]
    colors = [PALETTE["persistent"], PALETTE["delayed"], PALETTE["acute_only"]]
    x      = np.array([0, 1])
    labels = ["Transcriptome\n(RNA-seq)", "Methylome\n(Bisulfite-seq)"]

    bottom = np.zeros(2)
    for cat, col in zip(cats, colors):
        vals = [rna_cats[cat], meth_cats[cat]]
        ax.bar(x, vals, bottom=bottom, color=col, width=0.5, label=cat,
               edgecolor="white", linewidth=0.5)
        bottom += np.array(vals)

    ax.text(0, PERSISTENT_N / 2, f"{PERSISTENT_N}", ha="center", va="center", fontsize=ANNOT_FS,
            fontweight="bold", color="white")
    ax.text(0, PERSISTENT_N + DELAYED_N / 2, f"{DELAYED_N}", ha="center", va="center",
            fontsize=ANNOT_FS, color="white")
    ax.text(1, METH_PERSIST_N + METH_DELAYED_N / 2, f"{METH_DELAYED_N}\n(99%)", ha="center", va="center",
            fontsize=ANNOT_FS, fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=TICK_FS)
    ax.set_ylabel("Number of markers", fontsize=LABEL_FS)
    ax.set_title("", fontsize=TITLE_FS,
                 fontweight="bold", pad=4)
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="upper right")
    ax.tick_params(labelsize=TICK_FS)


# Curated pathways for Panel D (display name, GSEA Term substring, direction)
GSEA_DOWN = [
    ("Histone lysine methylation",   "Histone Lysine Methylation"),
    ("Histone H3-K4 methylation",    "Histone H3-K4 Methylation"),
    ("HDMs demethylate histones",    "HDMs Demethylate Histones"),
    ("Heterochromatin organization", "Heterochromatin Organization"),
    ("SUMOylation of target proteins", "SUMO E3 Ligases SUMOylate Target Proteins"),
]
GSEA_UP = [
    ("Antigen cross-presentation",   "Antigen processing-Cross Presentation"),
    ("Interferon-alpha response",    "Interferon Alpha Response"),
    ("Antimicrobial humoral response", "Antimicrobial Humoral Immune Response"),
]


def _lookup(g2, substring):
    hit = g2[g2["Term"].str.contains(substring, case=False, na=False, regex=False)]
    if hit.empty:
        return None
    return hit.iloc[0]


def gsea_panel(ax, gsea_df):
    """Panel D — two-sided NES barplot for the Convalescent contrast."""
    g2 = gsea_df[gsea_df["gate"] == "gate2_RC"].copy()

    rows = []   # (display, NES, q, nomp, is_up)
    for disp, sub in GSEA_DOWN:
        r = _lookup(g2, sub)
        if r is not None:
            rows.append((disp, r["NES"], r["FDR q-val"], r["NOM p-val"], False))
    for disp, sub in GSEA_UP:
        r = _lookup(g2, sub)
        if r is not None:
            rows.append((disp, r["NES"], r["FDR q-val"], r["NOM p-val"], True))

    # order: most-negative at bottom → most-positive at top
    rows.sort(key=lambda t: t[1])
    y = np.arange(len(rows))

    for i, (disp, nes, q, nomp, is_up) in enumerate(rows):
        col = PALETTE["up"] if is_up else PALETTE["down"]
        # FDR-significant = solid; nominal-only = lighter + hatched
        fdr_sig = q < 0.05
        ax.barh(i, nes, height=0.55, color=col,
                alpha=1.0 if fdr_sig else 0.45,
                hatch=None if fdr_sig else "////",
                edgecolor=col, linewidth=0.6)
        # significance annotation at bar end
        if fdr_sig:
            txt = f"q={q:.3f}"
        else:
            txt = f"nom. p={nomp:.3f}"
        xpos = nes + (0.06 if nes > 0 else -0.06)
        ax.text(xpos, i, txt, va="center",
                ha="left" if nes > 0 else "right",
                fontsize=ANNOT_FS-1.5, color="#555")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=TICK_FS)
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("Normalized enrichment score (NES)", fontsize=LABEL_FS)
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")
    ax.tick_params(labelsize=TICK_FS)
    ax.set_xlim(-3.1, 2.6)
    n = len(rows)
    ax.set_ylim(-0.8, n - 1 + 1.5)   # top headroom band for direction labels

    # direction guides in the top headroom (left = suppressed, right = induced)
    yg = n - 1 + 0.75
    ax.text(-3.0, yg, "← Suppressed (chromatin / epigenetic)",
            fontsize=ANNOT_FS, color=PALETTE["down"], va="center", ha="left",
            fontweight="bold")
    ax.text(2.5, yg, "Induced (immune / antigen) →",
            fontsize=ANNOT_FS, color=PALETTE["up"], va="center", ha="right",
            fontweight="bold")

    legend_handles = [
        mpatches.Patch(facecolor="#888", edgecolor="#888", label="FDR < 0.05"),
        mpatches.Patch(facecolor="white", edgecolor="#888", hatch="////",
                       label="nominal p < 0.05"),
    ]
    ax.legend(handles=legend_handles, fontsize=LEGEND_FS, frameon=False,
              loc="lower right")


def promoter_donut_panel(ax):
    """Panel E — Delayed CpG: promoter-hyper + other-hyper vs hypo (clean cohort)."""
    total     = METH_DELAYED_N           # 303 delayed
    hypo      = 32                        # delayed hypo (clean)
    hyper     = total - hypo              # 271 hyper
    prom_hyper = 155                      # annotated promoter-hyper (37 unannotated pending)
    other_hyper = hyper - prom_hyper      # 116

    # single ring: promoter-hyper vs other-hyper vs hypo
    sizes  = [prom_hyper, other_hyper, hypo]
    colors = [PALETTE["persistent"], "#F6C9B0", PALETTE["down"]]
    wedges, _ = ax.pie(
        sizes, colors=colors, startangle=90,
        wedgeprops=dict(width=0.42, edgecolor="white", linewidth=1.0),
    )
    ax.text(0, 0, f"{total}\ndelayed\nCpGs", ha="center", va="center",
            fontsize=LABEL_FS, fontweight="bold", color="#333")

    legend_handles = [
        mpatches.Patch(color=PALETTE["persistent"], label=f"Promoter, hyper  ({prom_hyper})"),
        mpatches.Patch(color="#F6C9B0",             label=f"Other, hyper  ({other_hyper})"),
        mpatches.Patch(color=PALETTE["down"],       label=f"Hypo  ({hypo})"),
    ]
    ax.legend(handles=legend_handles, fontsize=LEGEND_FS, frameon=False,
              loc="lower center", bbox_to_anchor=(0.5, -0.20))
    ax.set_title("", fontsize=TITLE_FS,
                 fontweight="bold", pad=4)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    g1   = pd.read_csv(RNA_G1, sep="\t", index_col=0)
    g2   = pd.read_csv(RNA_G2, sep="\t", index_col=0)
    dual = pd.read_csv(RNA_DUAL, sep="\t").set_index("gene_id")  # filtered file: key on gene_id, not symbol
    meth = pd.read_csv(METH_DG, sep="\t", index_col=0)
    gsea = pd.read_csv(GSEA_ALL, sep="\t")

    # only kept canonical genes carry a category; removed IG/pseudo/duplicate
    # points fall through to grey (not_sig) in the convalescent volcano
    g2 = g2.join(dual[["category"]], how="left")

    fig = plt.figure(figsize=(14, 9))
    gs = GridSpec(2, 3, figure=fig,
                  left=0.07, right=0.97, top=0.95, bottom=0.10,
                  hspace=0.55, wspace=0.40)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[0, 2])
    ax_D = fig.add_subplot(gs[1, 0:2])
    ax_E = fig.add_subplot(gs[1, 2])

    # Panel A — Acute contrast volcano (clip globin off-scale)
    volcano_panel(ax_A, g1, "log2FoldChange", "adj_pvalue", None,
                  "Acute contrast (AC_T0 vs HC)",
                  xlim=(-6, 9))

    # Panel B — Convalescent contrast volcano (NO gene labels, clean legend)
    volcano_panel(ax_B, g2, "log2FoldChange", "adj_pvalue", "category",
                  "Convalescent contrast (RC vs HC)",
                  xlim=(-3.5, 3.5))
    legend_patches = [
        mpatches.Patch(color=PALETTE["persistent"], label=f"Persistent (n={PERSISTENT_N})"),
        mpatches.Patch(color=PALETTE["delayed"],    label=f"Delayed (n={DELAYED_N})"),
        mpatches.Patch(color=PALETTE["acute_only"], label=f"Acute-specific / resolved (n={ACUTE_ONLY_N:,})"),
        mpatches.Patch(color=PALETTE["not_sig"],    label="Not significant"),
    ]
    ax_B.legend(handles=legend_patches, fontsize=LEGEND_FS, frameon=False,
                loc="upper left")

    bar_comparison_panel(ax_C)
    gsea_panel(ax_D, gsea)
    promoter_donut_panel(ax_E)

    for ax, lbl in zip([ax_A, ax_B, ax_C, ax_D, ax_E],
                       ["A", "B", "C", "D", "E"]):
        ax.annotate(lbl, xy=(0, 1), xycoords="axes fraction",
                    xytext=(-34, 14), textcoords="offset points",
                    fontsize=PANEL_FS, fontweight="bold", va="bottom", ha="left")

    out_png = f"{OUT}/fig2_discovery.png"
    out_pdf = f"{OUT}/fig2_discovery.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
