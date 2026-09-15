"""
Fig 1 — Study Overview
Panels:
  A: Study design schematic — tri-cohort timeline (matplotlib patches)
  B: RNA PCA of all samples coloured by Group/Timepoint
  C: Sample size summary per cohort × timepoint (bar chart)
  D: Age + Sex distribution by Group (COVID vs HC)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE = "PROJECT_ROOT/research/LongCOVID/Results"
NEW  = BASE + "_updated_260714"
OUT  = f"{NEW}/manuscript/figures"

EXPR_FILE = f"{BASE}/2.Data_preparation/2.RNA_batch_correction/data/rna_combat_corrected_v2.tsv"
INFO_FILE = f"{BASE}/2.Data_preparation/2.RNA_batch_correction/data/sample_info_v2.tsv"
META_COVID = f"{BASE}/1.Metadata/COVID-19_sample_metadata.tsv"
META_HC    = f"{BASE}/1.Metadata/HC_sample_metadata.tsv"

# clean cohort: exclude the 20 RNA-excluded RC samples (-L1 + V1<28d) so Fig 1
# reflects the analysis cohort (RC 158 -> 138)
import pandas as _pd
EXCLUDED = set(_pd.read_csv(f"{NEW}/3.Discovery/1.RNA_DE/data/excluded_samples.tsv",
                            sep="\t")["excluded_SampleID"])

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 8,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Colour mapping
TP_COLORS = {
    "HC":  "#AAAAAA",
    "T0":  "#E63946",
    "T1":  "#F4845F",
    "T2":  "#F4A261",
    "T3":  "#E9C46A",
    "RC":  "#457B9D",
    "AC":  "#E63946",   # fallback for group-only
}


# ── Panel A — Study design schematic ──────────────────────────────────────
def panel_A_design(ax):
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x, y, w, h, fc, ec, label, sublabel="", fs=8.5, lc="white"):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.08",
            facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2 + 0.14, label,
                ha="center", va="center", fontsize=fs,
                fontweight="bold", color=lc, zorder=3)
        if sublabel:
            ax.text(x + w/2, y + h/2 - 0.22, sublabel,
                    ha="center", va="center", fontsize=6.5, color=lc, zorder=3)

    def arr(x1, y1, x2, y2, col="#555", style="-|>"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=col, lw=1.2),
                    zorder=3)

    # --- G1 Acute cohort (row 3)
    g1_y = 5.7
    ax.text(0.2, g1_y + 0.9, "G1: Acute COVID-19", fontsize=8.5,
            fontweight="bold", color="#E63946")
    ax.text(0.2, g1_y + 0.55, "n ≈ 50 subjects", fontsize=7, color="#555")
    tps = [("T0\nDay 0", 1.5, "#E63946"), ("T1\nDay 7", 3.2, "#F4845F"),
           ("T2\nDay 14", 4.9, "#F4A261"), ("T3\nDay 21", 6.6, "#E9C46A")]
    for lbl, x, col in tps:
        box(x, g1_y, 1.3, 0.9, col, col, lbl, fs=7.5)
    # arrows between timepoints
    for i in range(len(tps) - 1):
        arr(tps[i][1] + 1.3, g1_y + 0.45, tps[i+1][1], g1_y + 0.45)

    # Dual-gate brackets
    ax.annotate("", xy=(1.5, g1_y - 0.25), xytext=(2.8, g1_y - 0.25),
                arrowprops=dict(arrowstyle="-", color="#E63946", lw=2.0))
    ax.text(2.15, g1_y - 0.55, "Gate 1", ha="center", fontsize=7,
            color="#E63946", fontweight="bold")

    # --- G2 Recovered cohort (row 2)
    g2_y = 3.8
    ax.text(0.2, g2_y + 0.9, "G2: Recovered COVID-19", fontsize=8.5,
            fontweight="bold", color="#457B9D")
    ax.text(0.2, g2_y + 0.55, "n = 138 subjects  (RC, ≥4 wk convalescent)", fontsize=7, color="#555")
    box(8.2, g2_y, 1.5, 0.9, "#457B9D", "#457B9D", "RC\n~Day 60", fs=7.5)
    ax.annotate("", xy=(8.2, g2_y - 0.25), xytext=(9.7, g2_y - 0.25),
                arrowprops=dict(arrowstyle="-", color="#457B9D", lw=2.0))
    ax.text(8.95, g2_y - 0.55, "Gate 2", ha="center", fontsize=7,
            color="#457B9D", fontweight="bold")

    # --- G3 HC cohort (row 1)
    g3_y = 2.0
    ax.text(0.2, g3_y + 0.9, "G3: Healthy Controls (KU10K)", fontsize=8.5,
            fontweight="bold", color="#555")
    ax.text(0.2, g3_y + 0.55, "n ≈ 657 subjects", fontsize=7, color="#555")
    box(8.2, g3_y, 1.5, 0.9, "#888888", "#666", "HC\n(anchor)", fs=7.5)

    # Persistent genes box (intersection annotation)
    box(9.9, 5.2, 2.0, 1.2, "#E6394620", "#E63946",
        "263\nPersistent\ngenes", fs=7, lc="#E63946")
    ax.text(9.95, 4.85, "Gate1 ∩ Gate2", fontsize=6.5, color="#E63946",
            style="italic")
    arr(9.7, g2_y + 0.45, 9.9, 5.5, "#457B9D")
    arr(7.9, g1_y + 0.45, 9.9, 6.0, "#E63946")

    # Multi-omics labels
    ax.text(11.5, 5.0, "RNA-seq\n+\nBisulfite-seq", ha="center",
            fontsize=7.5, color="#444", style="italic",
            bbox=dict(boxstyle="round,pad=0.3", fc="#F8F8F8", ec="#CCC", lw=0.6))

    ax.set_title("A    Tri-cohort dual-gate study design",
                 fontsize=8.5, fontweight="bold", pad=4, loc="left")


# ── Panel B — PCA ─────────────────────────────────────────────────────────
def panel_B_pca(ax):
    print("  Loading expression matrix for PCA (may take ~30s)...")
    expr = pd.read_csv(EXPR_FILE, sep="\t", index_col=0)
    info = pd.read_csv(INFO_FILE, sep="\t", index_col=0)

    # Match samples (drop clean-cohort-excluded RC: -L1 + V1<28d)
    common = [c for c in expr.columns.intersection(info.index) if c not in EXCLUDED]
    expr = expr[common]
    info = info.loc[common]

    # PCA on top 5000 most variable genes (faster, same structure)
    variances = expr.var(axis=1)
    top_genes = variances.nlargest(5000).index
    sub_expr  = expr.loc[top_genes].T.values  # shape: (n_samples, 5000)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sub_expr)
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)

    df = pd.DataFrame({"PC1": coords[:, 0], "PC2": coords[:, 1],
                       "Group": info["Group"].values},
                      index=common)

    # Assign timepoint-level colour
    # Group in info_v2 is 'AC' or 'HC' or 'RC'
    # For AC samples, merge COVID metadata to get timepoint
    covid_meta = pd.read_csv(META_COVID, sep="\t", index_col=0)
    df["TP"] = df["Group"].map({"HC": "HC", "RC": "RC"}).fillna("AC")
    ac_tp = covid_meta["Timepoint"]
    for sid in df[df["Group"] == "AC"].index:
        if sid in ac_tp.index:
            df.loc[sid, "TP"] = ac_tp[sid]

    # Plot by TP
    tp_order = ["HC", "T0", "T1", "T2", "T3", "RC"]
    for tp in tp_order:
        sub = df[df["TP"] == tp]
        if sub.empty:
            continue
        ax.scatter(sub["PC1"], sub["PC2"], c=TP_COLORS.get(tp, "#BBB"),
                   s=8, alpha=0.55, linewidths=0, rasterized=True,
                   label=tp, zorder=2)

    var1 = pca.explained_variance_ratio_[0] * 100
    var2 = pca.explained_variance_ratio_[1] * 100
    ax.set_xlabel(f"PC1 ({var1:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC2 ({var2:.1f}%)", fontsize=8)
    ax.set_title("B    RNA-seq PCA (all samples, top 5K variable genes)",
                 fontsize=8.5, fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=6.5, frameon=False, markerscale=1.5, loc="best")
    ax.tick_params(labelsize=7)


# ── Panel C — Sample size per group/timepoint ──────────────────────────────
def panel_C_sample_sizes(ax):
    covid_meta = pd.read_csv(META_COVID, sep="\t")
    covid_meta = covid_meta[(covid_meta["Has_RNA"] == "Y") & (~covid_meta["SampleID"].isin(EXCLUDED))]
    hc_meta    = pd.read_csv(META_HC, sep="\t")
    hc_meta    = hc_meta[hc_meta["Has_RNA"] == "Y"]

    # Count per Timepoint
    tp_counts = covid_meta["Timepoint"].value_counts()
    tp_order = ["T0", "T1", "T2", "T3", "RC"]
    valid_tps = [tp for tp in tp_order if tp in tp_counts.index]
    counts = [tp_counts[tp] for tp in valid_tps] + [len(hc_meta)]
    labels = valid_tps + ["HC"]
    colors = [TP_COLORS[tp] for tp in labels]

    x = np.arange(len(labels))
    bars = ax.bar(x, counts, color=colors, alpha=0.85, edgecolor="white")

    for xi, n in zip(x, counts):
        ax.text(xi, n + 1, str(n), ha="center", va="bottom",
                fontsize=7, fontweight="bold", color="#333")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Number of samples (RNA-seq)", fontsize=8)
    ax.set_title("C    Sample sizes per cohort/timepoint",
                 fontsize=8.5, fontweight="bold", pad=4, loc="left")
    ax.tick_params(labelsize=7)

    # Brace annotations for cohort groups
    y_brace = max(counts) * 1.12
    ax.annotate("", xy=(3.4, y_brace), xytext=(-0.4, y_brace),
                arrowprops=dict(arrowstyle="-", color="#E63946", lw=1.5))
    ax.text(1.5, y_brace + 2, "G1 Acute", ha="center", fontsize=7,
            color="#E63946", fontweight="bold")
    ax.annotate("", xy=(4.4, y_brace), xytext=(3.6, y_brace),
                arrowprops=dict(arrowstyle="-", color="#457B9D", lw=1.5))
    ax.text(4.0, y_brace + 2, "G2", ha="center", fontsize=7,
            color="#457B9D", fontweight="bold")
    ax.annotate("", xy=(5.4, y_brace), xytext=(4.6, y_brace),
                arrowprops=dict(arrowstyle="-", color="#888", lw=1.5))
    ax.text(5.0, y_brace + 2, "G3", ha="center", fontsize=7,
            color="#888", fontweight="bold")


# ── Panel D — Age & Sex distribution ──────────────────────────────────────
def panel_D_demographics(ax):
    covid_meta = pd.read_csv(META_COVID, sep="\t")
    covid_meta = covid_meta[~covid_meta["SampleID"].isin(EXCLUDED)]   # clean cohort
    hc_meta    = pd.read_csv(META_HC,    sep="\t")

    # Use subject-level (T0 for COVID, or first available)
    covid_subj = covid_meta[covid_meta["Timepoint"].isin(["T0", "RC"])].drop_duplicates("SubjectID")
    hc_subj    = hc_meta.copy()

    groups = {"HC": hc_subj, "COVID": covid_subj}
    x      = [0, 1]
    labels = ["HC", "COVID"]
    colors_box = ["#AAAAAA", "#E63946"]

    # Age boxplot
    age_data = [pd.to_numeric(hc_subj["Age"], errors="coerce").dropna().values,
                pd.to_numeric(covid_subj["Age"], errors="coerce").dropna().values]
    bp = ax.boxplot(age_data, positions=x, widths=0.4, patch_artist=True,
                    showfliers=False, medianprops=dict(color="white", lw=2))
    for patch, col in zip(bp["boxes"], colors_box):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)
    for whisker, col in zip(bp["whiskers"], [colors_box[0]]*2 + [colors_box[1]]*2):
        whisker.set_color(col)
    for cap, col in zip(bp["caps"], [colors_box[0]]*2 + [colors_box[1]]*2):
        cap.set_color(col)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Age (years)", fontsize=8)
    ax.set_title("D    Age distribution (HC vs COVID-19)",
                 fontsize=8.5, fontweight="bold", pad=4, loc="left")
    ax.tick_params(labelsize=7)

    # Sex proportion as secondary inset
    ax2 = ax.inset_axes([0.65, 0.55, 0.32, 0.38])
    hc_sex  = hc_subj["Sex"].value_counts(normalize=True)
    cov_sex = covid_subj["Sex"].value_counts(normalize=True)
    f_pct = [hc_sex.get("F", 0), cov_sex.get("F", 0)]
    m_pct = [1 - f for f in f_pct]
    ax2.bar([0, 1], f_pct, color="#F06292", alpha=0.8, width=0.5, label="F")
    ax2.bar([0, 1], m_pct, bottom=f_pct, color="#64B5F6", alpha=0.8, width=0.5, label="M")
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["HC", "COV"], fontsize=5.5)
    ax2.set_ylabel("Sex %", fontsize=5.5)
    ax2.set_yticks([0, 0.5, 1.0])
    ax2.set_yticklabels(["0", "50", "100"], fontsize=5)
    ax2.legend(fontsize=5, frameon=False, loc="upper right")
    ax2.tick_params(labelsize=5)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(14, 11))
    gs = GridSpec(2, 2, figure=fig,
                  left=0.07, right=0.97, top=0.93, bottom=0.09,
                  hspace=0.52, wspace=0.38)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, 0])
    ax_D = fig.add_subplot(gs[1, 1])

    panel_A_design(ax_A)

    panel_B_pca(ax_B)
    panel_C_sample_sizes(ax_C)
    panel_D_demographics(ax_D)

    fig.suptitle(
        "Figure 1  |  Tri-cohort whole-blood multi-omics study design and "
        "sample overview",
        fontsize=10, fontweight="bold", x=0.5, y=0.98)

    out_png = f"{OUT}/fig1_overview.png"
    out_pdf = f"{OUT}/fig1_overview.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    plt.close()


if __name__ == "__main__":
    main()
