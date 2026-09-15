"""
RNA-seq cell-type deconvolution v2 (CIBERSORT-like NNLS).

Changes from v1:
  - Input: rna_combat_corrected_v2.tsv / sample_info_v2.tsv
    (K1 year-corrected → cross-batch HC ComBat-seq)
  - Output: rna_cell_fractions_v2.tsv, rna_deconv_stats_v2.tsv,
            rna_celltype_boxplot_v2.png, rna_celltype_heatmap_v2.png

Method identical to v1:
  CPM → ENSG.v_SYMBOL → SYMBOL → quantile-normalize → NNLS (LM22, 22 cell types)
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import nnls
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE      = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RNA_DIR   = os.path.join(os.path.dirname(BASE), "2.RNA_batch_correction", "data")
DATA      = os.path.join(BASE, "data")
FIGS      = os.path.join(BASE, "figures")
RESOURCES = os.path.join(BASE, "resources")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

LM22_PATH = os.path.join(RESOURCES, "LM22.txt")
PALETTE   = {"AC": "#E63946", "RC": "#F4A261", "HC": "#457B9D"}

if not os.path.isfile(LM22_PATH):
    print(f"ERROR: LM22 not found at {LM22_PATH}")
    sys.exit(1)

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading ComBat-seq v2 corrected counts...")
counts = pd.read_csv(os.path.join(RNA_DIR, "rna_combat_corrected_v2.tsv"),
                     sep="\t", index_col=0)
sample_info = pd.read_csv(os.path.join(RNA_DIR, "sample_info_v2.tsv"), sep="\t",
                          index_col="SampleID")
sample_info = sample_info.loc[counts.columns]
print(f"  Genes: {counts.shape[0]:,}, Samples: {counts.shape[1]:,}")
print(f"  Groups: {sample_info['Group'].value_counts().to_dict()}")

print("\nLoading LM22 signature matrix...")
lm22 = pd.read_csv(LM22_PATH, sep="\t", index_col=0)
print(f"  LM22: {lm22.shape[0]:,} genes × {lm22.shape[1]} cell types")

# ── 2. CPM + symbol conversion ────────────────────────────────────────────────
print("\nConverting counts to CPM...")
lib_sizes = counts.sum(axis=0)
cpm = counts.div(lib_sizes, axis=1) * 1e6

if cpm.index[0].count("_") >= 1 and cpm.index[0].startswith("ENSG"):
    cpm.index = cpm.index.str.split("_", n=1).str[1]
    cpm = cpm.groupby(cpm.index).mean()
    print(f"  ENSG.v_SYMBOL → SYMBOL: {len(cpm):,} unique symbols")

# ── 3. Gene overlap ───────────────────────────────────────────────────────────
common_genes = cpm.index.intersection(lm22.index)
print(f"LM22 genes in mixture: {len(common_genes):,} / {lm22.shape[0]:,}")

mix_sub = cpm.loc[common_genes]
sig_sub = lm22.loc[common_genes]

# ── 4. Quantile normalization ─────────────────────────────────────────────────
def quantile_normalize_sample(v):
    return rankdata(v, method="average") / len(v)

print("Quantile-normalizing samples...")
mix_norm = mix_sub.apply(quantile_normalize_sample, axis=0)
sig_norm = sig_sub.apply(quantile_normalize_sample, axis=0)

# ── 5. NNLS deconvolution ─────────────────────────────────────────────────────
print(f"\nRunning NNLS ({mix_norm.shape[1]} samples)...")
sig_mat    = sig_norm.values
cell_types = sig_norm.columns.tolist()
n_samples  = mix_norm.shape[1]
sample_ids = mix_norm.columns.tolist()

fractions = np.zeros((n_samples, len(cell_types)), dtype=np.float64)
rmse_list = np.zeros(n_samples, dtype=np.float64)

for i, sid in enumerate(sample_ids):
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_samples}", flush=True)
    b = mix_norm[sid].values
    x, _ = nnls(sig_mat, b)
    total = x.sum()
    fractions[i] = x / total if total > 0 else x
    rmse_list[i] = np.sqrt(np.mean((b - sig_mat @ x) ** 2))

print(f"  Done. Median RMSE: {np.median(rmse_list):.4f}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
frac_df = pd.DataFrame(fractions, index=sample_ids, columns=cell_types)
frac_df.index.name = "SampleID"
frac_df.to_csv(os.path.join(DATA, "rna_cell_fractions_v2.tsv"), sep="\t", float_format="%.6f")
print(f"\nSaved → data/rna_cell_fractions_v2.tsv")

pd.DataFrame({"SampleID": sample_ids, "RMSE": rmse_list, "n_genes": len(common_genes)}).to_csv(
    os.path.join(DATA, "rna_deconv_stats_v2.tsv"), sep="\t", index=False)

# ── 7. Boxplot ────────────────────────────────────────────────────────────────
print("Generating boxplot...")
groups      = sample_info.loc[sample_ids, "Group"].values
group_order = ["AC", "RC", "HC"]
n_types = len(cell_types)
n_cols  = 4
n_rows  = int(np.ceil(n_types / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axes = axes.flatten()

for k, ct in enumerate(cell_types):
    ax = axes[k]
    bp = ax.boxplot([frac_df.loc[groups == g, ct].values for g in group_order],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=1.5),
                    flierprops=dict(marker=".", ms=3, alpha=0.5))
    for patch, grp in zip(bp["boxes"], group_order):
        patch.set_facecolor(PALETTE[grp])
    ax.set_xticks(range(1, len(group_order) + 1))
    ax.set_xticklabels(group_order, fontsize=8)
    ax.set_title(ct, fontsize=8, pad=2)
    ax.set_ylabel("Proportion", fontsize=7)
    ax.tick_params(axis="y", labelsize=7)

for k in range(n_types, len(axes)):
    axes[k].set_visible(False)

handles = [mpatches.Patch(color=PALETTE[g], label=g) for g in group_order]
fig.legend(handles=handles, loc="lower right", ncol=1, fontsize=9, frameon=False)
fig.suptitle("RNA Cell Type Fractions v2 (NNLS / LM22)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "rna_celltype_boxplot_v2.png"), dpi=150, bbox_inches="tight")
print(f"Saved → figures/rna_celltype_boxplot_v2.png")

# ── 8. Heatmap ────────────────────────────────────────────────────────────────
print("Generating heatmap...")
sort_idx     = np.argsort(groups)
frac_sorted  = frac_df.iloc[sort_idx].T
group_sorted = groups[sort_idx]

def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

fig, ax = plt.subplots(figsize=(min(n_samples * 0.02 + 2, 20), n_types * 0.4 + 2))
im = ax.imshow(frac_sorted.values, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=frac_sorted.values.max())
ax.set_yticks(range(n_types))
ax.set_yticklabels(cell_types, fontsize=7)
ax.set_xlabel("Samples (sorted by group)", fontsize=9)
ax.set_title("Cell-type fractions v2 (RNA NNLS / LM22)", fontsize=10)

pos = ax.get_position()
strip_h = 0.02
ax_s = fig.add_axes([pos.x0, pos.y1 + strip_h * 0.3, pos.width, strip_h])
ax_s.imshow(np.array([[hex_to_rgb(PALETTE[g]) for g in group_sorted]]), aspect="auto")
ax_s.axis("off")

plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Proportion")
plt.savefig(os.path.join(FIGS, "rna_celltype_heatmap_v2.png"), dpi=150, bbox_inches="tight")
print(f"Saved → figures/rna_celltype_heatmap_v2.png")

# ── 9. Summary ────────────────────────────────────────────────────────────────
print(f"\nSummary (v2):")
print(f"  Genes used  : {len(common_genes):,}")
print(f"  Samples     : {n_samples:,}")
print(f"  Cell types  : {len(cell_types)}")
print(f"  Median RMSE : {np.median(rmse_list):.4f}")
print("\nMean fractions by group:")
for g in group_order:
    mask = groups == g
    row  = frac_df.loc[mask].mean()
    print(f"  {g}: " + "  ".join(f"{ct[:12]}={row[ct]:.3f}" for ct in cell_types[:6]) + " ...")
print("Done.")
