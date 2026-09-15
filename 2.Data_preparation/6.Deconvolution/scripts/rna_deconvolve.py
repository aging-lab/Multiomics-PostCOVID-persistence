"""
RNA-seq cell-type deconvolution (CIBERSORT-like normalized NNLS).

Input:
  rna_combat_corrected.tsv   — ComBat-seq corrected counts (genes × samples)
  sample_info.tsv            — SampleID, Group, FinalBatch
  resources/LM22.txt         — CIBERSORT LM22 signature matrix (genes × 22 cell types)
                               Download from: https://cibersortx.stanford.edu  (requires free account)
                               Place at: 6.Deconvolution/resources/LM22.txt

Method:
  1. Convert counts to CPM (per sample).
  2. Subset to LM22 genes present in both mixture and signature.
  3. Quantile-normalize each sample mixture to uniform [0,1] (CIBERSORT preprocessing).
  4. NNLS regression per sample: fractions = argmin ||Sig * x - mixture||, x >= 0.
  5. Normalize fractions to sum to 1.

Output:
  data/rna_cell_fractions.tsv         — 22 cell types × 1,108 samples (proportions)
  data/rna_deconv_stats.tsv           — per-sample RMSE, n_genes_used
  figures/rna_celltype_boxplot.png    — per-group distribution of each cell type
  figures/rna_celltype_heatmap.png    — sample × cell-type fraction heatmap

Note:
  This does NOT implement CIBERSORT's permutation p-value or SVR correction.
  Proportions are relative estimates suitable for group comparison (Mann-Whitney U).
  For absolute counts, use CIBERSORT-ABS via the web portal.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.optimize import nnls
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import rankdata

BASE       = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RNA_DIR    = os.path.join(os.path.dirname(BASE), "2.RNA_batch_correction", "data")
DATA       = os.path.join(BASE, "data")
FIGS       = os.path.join(BASE, "figures")
RESOURCES  = os.path.join(BASE, "resources")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

LM22_PATH  = os.path.join(RESOURCES, "LM22.txt")
PALETTE    = {"AC": "#E63946", "RC": "#F4A261", "HC": "#457B9D"}

# ── 0. Check reference ─────────────────────────────────────────────────────────
if not os.path.isfile(LM22_PATH):
    print("ERROR: LM22 signature matrix not found.")
    print(f"  Expected: {LM22_PATH}")
    print("  Download from https://cibersortx.stanford.edu (free account required)")
    print("  File name: LM22.txt (tab-separated, gene × cell-type matrix)")
    sys.exit(1)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading ComBat-seq corrected counts...")
counts = pd.read_csv(os.path.join(RNA_DIR, "rna_combat_corrected.tsv"),
                     sep="\t", index_col=0)
sample_info = pd.read_csv(os.path.join(RNA_DIR, "sample_info.tsv"), sep="\t",
                          index_col="SampleID")
sample_info = sample_info.loc[counts.columns]
print(f"  Genes: {counts.shape[0]:,}, Samples: {counts.shape[1]:,}")
print(f"  Groups: {sample_info['Group'].value_counts().to_dict()}")

print("\nLoading LM22 signature matrix...")
lm22 = pd.read_csv(LM22_PATH, sep="\t", index_col=0)
print(f"  LM22: {lm22.shape[0]:,} genes × {lm22.shape[1]} cell types")
print(f"  Cell types: {list(lm22.columns)}")

# ── 2. CPM conversion ─────────────────────────────────────────────────────────
print("\nConverting counts to CPM...")
lib_sizes = counts.sum(axis=0)
cpm = counts.div(lib_sizes, axis=1) * 1e6

# Extract gene symbol if gene IDs are "ENSGxxx.v_SYMBOL" (RSEM format)
if cpm.index[0].count("_") >= 1 and cpm.index[0].startswith("ENSG"):
    symbol_index = cpm.index.str.split("_", n=1).str[1]
    cpm.index = symbol_index
    # Collapse duplicate symbols by taking the mean CPM across duplicates
    cpm = cpm.groupby(cpm.index).mean()
    print(f"  Gene IDs converted: ENSG.version_SYMBOL → SYMBOL ({len(cpm):,} unique symbols)")

# ── 3. Gene overlap ───────────────────────────────────────────────────────────
common_genes = cpm.index.intersection(lm22.index)
print(f"Genes in LM22 found in mixture: {len(common_genes):,} / {lm22.shape[0]:,}")
if len(common_genes) < 50:
    print("WARNING: very few shared genes — check gene ID format (symbol vs Ensembl)")

mix_sub = cpm.loc[common_genes]   # genes × samples
sig_sub = lm22.loc[common_genes]  # genes × cell_types

# ── 4. Quantile normalization of mixture (per sample) ─────────────────────────
def quantile_normalize_sample(v):
    """Scale one sample's CPM values to uniform [0,1] rank-based transform."""
    r = rankdata(v, method="average") / len(v)
    return r

print("Quantile-normalizing mixture samples...")
mix_norm = mix_sub.apply(quantile_normalize_sample, axis=0)  # genes × samples

# Normalize signature columns the same way
sig_norm = sig_sub.apply(quantile_normalize_sample, axis=0)  # genes × cell_types

# ── 5. NNLS deconvolution ─────────────────────────────────────────────────────
print(f"\nRunning NNLS deconvolution ({mix_norm.shape[1]} samples)...")
sig_mat  = sig_norm.values        # (n_genes, n_celltypes)
cell_types = sig_norm.columns.tolist()
n_samples  = mix_norm.shape[1]
sample_ids = mix_norm.columns.tolist()

fractions  = np.zeros((n_samples, len(cell_types)), dtype=np.float64)
rmse_list  = np.zeros(n_samples, dtype=np.float64)

for i, sid in enumerate(sample_ids):
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_samples}", flush=True)
    b = mix_norm[sid].values
    x, res = nnls(sig_mat, b)
    total = x.sum()
    fractions[i] = x / total if total > 0 else x
    fitted = sig_mat @ x
    rmse_list[i] = np.sqrt(np.mean((b - fitted) ** 2))

print(f"  Done. Median RMSE: {np.median(rmse_list):.4f}")

# ── 6. Save fractions ─────────────────────────────────────────────────────────
frac_df = pd.DataFrame(fractions, index=sample_ids, columns=cell_types)
frac_df.index.name = "SampleID"
frac_df.to_csv(os.path.join(DATA, "rna_cell_fractions.tsv"), sep="\t", float_format="%.6f")
print(f"\nSaved → {DATA}/rna_cell_fractions.tsv")

stats_df = pd.DataFrame({"SampleID": sample_ids, "RMSE": rmse_list,
                          "n_genes": len(common_genes)})
stats_df.to_csv(os.path.join(DATA, "rna_deconv_stats.tsv"), sep="\t", index=False)

# ── 7. Boxplot: per-group per-cell-type ───────────────────────────────────────
print("Generating boxplot...")
groups = sample_info.loc[sample_ids, "Group"].values
group_order = ["AC", "RC", "HC"]

n_types = len(cell_types)
n_cols  = 4
n_rows  = int(np.ceil(n_types / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axes = axes.flatten()

for k, ct in enumerate(cell_types):
    ax = axes[k]
    data_by_group = [frac_df.loc[groups == g, ct].values for g in group_order]
    bp = ax.boxplot(data_by_group, patch_artist=True, widths=0.5,
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
fig.suptitle("RNA-seq Cell Type Fractions (NNLS / LM22)", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "rna_celltype_boxplot.png"), dpi=150, bbox_inches="tight")
print(f"Saved → {FIGS}/rna_celltype_boxplot.png")

# ── 8. Heatmap: sample × cell-type ───────────────────────────────────────────
print("Generating heatmap...")
sort_idx = np.argsort(groups)
frac_sorted = frac_df.iloc[sort_idx].T
group_sorted = groups[sort_idx]

fig, ax = plt.subplots(figsize=(min(n_samples * 0.02 + 2, 20), n_types * 0.4 + 2))
im = ax.imshow(frac_sorted.values, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=frac_sorted.values.max())
ax.set_yticks(range(n_types))
ax.set_yticklabels(cell_types, fontsize=7)
ax.set_xlabel("Samples (sorted by group)", fontsize=9)
ax.set_title("Cell-type fractions (RNA NNLS)", fontsize=10)

# Group color strip above heatmap (convert hex to RGB array)
def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

strip_height = 0.02
y_strip = ax.get_position().y1 + strip_height
ax_strip = fig.add_axes([ax.get_position().x0, y_strip,
                          ax.get_position().width, strip_height])
strip_rgb = np.array([[hex_to_rgb(PALETTE[g]) for g in group_sorted]])  # (1, n, 3)
ax_strip.imshow(strip_rgb, aspect="auto")
ax_strip.axis("off")

plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Proportion")
plt.savefig(os.path.join(FIGS, "rna_celltype_heatmap.png"), dpi=150, bbox_inches="tight")
print(f"Saved → {FIGS}/rna_celltype_heatmap.png")

# ── 9. Summary ────────────────────────────────────────────────────────────────
print("\nSummary:")
print(f"  Genes used       : {len(common_genes):,}")
print(f"  Samples          : {n_samples:,}")
print(f"  Cell types       : {len(cell_types)}")
print(f"  Median RMSE      : {np.median(rmse_list):.4f}")
print("\nMean fractions by group:")
for g in group_order:
    mask = groups == g
    print(f"  {g}: " + "  ".join(f"{ct[:12]}={frac_df.loc[mask, ct].mean():.3f}"
                                   for ct in cell_types[:5]) + " ...")
print("Done.")
