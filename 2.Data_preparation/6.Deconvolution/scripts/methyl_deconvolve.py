"""
Methylation cell-type deconvolution (NNLS, EpiDISH centDHSbloodDMC reference).

Data type: SureSelect Targeted Bisulfite-seq (Agilent)
  - CpG island / promoter / enhancer enriched capture
  - ~5.8% genome coverage, 1.6M CpGs (hg38, coverage >= 10x)
  - CpG-rich context matches Illumina 450K array probe positions

Reference choice rationale:
  - Loyfer 2023 WGBS atlas: only 13.5% overlap (markers spread genome-wide, outside targets)
  - EpiDISH centDHSbloodDMC (450K-based): 76.4% overlap (both CpG-island enriched)
  - Array vs sequencing beta-values at same positions: r > 0.95 (published literature)
  => EpiDISH is the appropriate choice for this data type

Reference build:
  EpiDISH centDHSbloodDMC.m (333 probes, 7 cell types)
  -> Illumina 450K manifest hg19 positions
  -> pyliftover hg19->hg38
  -> 326 hg38 positions; 249 present in our matrix

Cell types (7): B, NK, CD4T, CD8T, Mono, Neutro, Eosino

Input:
  cpg_matrix_beta.npz      — beta matrix (1,615,056 CpGs x 630 samples, hg38)
  cpg_sample_info.tsv      — SampleID, Group
  resources/methyl_blood_ref.tsv  — EpiDISH centDHSbloodDMC (hg38 positions x 7 cell types)

Output:
  data/methyl_cell_fractions.tsv      — 7 cell types x 630 samples (proportions)
  data/methyl_deconv_stats.tsv        — per-sample RMSE, n_cpgs_used
  figures/methyl_celltype_boxplot.png
  figures/methyl_celltype_heatmap.png
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

BASE       = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
METHYL_DIR = os.path.join(os.path.dirname(BASE), "5.Methyl_matrix", "data")
DATA       = os.path.join(BASE, "data")
FIGS       = os.path.join(BASE, "figures")
RESOURCES  = os.path.join(BASE, "resources")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

REF_PATH = os.path.join(RESOURCES, "methyl_blood_ref.tsv")
PALETTE  = {"AC": "#E63946", "RC": "#F4A261", "HC": "#457B9D"}

# ── 0. Check ───────────────────────────────────────────────────────────────────
if not os.path.isfile(REF_PATH):
    print(f"ERROR: Reference not found at {REF_PATH}")
    print("  Run: python resources/build_methyl_ref.py")
    sys.exit(1)

# ── 1. Load beta matrix ───────────────────────────────────────────────────────
print("Loading beta-value matrix...")
npz        = np.load(os.path.join(METHYL_DIR, "cpg_matrix_beta.npz"), allow_pickle=True)
beta_mat   = npz["matrix"]      # (n_cpg, n_samples) float32
cpg_ids    = npz["cpg_ids"]
sample_ids = list(npz["sample_ids"])
print(f"  Matrix: {beta_mat.shape[0]:,} CpGs x {beta_mat.shape[1]} samples")

sample_info = pd.read_csv(os.path.join(METHYL_DIR, "cpg_sample_info.tsv"), sep="\t",
                          index_col="SampleID")

# ── 2. Load reference ─────────────────────────────────────────────────────────
print("Loading EpiDISH reference (hg38)...")
ref = pd.read_csv(REF_PATH, sep="\t", index_col=0)
cell_types = ref.columns.tolist()
print(f"  Reference: {len(ref):,} CpGs x {len(cell_types)} cell types: {cell_types}")

# ── 3. CpG overlap ────────────────────────────────────────────────────────────
cpg_to_idx  = {cid: i for i, cid in enumerate(cpg_ids)}
common_cpgs = [c for c in ref.index if c in cpg_to_idx]
print(f"  Shared CpGs: {len(common_cpgs)} / {len(ref)} ({len(common_cpgs)/len(ref)*100:.1f}%)")

if len(common_cpgs) < 20:
    print("ERROR: too few shared CpGs.")
    sys.exit(1)

row_idx = np.array([cpg_to_idx[c] for c in common_cpgs])
ref_mat = ref.loc[common_cpgs].values.astype(np.float64)  # (n_common, n_celltypes)

# ── 4. Extract mixture and impute NaN ─────────────────────────────────────────
mix_sub  = beta_mat[row_idx, :].astype(np.float64)   # (n_common, n_samples)
ref_mean = ref_mat.mean(axis=1)
for j in range(mix_sub.shape[1]):
    nan_mask = np.isnan(mix_sub[:, j])
    mix_sub[nan_mask, j] = ref_mean[nan_mask]

# ── 5. NNLS deconvolution ─────────────────────────────────────────────────────
n_samples   = len(sample_ids)
n_celltypes = len(cell_types)
fractions   = np.zeros((n_samples, n_celltypes), dtype=np.float64)
rmse_list   = np.zeros(n_samples, dtype=np.float64)

print(f"\nRunning NNLS ({n_samples} samples, {len(common_cpgs)} CpGs)...")
for i in range(n_samples):
    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{n_samples}", flush=True)
    b = mix_sub[:, i]
    x, _ = nnls(ref_mat, b)
    total = x.sum()
    fractions[i] = x / total if total > 0 else x
    rmse_list[i] = np.sqrt(np.mean((b - ref_mat @ x) ** 2))

print(f"  Done. Median RMSE: {np.median(rmse_list):.4f}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
frac_df = pd.DataFrame(fractions, index=sample_ids, columns=cell_types)
frac_df.index.name = "SampleID"
frac_df.to_csv(os.path.join(DATA, "methyl_cell_fractions.tsv"), sep="\t", float_format="%.6f")
print(f"\nSaved -> {DATA}/methyl_cell_fractions.tsv")

stats_df = pd.DataFrame({"SampleID": sample_ids, "RMSE": rmse_list,
                          "n_cpgs": len(common_cpgs)})
stats_df.to_csv(os.path.join(DATA, "methyl_deconv_stats.tsv"), sep="\t", index=False)

# ── 7. Plots ──────────────────────────────────────────────────────────────────
groups      = sample_info.loc[sample_ids, "Group"].values
group_order = ["AC", "RC", "HC"]

# Boxplot
print("Generating boxplot...")
n_cols = min(4, n_celltypes)
n_rows = int(np.ceil(n_celltypes / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3.5, n_rows * 3.5))
axes = np.array(axes).flatten()

for k, ct in enumerate(cell_types):
    ax = axes[k]
    data_by_group = [frac_df.loc[groups == g, ct].values for g in group_order]
    bp = ax.boxplot(data_by_group, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=1.5),
                    flierprops=dict(marker=".", ms=3, alpha=0.5))
    for patch, grp in zip(bp["boxes"], group_order):
        patch.set_facecolor(PALETTE[grp])
    ax.set_xticks(range(1, len(group_order) + 1))
    ax.set_xticklabels(group_order, fontsize=9)
    ax.set_title(ct, fontsize=9, pad=2)
    ax.set_ylabel("Proportion", fontsize=8)

for k in range(n_celltypes, len(axes)):
    axes[k].set_visible(False)

handles = [mpatches.Patch(color=PALETTE[g], label=g) for g in group_order]
fig.legend(handles=handles, loc="lower right", ncol=1, fontsize=9, frameon=False)
fig.suptitle(
    "Methylation Cell-Type Fractions\n"
    "(NNLS / EpiDISH centDHSbloodDMC, hg38; SureSelect Targeted Bisulfite-seq)",
    fontsize=10
)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "methyl_celltype_boxplot.png"), dpi=150, bbox_inches="tight")
print(f"Saved -> {FIGS}/methyl_celltype_boxplot.png")

# Heatmap
print("Generating heatmap...")
sort_idx     = np.argsort(groups)
frac_sorted  = frac_df.iloc[sort_idx].T
group_sorted = groups[sort_idx]

def hex_to_rgb(h):
    h = h.lstrip("#")
    return [int(h[i:i+2], 16) / 255.0 for i in (0, 2, 4)]

fig, ax = plt.subplots(figsize=(min(n_samples * 0.025 + 2, 22), n_celltypes * 0.65 + 2))
im = ax.imshow(frac_sorted.values, aspect="auto", cmap="YlOrRd",
               vmin=0, vmax=frac_sorted.values.max())
ax.set_yticks(range(n_celltypes))
ax.set_yticklabels(cell_types, fontsize=9)
ax.set_xlabel("Samples (sorted by group)", fontsize=9)
ax.set_title("Cell-type fractions (Methyl NNLS / EpiDISH)", fontsize=10)
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Proportion")

pos = ax.get_position()
strip_h = 0.015
ax_s = fig.add_axes([pos.x0, pos.y1 + strip_h * 0.2, pos.width, strip_h])
strip_rgb = np.array([[hex_to_rgb(PALETTE[g]) for g in group_sorted]])
ax_s.imshow(strip_rgb, aspect="auto")
ax_s.axis("off")

plt.savefig(os.path.join(FIGS, "methyl_celltype_heatmap.png"), dpi=150, bbox_inches="tight")
print(f"Saved -> {FIGS}/methyl_celltype_heatmap.png")

# ── 8. Summary ────────────────────────────────────────────────────────────────
print("\nSummary:")
print(f"  CpGs used  : {len(common_cpgs)}")
print(f"  Samples    : {n_samples}")
print(f"  Cell types : {n_celltypes}")
print(f"  Median RMSE: {np.median(rmse_list):.4f}")
print("\nMean fractions by group:")
for g in group_order:
    mask = groups == g
    row  = frac_df.loc[mask].mean()
    print(f"  {g}: " + "  ".join(f"{ct}={row[ct]:.3f}" for ct in cell_types))
print("Done.")
