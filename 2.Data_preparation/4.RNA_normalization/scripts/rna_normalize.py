"""
RNA-seq normalization: CPM filter + log2(CPM+1).

Input : rna_combat_corrected.tsv  (ComBat-seq corrected counts, genes × samples)
        sample_info.tsv            (SampleID, Group, FinalBatch)

Steps:
  1. Load corrected counts + sample metadata.
  2. CPM filter: keep genes with CPM > 1 in >= 50% of the smallest group (RC=158 → 79 samples).
  3. log2(CPM + 1) normalization per sample (library-size normalized).
  4. Save:
       data/rna_logcpm.tsv.gz     — log2(CPM+1) matrix (genes × samples)
       data/rna_gene_filter.tsv   — kept / removed gene list with stats
       figures/pca_logcpm.png     — PCA (coloured by Group + FinalBatch)

Note: This matrix is for visualization, normative deviation modeling (Path B), and
      exploratory analysis. DESeq2 DE tests (3.Discovery) use raw ComBat-corrected
      counts directly with ~FinalBatch + condition design.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA

BASE   = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
IN_DIR = os.path.join(os.path.dirname(BASE), "2.RNA_batch_correction", "data")
DATA   = os.path.join(BASE, "data")
FIGS   = os.path.join(BASE, "figures")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

CPM_CUTOFF = 1
MIN_FRAC   = 0.50   # ≥50% of smallest group

PALETTE_GROUP = {"AC": "#E63946", "RC": "#F4A261", "HC": "#457B9D"}
PALETTE_BATCH = {
    "AC": "#E63946", "RC": "#F4A261", "HC": "#457B9D",
}

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("Loading ComBat-seq corrected counts...")
counts = pd.read_csv(os.path.join(IN_DIR, "rna_combat_corrected.tsv"),
                     sep="\t", index_col=0)
sample_info = pd.read_csv(os.path.join(IN_DIR, "sample_info.tsv"), sep="\t",
                          index_col="SampleID")
sample_info = sample_info.loc[counts.columns]
print(f"  Genes: {counts.shape[0]}, Samples: {counts.shape[1]}")
print(f"  Groups: {sample_info['Group'].value_counts().to_dict()}")

# ── 2. CPM filter ─────────────────────────────────────────────────────────────
print("\nApplying CPM filter (CPM > 1 in >= 50% of smallest group)...")
lib_sizes = counts.sum(axis=0)
cpm = counts.div(lib_sizes, axis=1) * 1e6

group_sizes = sample_info["Group"].value_counts()
smallest_n  = group_sizes.min()
min_samples = int(np.ceil(MIN_FRAC * smallest_n))
print(f"  Smallest group: {group_sizes.idxmin()} (n={smallest_n}) → min {min_samples} samples")

pass_filter = (cpm > CPM_CUTOFF).sum(axis=1) >= min_samples
counts_filt = counts.loc[pass_filter]
cpm_filt    = cpm.loc[pass_filter]
print(f"  Genes passing: {pass_filter.sum():,} / {len(pass_filter):,}")

# Gene filter log
gene_log = pd.DataFrame({
    "gene": counts.index,
    "mean_count": counts.mean(axis=1).values,
    "mean_cpm":   cpm.mean(axis=1).values,
    "pass_filter": pass_filter.values,
})
gene_log.to_csv(os.path.join(DATA, "rna_gene_filter.tsv"), sep="\t", index=False)

# ── 3. log2(CPM + 1) normalization ────────────────────────────────────────────
print("\nComputing log2(CPM + 1)...")
logcpm = np.log2(cpm_filt + 1)
print(f"  logCPM matrix shape: {logcpm.shape}")

# ── 4. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA, "rna_logcpm.tsv.gz")
logcpm.to_csv(out_path, sep="\t", compression="gzip")
print(f"\nSaved → {out_path}")

# ── 5. PCA ────────────────────────────────────────────────────────────────────
print("Running PCA (top 5,000 variable genes)...")
gene_var = logcpm.var(axis=1)
top5k    = gene_var.nlargest(5000).index
X = logcpm.loc[top5k].T.values
X = X - X.mean(axis=0)
pca = PCA(n_components=10, random_state=42)
pcs = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_ * 100
print(f"  PC1: {var_exp[0]:.1f}%  PC2: {var_exp[1]:.1f}%  PC3: {var_exp[2]:.1f}%")

groups = sample_info["Group"].values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RNA-seq log2(CPM+1) — PCA (ComBat-seq HC corrected)", fontsize=12)
for ax, (pcx, pcy) in zip(axes, [(0,1),(1,2)]):
    for grp, color in PALETTE_GROUP.items():
        mask = groups == grp
        ax.scatter(pcs[mask, pcx], pcs[mask, pcy],
                   c=color, s=12, alpha=0.7, linewidths=0)
    ax.set_xlabel(f"PC{pcx+1} ({var_exp[pcx]:.1f}%)")
    ax.set_ylabel(f"PC{pcy+1} ({var_exp[pcy]:.1f}%)")
    ax.set_title(f"PC{pcx+1} vs PC{pcy+1}")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
handles = [mpatches.Patch(color=c, label=g) for g,c in PALETTE_GROUP.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.06), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "pca_logcpm.png"), dpi=150, bbox_inches="tight")
print(f"Saved → {os.path.join(FIGS, 'pca_logcpm.png')}")

print(f"\nSummary:")
print(f"  Genes after CPM filter : {logcpm.shape[0]:,}")
print(f"  Samples                : {logcpm.shape[1]:,}")
print(f"  PCA PC1: {var_exp[0]:.1f}%  PC2: {var_exp[1]:.1f}%")
print("Done.")
