"""
RNA-seq normalization v2: CPM filter + log2(CPM+1).

Changes from v1:
  - Input: rna_combat_corrected_v2.tsv / sample_info_v2.tsv
    (K1 year-corrected → cross-batch ComBat)
  - Output: rna_logcpm_v2.tsv.gz, rna_gene_filter_v2.tsv, pca_logcpm_v2.png

Steps identical to v1:
  1. CPM filter: CPM > 1 in >= 50% of smallest group
  2. log2(CPM + 1) per sample
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
MIN_FRAC   = 0.50

PALETTE_GROUP = {"AC": "#E63946", "RC": "#F4A261", "HC": "#457B9D"}

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading ComBat-seq v2 corrected counts...")
counts = pd.read_csv(os.path.join(IN_DIR, "rna_combat_corrected_v2.tsv"),
                     sep="\t", index_col=0)
sample_info = pd.read_csv(os.path.join(IN_DIR, "sample_info_v2.tsv"), sep="\t",
                          index_col="SampleID")
sample_info = sample_info.loc[counts.columns]
print(f"  Genes: {counts.shape[0]:,}, Samples: {counts.shape[1]}")
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

gene_log = pd.DataFrame({
    "gene":        counts.index,
    "mean_count":  counts.mean(axis=1).values,
    "mean_cpm":    cpm.mean(axis=1).values,
    "pass_filter": pass_filter.values,
})
gene_log.to_csv(os.path.join(DATA, "rna_gene_filter_v2.tsv"), sep="\t", index=False)

# ── 3. log2(CPM + 1) ──────────────────────────────────────────────────────────
print("\nComputing log2(CPM + 1)...")
logcpm = np.log2(cpm_filt + 1)
print(f"  logCPM matrix: {logcpm.shape}")

# ── 4. Save ───────────────────────────────────────────────────────────────────
out_path = os.path.join(DATA, "rna_logcpm_v2.tsv.gz")
logcpm.to_csv(out_path, sep="\t", compression="gzip")
print(f"Saved → {out_path}")

# ── 5. PCA ────────────────────────────────────────────────────────────────────
print("Running PCA (top 5,000 variable genes)...")
top5k = logcpm.var(axis=1).nlargest(5000).index
X = logcpm.loc[top5k].T.values
X = X - X.mean(axis=0)
pca = PCA(n_components=10, random_state=42)
pcs = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_ * 100
print(f"  PC1: {var_exp[0]:.1f}%  PC2: {var_exp[1]:.1f}%  PC3: {var_exp[2]:.1f}%")

groups = sample_info["Group"].values
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RNA-seq log2(CPM+1) v2 — PCA\n(K1 year-corrected → cross-batch HC ComBat-seq)", fontsize=12)
for ax, (pcx, pcy) in zip(axes, [(0,1),(1,2)]):
    for grp, color in PALETTE_GROUP.items():
        mask = groups == grp
        ax.scatter(pcs[mask, pcx], pcs[mask, pcy],
                   c=color, s=12, alpha=0.7, linewidths=0, label=grp)
    ax.set_xlabel(f"PC{pcx+1} ({var_exp[pcx]:.1f}%)")
    ax.set_ylabel(f"PC{pcy+1} ({var_exp[pcy]:.1f}%)")
    ax.set_title(f"PC{pcx+1} vs PC{pcy+1}")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")
handles = [mpatches.Patch(color=c, label=g) for g, c in PALETTE_GROUP.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.06), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "pca_logcpm_v2.png"), dpi=150, bbox_inches="tight")
print(f"Saved → figures/pca_logcpm_v2.png")

print(f"\nSummary (v2):")
print(f"  Genes after CPM filter : {logcpm.shape[0]:,}")
print(f"  Samples                : {logcpm.shape[1]:,}")
print(f"  PCA PC1: {var_exp[0]:.1f}%  PC2: {var_exp[1]:.1f}%")
print("Done.")
