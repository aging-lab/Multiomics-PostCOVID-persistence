"""
RNA batch-effect PCA.

Batches:
  AC              — Acute COVID (single sequencing run)
  RC              — Recovered COVID (single sequencing run)
  HC-HealthyCtrl  — Korea10K pilot batch (HealhyControls)
  HC-Korea10K_1   — Korea10K large batch
  HC-Korea10K_2   — Korea10K additional batch
  HC-WelfareGenome— WelfareGenome batch

Exclusions (Korea10K_1):
  - 심근경색 87 samples
  - 소아뇌암 18 samples  (KU10K + U10K prefix variants)
  - 숫자 ID 25 samples   (WelfareGenome duplicates — kept in WG batch, dropped from K1)
  Source: Analysis/2.Data_preparation/korea10k1_exclude.tsv
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RNA_ROOT    = "/BiO/Research/Infectomics/LongCOVID/Resources/RNA"
EXCLUDE_TSV = "/BiO/Research/Infectomics/LongCOVID/Analysis/2.Data_preparation/korea10k1_exclude.tsv"
OUT_DIR     = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

BATCHES = {
    "AC":               os.path.join(RNA_ROOT, "Acute_phase", "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
    "RC":               os.path.join(RNA_ROOT, "Recovered_phase", "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
    "HC-HealthyCtrl":   os.path.join(RNA_ROOT, "Healthy_controls", "Korea10K",
                                     "HealhyControls", "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
    "HC-Korea10K_1":    os.path.join(RNA_ROOT, "Healthy_controls", "Korea10K",
                                     "Korea10K_1", "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
    "HC-Korea10K_2":    os.path.join(RNA_ROOT, "Healthy_controls", "Korea10K",
                                     "Korea10K_2", "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
    "HC-WelfareGenome": os.path.join(RNA_ROOT, "Healthy_controls", "WelfareGenome",
                                     "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
}

PALETTE = {
    "AC":               "#E63946",
    "RC":               "#F4A261",
    "HC-HealthyCtrl":   "#2A9D8F",
    "HC-Korea10K_1":    "#457B9D",
    "HC-Korea10K_2":    "#A8DADC",
    "HC-WelfareGenome": "#6A0572",
}

# ── 1. Load exclusion list ────────────────────────────────────────────────────
exclude_ids = set()
if os.path.isfile(EXCLUDE_TSV):
    excl_df = pd.read_csv(EXCLUDE_TSV, sep="\t")
    exclude_ids = set(excl_df["SampleID"].astype(str))
    print(f"Exclusion list loaded: {len(exclude_ids)} samples")

# ── 2. Load and merge ─────────────────────────────────────────────────────────
print("Loading expression matrices...")
dfs = []
batch_labels = []

for batch, path in BATCHES.items():
    df = pd.read_csv(path, sep="\t", index_col=0)
    before = df.shape[1]
    df = df.drop(columns=[c for c in df.columns if c in exclude_ids], errors="ignore")
    after = df.shape[1]
    if before != after:
        print(f"  {batch}: {before} -> {after} samples ({before-after} excluded)")
    else:
        print(f"  {batch}: {after} samples")
    df.columns = [f"{batch}::{c}" for c in df.columns]
    dfs.append(df)
    batch_labels.extend([batch] * df.shape[1])

merged = pd.concat(dfs, axis=1, join="inner")
print(f"Merged: {merged.shape[0]:,} genes x {merged.shape[1]} samples")

# ── 3. Normalize + filter ─────────────────────────────────────────────────────
logmat    = np.log2(merged.values + 1)
gene_mean = logmat.mean(axis=1)
keep      = gene_mean > 0.5
logmat    = logmat[keep]
print(f"Genes after mean>0.5 filter: {keep.sum():,}")

# ── 4. PCA ────────────────────────────────────────────────────────────────────
gene_var = logmat.var(axis=1)
top5k    = np.argsort(gene_var)[-5000:]
X = logmat[top5k].T
X = X - X.mean(axis=0)
pca     = PCA(n_components=10, random_state=42)
pcs     = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_ * 100
print(f"PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%  PC3={var_exp[2]:.1f}%")

batch_arr = np.array(batch_labels)

# ── 5. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("RNA-seq Batch Effect PCA\n(Korea10K_1: 심근경색 87 + 소아뇌암 18 + WG중복 25 제외)",
             fontsize=11)

for ax, (pcx, pcy) in zip(axes, [(0, 1), (1, 2)]):
    for batch, color in PALETTE.items():
        m = batch_arr == batch
        ax.scatter(pcs[m, pcx], pcs[m, pcy],
                   c=color, s=10, alpha=0.65, linewidths=0, label=batch)
    ax.set_xlabel(f"PC{pcx+1} ({var_exp[pcx]:.1f}%)")
    ax.set_ylabel(f"PC{pcy+1} ({var_exp[pcy]:.1f}%)")
    ax.set_title(f"PC{pcx+1} vs PC{pcy+1}")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")

handles = [mpatches.Patch(color=c, label=b) for b, c in PALETTE.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.06), frameon=False)
plt.tight_layout()
out = os.path.join(OUT_DIR, "pca_rna_batch.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close()

# Scree plot
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(range(1, len(var_exp)+1), var_exp, color="#457B9D")
ax.set_xlabel("PC")
ax.set_ylabel("Explained variance (%)")
ax.set_title("Scree plot (after non-normal exclusion)")
for i, v in enumerate(var_exp):
    ax.text(i+1, v+0.1, f"{v:.1f}", ha="center", fontsize=7)
plt.tight_layout()
out = os.path.join(OUT_DIR, "pca_rna_scree.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close()

print(f"\nSample counts after exclusion:")
for b in PALETTE:
    n = (batch_arr == b).sum()
    print(f"  {b}: {n}")
print(f"  Total: {len(batch_arr)}")
