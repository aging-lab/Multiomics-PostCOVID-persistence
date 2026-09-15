"""
Korea10K_1 내부 sub-grouping 원인 분석.

전체 샘플 PCA에서 Korea10K_1이 나머지와 분리됨 (PC1 18.6%).
Korea10K_1 내부에서도 샘플들이 추가로 나뉘는 패턴 확인:
  - Sequencing Year (2017 / 2018 / 2019)
  - WGS Platform (NovaSeq / HiSeq X)
  - 모집군 (일반 / 심근경색 / 기타)
  - Sex (M / F)
"""

import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

RNA_ROOT = "/BiO/Research/Infectomics/LongCOVID/Resources/RNA"
META_PATH = "/BiO/Research/Infectomics/LongCOVID/Resources/Metadata/KOREA10K_multiomics_metadata.xlsx"
OUT_DIR  = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

BATCHES = {
    "AC":              os.path.join(RNA_ROOT, "Acute_phase", "4_expmtx",
                                    "expression_matrix_genes.results_expected_count.tsv"),
    "RC":              os.path.join(RNA_ROOT, "Recovered_phase", "4_expmtx",
                                    "expression_matrix_genes.results_expected_count.tsv"),
    "HC-HealthyCtrl":  os.path.join(RNA_ROOT, "Healthy_controls", "Korea10K",
                                    "HealhyControls", "4_expmtx",
                                    "expression_matrix_genes.results_expected_count.tsv"),
    "HC-Korea10K_1":   os.path.join(RNA_ROOT, "Healthy_controls", "Korea10K",
                                    "Korea10K_1", "4_expmtx",
                                    "expression_matrix_genes.results_expected_count.tsv"),
    "HC-Korea10K_2":   os.path.join(RNA_ROOT, "Healthy_controls", "Korea10K",
                                    "Korea10K_2", "4_expmtx",
                                    "expression_matrix_genes.results_expected_count.tsv"),
    "HC-WelfareGenome":os.path.join(RNA_ROOT, "Healthy_controls", "WelfareGenome",
                                    "4_expmtx",
                                    "expression_matrix_genes.results_expected_count.tsv"),
}

# ── 1. Load and merge ─────────────────────────────────────────────────────────
print("Loading expression matrices...")
dfs = []
batch_labels = []
for batch, path in BATCHES.items():
    df = pd.read_csv(path, sep="\t", index_col=0)
    df.columns = [f"{batch}::{c}" for c in df.columns]
    dfs.append(df)
    batch_labels.extend([batch] * df.shape[1])
    print(f"  {batch}: {df.shape[1]} samples")

merged = pd.concat(dfs, axis=1, join="inner")
print(f"Merged: {merged.shape[0]:,} genes x {merged.shape[1]} samples")

# ── 2. Normalize + filter ─────────────────────────────────────────────────────
logmat = np.log2(merged.values + 1)
gene_mean = logmat.mean(axis=1)
keep = gene_mean > 0.5
logmat = logmat[keep]
print(f"Genes after mean>0.5 filter: {keep.sum():,}")

# ── 3. PCA (top 5000 variable genes) ─────────────────────────────────────────
gene_var = logmat.var(axis=1)
top5k = np.argsort(gene_var)[-5000:]
X = logmat[top5k].T
X = X - X.mean(axis=0)
pca = PCA(n_components=20, random_state=42)
pcs = pca.fit_transform(X)
var_exp = pca.explained_variance_ratio_ * 100
print(f"PCA done. PC1={var_exp[0]:.1f}%  PC2={var_exp[1]:.1f}%  PC3={var_exp[2]:.1f}%")

# Sample index mapping
sample_ids_full = [c.split("::", 1)[1] for c in merged.columns]
batch_arr = np.array(batch_labels)

# ── 4. Load metadata ──────────────────────────────────────────────────────────
print("Loading metadata...")
meta = pd.read_excel(META_PATH)
meta = meta.set_index("KU10K-ID")

k1_mask  = batch_arr == "HC-Korea10K_1"
k1_ids   = np.array(sample_ids_full)[k1_mask]
k1_pcs   = pcs[k1_mask]

meta_k1  = meta.reindex(k1_ids)

# ── 5. Figure 1: All samples, Korea10K_1 highlighted ─────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("PCA — Korea10K_1 sub-grouping investigation", fontsize=12)

batch_palette = {
    "AC":               "#E63946",
    "RC":               "#F4A261",
    "HC-HealthyCtrl":   "#2A9D8F",
    "HC-Korea10K_1":    "#457B9D",
    "HC-Korea10K_2":    "#A8DADC",
    "HC-WelfareGenome": "#6A0572",
}

for ax, (pcx, pcy) in zip(axes, [(0,1),(1,2)]):
    for batch, color in batch_palette.items():
        m = batch_arr == batch
        alpha = 0.7 if "Korea10K_1" in batch else 0.25
        size  = 14  if "Korea10K_1" in batch else 8
        ax.scatter(pcs[m, pcx], pcs[m, pcy], c=color, s=size,
                   alpha=alpha, linewidths=0, label=batch)
    ax.set_xlabel(f"PC{pcx+1} ({var_exp[pcx]:.1f}%)")
    ax.set_ylabel(f"PC{pcy+1} ({var_exp[pcy]:.1f}%)")

handles = [mpatches.Patch(color=c, label=b) for b, c in batch_palette.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9,
           bbox_to_anchor=(0.5, -0.06), frameon=False)
plt.tight_layout()
out = os.path.join(OUT_DIR, "pca_korea10k1_overview.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close()

# ── 6. Figure 2: Korea10K_1 only, coloured by metadata variables ─────────────
META_VARS = {
    "10K Sequencing Year": {
        2017: "#1F77B4", 2018: "#FF7F0E", 2019: "#2CA02C", np.nan: "#CCCCCC"
    },
    "WGS Platform": {
        "NovaSeq": "#9467BD", "HiSeq X": "#8C564B", "unknown": "#CCCCCC"
    },
    "모집군": {
        "일반": "#17BECF", "심근경색": "#D62728", "소아뇌암": "#BCBD22",
        "unknown": "#CCCCCC"
    },
    "Sex": {
        "M": "#1F77B4", "F": "#E377C2", "unknown": "#CCCCCC"
    },
}

fig, axes = plt.subplots(len(META_VARS), 2,
                          figsize=(12, len(META_VARS) * 4))
fig.suptitle("Korea10K_1 sub-grouping — coloured by metadata", fontsize=12)

for row_i, (var, palette) in enumerate(META_VARS.items()):
    values = meta_k1[var].fillna("unknown") if var in meta_k1.columns else \
             pd.Series(["unknown"] * len(k1_ids), index=k1_ids)
    # normalise year to string
    if var == "10K Sequencing Year":
        values = values.apply(lambda v: int(v) if v != "unknown" else "unknown")

    for col_j, (pcx, pcy) in enumerate([(0,1),(1,2)]):
        ax = axes[row_i, col_j]
        plotted = set()
        for val, color in palette.items():
            m = values == val
            if m.sum() == 0:
                continue
            ax.scatter(k1_pcs[m, pcx], k1_pcs[m, pcy],
                       c=color, s=10, alpha=0.7, linewidths=0, label=str(val))
            plotted.add(val)
        # leftover values not in palette
        leftover = ~values.isin(plotted)
        if leftover.sum() > 0:
            ax.scatter(k1_pcs[leftover, pcx], k1_pcs[leftover, pcy],
                       c="#CCCCCC", s=10, alpha=0.5, linewidths=0, label="other")
        ax.set_xlabel(f"PC{pcx+1} ({var_exp[pcx]:.1f}%)", fontsize=8)
        ax.set_ylabel(f"PC{pcy+1} ({var_exp[pcy]:.1f}%)", fontsize=8)
        if col_j == 0:
            ax.set_ylabel(f"{var}\nPC{pcy+1}", fontsize=8)
        ax.legend(fontsize=7, markerscale=2, frameon=False,
                  loc="upper right", ncol=2)

plt.tight_layout()
out = os.path.join(OUT_DIR, "pca_korea10k1_subgroup.png")
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved -> {out}")
plt.close()

# ── 7. Report: which samples are in which PC region ──────────────────────────
print("\nKorea10K_1 sub-group summary:")
for var in ["10K Sequencing Year", "WGS Platform", "모집군", "Sex"]:
    if var not in meta_k1.columns:
        continue
    vals = meta_k1[var].fillna("unknown")
    if var == "10K Sequencing Year":
        vals = vals.apply(lambda v: int(v) if v != "unknown" else "unknown")
    print(f"\n  {var}:")
    print(f"    {vals.value_counts().to_dict()}")

# PC1 split: find Korea10K_1 samples with high vs low PC1
k1_pc1 = k1_pcs[:, 0]
q33 = np.percentile(k1_pc1, 33)
q67 = np.percentile(k1_pc1, 67)
low_mask  = k1_pc1 < q33
high_mask = k1_pc1 > q67
mid_mask  = ~low_mask & ~high_mask

print(f"\nPC1 tertile split within Korea10K_1:")
print(f"  Low  (PC1 < {q33:.1f}): n={low_mask.sum()}")
print(f"  Mid  ({q33:.1f} - {q67:.1f}): n={mid_mask.sum()}")
print(f"  High (PC1 > {q67:.1f}): n={high_mask.sum()}")

for var in ["10K Sequencing Year", "WGS Platform", "모집군", "Sex"]:
    if var not in meta_k1.columns:
        continue
    vals = meta_k1[var].fillna("unknown")
    if var == "10K Sequencing Year":
        vals = vals.apply(lambda v: int(v) if v != "unknown" else "unknown")
    print(f"\n  {var} by PC1 tertile:")
    for label, m in [("Low", low_mask), ("Mid", mid_mask), ("High", high_mask)]:
        print(f"    {label}: {vals[m].value_counts().to_dict()}")

print("\nDone.")
