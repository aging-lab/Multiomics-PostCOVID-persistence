"""
Korea10K_1 내부 sequencing year 배치 효과 제거 (ComBat-seq).

Background:
  PCA에서 Korea10K_1 샘플이 Sequencing Year에 따라 분리됨:
    2017 (HiSeq X):  163 samples
    2018 (NovaSeq):  109 samples
    2019 (NovaSeq):   98 samples  <- PC1 high / PC2 low 클러스터
  25개 numeric ID 샘플 (WG duplicates)은 제외.

Strategy:
  - Korea10K_1 단독으로 ComBat-seq 실행 (all HC/일반 → biological confounding 없음)
  - Batch variable: Sequencing Year (2017 / 2018 / 2019)
  - 보정된 K1 count matrix를 이후 cross-batch ComBat 입력으로 사용

Input:
  Korea10K_1 expression_matrix_genes.results_expected_count.tsv
  korea10k1_exclude.tsv  (심근경색 87 + 소아뇌암 18)
  KOREA10K_multiomics_metadata.xlsx

Output:
  data/k1_corrected_counts.tsv   — 보정된 count matrix (genes × samples)
  data/k1_sample_info.tsv        — SampleID, Year, n_samples
  figures/pca_k1_before_after.png
"""

import os
import numpy as np
import pandas as pd
from inmoose.pycombat import pycombat_seq
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE        = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
RNA_ROOT    = "/BiO/Research/Infectomics/LongCOVID/Resources/RNA"
META_PATH   = "/BiO/Research/Infectomics/LongCOVID/Resources/Metadata/KOREA10K_multiomics_metadata.xlsx"
EXCLUDE_TSV = "/BiO/Research/Infectomics/LongCOVID/Analysis/2.Data_preparation/korea10k1_exclude.tsv"
K1_PATH     = os.path.join(RNA_ROOT, "Healthy_controls", "Korea10K", "Korea10K_1",
                            "4_expmtx", "expression_matrix_genes.results_expected_count.tsv")
DATA_DIR    = os.path.join(BASE, "data")
FIG_DIR     = os.path.join(BASE, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

YEAR_PALETTE = {2017: "#1F77B4", 2018: "#FF7F0E", 2019: "#2CA02C"}

# ── 1. Load K1 matrix, apply exclusions ──────────────────────────────────────
print("Loading Korea10K_1 expression matrix...")
exclude_ids = set(pd.read_csv(EXCLUDE_TSV, sep="\t")["SampleID"].astype(str))
k1 = pd.read_csv(K1_PATH, sep="\t", index_col=0)
before = k1.shape[1]
k1 = k1.drop(columns=[c for c in k1.columns if c in exclude_ids], errors="ignore")
print(f"  {before} -> {k1.shape[1]} samples after exclusions")

# ── 2. Map sample IDs to Sequencing Year ─────────────────────────────────────
print("Loading metadata...")
meta = pd.read_excel(META_PATH).set_index("KU10K-ID")

sample_years = {}
dropped_no_year = []
for sid in k1.columns:
    if sid in meta.index:
        yr = meta.at[sid, "10K Sequencing Year"]
        if pd.notna(yr):
            sample_years[sid] = int(yr)
        else:
            dropped_no_year.append(sid)
    else:
        dropped_no_year.append(sid)   # numeric IDs (WG duplicates) — drop

if dropped_no_year:
    print(f"  Dropping {len(dropped_no_year)} samples with no year info "
          f"(WG duplicates / unresolved): {dropped_no_year[:5]}{'...' if len(dropped_no_year)>5 else ''}")
    k1 = k1.drop(columns=dropped_no_year, errors="ignore")

print(f"  Remaining: {k1.shape[1]} samples")
from collections import Counter
yr_counts = Counter(sample_years[s] for s in k1.columns)
for yr in sorted(yr_counts):
    print(f"    {yr}: {yr_counts[yr]} samples")

# ── 3. Low-expression filter ─────────────────────────────────────────────────
row_sum = k1.sum(axis=1)
k1 = k1.loc[row_sum > 10]
k1 = k1.round().astype(int)
print(f"  After low-expression filter (sum>10): {k1.shape[0]:,} genes")

# ── 4. Run ComBat-seq (Year as batch) ────────────────────────────────────────
batch_labels = [sample_years[s] for s in k1.columns]
print(f"\nRunning ComBat-seq (batch=Year, {k1.shape[1]} samples)...")
k1_corrected = pycombat_seq(counts=k1, batch=batch_labels)
print(f"  Done. Shape: {k1_corrected.shape}")

# ── 5. Save ───────────────────────────────────────────────────────────────────
out_counts = os.path.join(DATA_DIR, "k1_corrected_counts.tsv")
k1_corrected.to_csv(out_counts, sep="\t")
print(f"\nSaved -> {out_counts}")

info_df = pd.DataFrame({
    "SampleID": k1.columns,
    "Year":     batch_labels,
})
info_df.to_csv(os.path.join(DATA_DIR, "k1_sample_info.tsv"), sep="\t", index=False)

# ── 6. PCA before / after ─────────────────────────────────────────────────────
def pca_top5k(mat_df):
    log = np.log2(mat_df.values.astype(float) + 1)
    log_df = pd.DataFrame(log, index=mat_df.index, columns=mat_df.columns)
    log_df = log_df.loc[log_df.mean(axis=1) > 0.5]
    top = log_df.var(axis=1).nlargest(5000).index
    X = log_df.loc[top].T.values
    X = X - X.mean(axis=0)
    pca = PCA(n_components=5, random_state=42)
    pcs = pca.fit_transform(X)
    return pcs, pca.explained_variance_ratio_ * 100

print("\nRunning PCA (before)...")
pcs_before, var_before = pca_top5k(k1)
print(f"  PC1={var_before[0]:.1f}%  PC2={var_before[1]:.1f}%")

print("Running PCA (after)...")
pcs_after, var_after = pca_top5k(k1_corrected)
print(f"  PC1={var_after[0]:.1f}%  PC2={var_after[1]:.1f}%")

years_arr = np.array(batch_labels)

fig, axes = plt.subplots(2, 2, figsize=(14, 11))
fig.suptitle("Korea10K_1 Internal ComBat-seq (Batch = Sequencing Year)\n"
             "top 5,000 variable genes, log2(count+1)", fontsize=12)

for col_i, (pcs, var_exp, title) in enumerate([
        (pcs_before, var_before, "Before ComBat-seq"),
        (pcs_after,  var_after,  "After ComBat-seq"),
]):
    for row_i, (pcx, pcy) in enumerate([(0, 1), (1, 2)]):
        ax = axes[row_i][col_i]
        for yr, color in YEAR_PALETTE.items():
            m = years_arr == yr
            if m.sum() == 0:
                continue
            ax.scatter(pcs[m, pcx], pcs[m, pcy],
                       c=color, s=12, alpha=0.7, linewidths=0, label=str(yr))
        ax.set_xlabel(f"PC{pcx+1} ({var_exp[pcx]:.1f}%)", fontsize=9)
        ax.set_ylabel(f"PC{pcy+1} ({var_exp[pcy]:.1f}%)", fontsize=9)
        ax.set_title(f"{title} — PC{pcx+1} vs PC{pcy+1}", fontsize=10)
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        ax.axvline(0, color="grey", lw=0.4, ls="--")

handles = [mpatches.Patch(color=c, label=str(y)) for y, c in YEAR_PALETTE.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.03), frameon=False, title="Sequencing Year")
plt.tight_layout()
out_fig = os.path.join(FIG_DIR, "pca_k1_before_after.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
print(f"Saved -> {out_fig}")
plt.close()

print("\nDone.")
print(f"  Input samples : {k1.shape[1]}")
print(f"  Genes         : {k1_corrected.shape[0]:,}")
print(f"  Year batches  : {dict(sorted(yr_counts.items()))}")
