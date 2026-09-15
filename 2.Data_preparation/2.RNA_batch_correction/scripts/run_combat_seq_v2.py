"""
RNA-seq batch correction v2 — ComBat-seq (cross-batch, HC only).

Changes from v1:
  - Korea10K_1 input: raw counts  →  Year-corrected counts
    (2.1.K1_subcorrection/data/k1_corrected_counts.tsv)
    Pre-corrected within K1 for 2017/2018/2019 sequencing year batch effect.
  - K1 WG duplicates (25 numeric ID samples) already excluded in sub-correction.
  - Everything else identical to v1.

Strategy:
  1. Load AC, RC, HC-HealthyCtrl, HC-Korea10K_2, HC-WelfareGenome from raw matrices.
     Load HC-Korea10K_1 from pre-corrected k1_corrected_counts.tsv.
  2. Inner-join on common genes.
  3. Deduplicate HC samples (Korea10K_1 ↔ WelfareGenome overlap).
  4. Low-expression filter (row sum > 10).
  5. ComBat-seq on HC sub-batches (HC-HealthyCtrl / HC-Korea10K_1 / HC-Korea10K_2 / HC-WelfareGenome).
  6. Merge corrected HC + AC + RC.
  7. Save + PCA before/after.

Outputs:
  data/rna_combat_corrected_v2.tsv
  data/sample_info_v2.tsv
  figures/pca_before_after_v2.png
  figures/pca_scree_comparison_v2.png
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

RNA_ROOT   = "/BiO/Research/Infectomics/LongCOVID/Resources/RNA"
BASE_DIR   = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
K1_CORRECTED = os.path.normpath(os.path.join(
    BASE_DIR, "..", "2.1.K1_subcorrection", "data", "k1_corrected_counts.tsv"))
DATA_DIR   = os.path.join(BASE_DIR, "data")
FIG_DIR    = os.path.join(BASE_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

RAW_BATCHES = {
    "AC":               os.path.join(RNA_ROOT, "Acute_phase", "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
    "RC":               os.path.join(RNA_ROOT, "Recovered_phase", "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
    "HC-HealthyCtrl":   os.path.join(RNA_ROOT, "Healthy_controls", "Korea10K",
                                     "HealhyControls", "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
    "HC-Korea10K_2":    os.path.join(RNA_ROOT, "Healthy_controls", "Korea10K",
                                     "Korea10K_2", "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
    "HC-WelfareGenome": os.path.join(RNA_ROOT, "Healthy_controls", "WelfareGenome",
                                     "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
}

BATCH_TO_GROUP = {b: ("HC" if b.startswith("HC") else b) for b in list(RAW_BATCHES) + ["HC-Korea10K_1"]}
HC_BATCHES     = ["HC-HealthyCtrl", "HC-Korea10K_1", "HC-Korea10K_2", "HC-WelfareGenome"]

PALETTE_BATCH = {
    "AC":               "#E63946",
    "RC":               "#F4A261",
    "HC-HealthyCtrl":   "#2A9D8F",
    "HC-Korea10K_1":    "#457B9D",
    "HC-Korea10K_2":    "#1D3557",
    "HC-WelfareGenome": "#A8DADC",
}

# ── 1. Load matrices ──────────────────────────────────────────────────────────
print("Loading expression matrices...")
dfs_prefixed = {}

for batch, fpath in RAW_BATCHES.items():
    df = pd.read_csv(fpath, sep="\t", index_col=0)
    df.columns = [f"{batch}__{c}" for c in df.columns]
    dfs_prefixed[batch] = df
    print(f"  {batch}: {df.shape[1]} samples (raw)")

# K1: load pre-corrected matrix
print(f"  HC-Korea10K_1: loading pre-corrected from {K1_CORRECTED}")
k1 = pd.read_csv(K1_CORRECTED, sep="\t", index_col=0)
k1.columns = [f"HC-Korea10K_1__{c}" for c in k1.columns]
dfs_prefixed["HC-Korea10K_1"] = k1
print(f"  HC-Korea10K_1: {k1.shape[1]} samples (year-corrected)")

# ── 2. Inner join on common genes ─────────────────────────────────────────────
all_batches_order = ["AC", "RC", "HC-HealthyCtrl", "HC-Korea10K_1", "HC-Korea10K_2", "HC-WelfareGenome"]
merged_full = pd.concat([dfs_prefixed[b] for b in all_batches_order], axis=1, join="inner")
print(f"\nMerged: {merged_full.shape[0]:,} genes × {merged_full.shape[1]} samples")

# ── 3. Build per-column metadata, deduplicate HC ──────────────────────────────
col_meta = []
for batch in all_batches_order:
    for col in dfs_prefixed[batch].columns:
        orig_id = col.split("__", 1)[1]
        col_meta.append({"prefixed": col, "orig_id": orig_id,
                          "batch": batch, "group": BATCH_TO_GROUP[batch]})

seen_hc = set()
keep_flags = []
for m in col_meta:
    if m["batch"] in HC_BATCHES:
        if m["orig_id"] in seen_hc:
            keep_flags.append(False)
        else:
            seen_hc.add(m["orig_id"])
            keep_flags.append(True)
    else:
        keep_flags.append(True)

col_meta_kept = [m for m, k in zip(col_meta, keep_flags) if k]
keep_cols     = [m["prefixed"] for m in col_meta_kept]
merged        = merged_full[keep_cols]
n_dropped     = sum(1 for k in keep_flags if not k)
print(f"Dropped {n_dropped} duplicate HC samples → {merged.shape[1]} unique samples")

# ── 4. Low-expression filter ─────────────────────────────────────────────────
merged = merged.loc[merged.sum(axis=1) > 10]
merged = merged.round().astype(int)
print(f"After low-expression filter: {merged.shape[0]:,} genes")

# ── 5. ComBat-seq on HC sub-batches ───────────────────────────────────────────
hc_meta  = [m for m in col_meta_kept if m["batch"] in HC_BATCHES]
hc_cols  = [m["prefixed"] for m in hc_meta]
hc_batch = [m["batch"] for m in hc_meta]

hc_counts = merged[hc_cols]
print(f"\nHC samples for ComBat-seq: {hc_counts.shape[1]}")
for b in HC_BATCHES:
    print(f"  {b}: {hc_batch.count(b)}")

print("\nRunning ComBat-seq (v2)...")
hc_corrected = pycombat_seq(counts=hc_counts, batch=hc_batch)
hc_corrected.columns = [c.split("__", 1)[1] for c in hc_corrected.columns]
print(f"  Done. Shape: {hc_corrected.shape}")

# ── 6. Merge corrected HC + AC + RC ───────────────────────────────────────────
non_hc_meta = [m for m in col_meta_kept if m["batch"] not in HC_BATCHES]
non_hc      = merged[[m["prefixed"] for m in non_hc_meta]].copy()
non_hc.columns = [c.split("__", 1)[1] for c in non_hc.columns]

corrected = pd.concat([non_hc, hc_corrected], axis=1)
print(f"Final matrix: {corrected.shape}")

# ── 7. Save ───────────────────────────────────────────────────────────────────
corrected.to_csv(os.path.join(DATA_DIR, "rna_combat_corrected_v2.tsv"), sep="\t")
print(f"Saved -> data/rna_combat_corrected_v2.tsv")

info_rows = [{"SampleID": m["orig_id"], "OrigBatch": m["batch"],
               "Group": m["group"], "FinalBatch": "HC" if m["batch"] in HC_BATCHES else m["batch"]}
             for m in col_meta_kept]
pd.DataFrame(info_rows).to_csv(os.path.join(DATA_DIR, "sample_info_v2.tsv"), sep="\t", index=False)
print(f"Saved -> data/sample_info_v2.tsv ({len(info_rows)} rows)")

# ── 8. PCA before / after ─────────────────────────────────────────────────────
def run_pca(mat_df, n_top=5000, n_comp=10):
    log = np.log2(mat_df.values.astype(float) + 1)
    log_df = pd.DataFrame(log, index=mat_df.index, columns=mat_df.columns)
    log_df = log_df.loc[log_df.mean(axis=1) > 0.5]
    top = log_df.var(axis=1).nlargest(n_top).index
    X = log_df.loc[top].T.values
    X -= X.mean(axis=0)
    pca = PCA(n_components=n_comp, random_state=42)
    return pca.fit_transform(X), pca.explained_variance_ratio_ * 100

batch_labels = np.array([m["batch"] for m in col_meta_kept])

print("\nRunning PCA (before)...")
pcs_before, var_before = run_pca(merged)
print(f"  PC1={var_before[0]:.1f}%  PC2={var_before[1]:.1f}%")

print("Running PCA (after)...")
pcs_after, var_after = run_pca(corrected)
print(f"  PC1={var_after[0]:.1f}%  PC2={var_after[1]:.1f}%")

# ── 9. Plot ───────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("RNA-seq Batch Correction v2: Before vs After\n"
             "(K1 year-corrected input → cross-batch HC ComBat-seq)",
             fontsize=13, y=1.01)

for col_i, (pcs, var_exp, title) in enumerate([
        (pcs_before, var_before, "Before ComBat-seq (v2)"),
        (pcs_after,  var_after,  "After ComBat-seq (v2)"),
]):
    for row_i, (pcx, pcy) in enumerate([(0, 1), (1, 2)]):
        ax = axes[row_i][col_i]
        for batch, color in PALETTE_BATCH.items():
            m = batch_labels == batch
            if m.sum() == 0:
                continue
            ax.scatter(pcs[m, pcx], pcs[m, pcy],
                       c=color, s=12, alpha=0.7, linewidths=0, label=batch)
        ax.set_xlabel(f"PC{pcx+1} ({var_exp[pcx]:.1f}%)", fontsize=10)
        ax.set_ylabel(f"PC{pcy+1} ({var_exp[pcy]:.1f}%)", fontsize=10)
        ax.set_title(f"{title} — PC{pcx+1} vs PC{pcy+1}", fontsize=11)
        ax.axhline(0, color="grey", lw=0.4, ls="--")
        ax.axvline(0, color="grey", lw=0.4, ls="--")

handles = [mpatches.Patch(color=c, label=b) for b, c in PALETTE_BATCH.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10,
           bbox_to_anchor=(0.5, -0.04), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_before_after_v2.png"), dpi=150, bbox_inches="tight")
print(f"Saved -> figures/pca_before_after_v2.png")

# Scree comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
for ax2, (var_exp, title) in zip(axes2, [
        (var_before, "Before ComBat-seq (v2)"),
        (var_after,  "After ComBat-seq (v2)"),
]):
    ax2.bar(range(1, 11), var_exp[:10], color="#457B9D", edgecolor="white")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title(title)
    ax2.set_xticks(range(1, 11))
plt.suptitle("RNA-seq PCA Scree Comparison (v2)", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "pca_scree_comparison_v2.png"), dpi=150, bbox_inches="tight")
print(f"Saved -> figures/pca_scree_comparison_v2.png")
print("\nAll done.")
