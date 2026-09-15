"""
RNA-seq batch correction using ComBat-seq (pycombat_seq).

Problem context:
  - AC and RC are each a single batch → group is confounded with batch for these two.
    ComBat-seq cannot correct AC/RC independently.
  - HC spans 4 sub-batches (HealthyCtrl, Korea10K_1, Korea10K_2, WelfareGenome)
    with no biological confounding → can be corrected.
  - 26 samples overlap between Korea10K_1 and WelfareGenome (same biological sample,
    different sequencing runs). These are deduplicated before correction.

Strategy:
  1. Load all 6 batch matrices. Add batch-prefix to column names for uniqueness.
  2. Deduplicate: for samples in multiple HC batches, keep only one (first occurrence).
  3. Filter low-expression genes (row sum > 10 across all samples).
  4. Run pycombat_seq on HC sub-batches only (all group=HC → no confounding).
  5. Merge corrected HC with AC + RC into one matrix.
  6. Save corrected matrix + sample info.
  7. PCA before vs after — 2×2 panel.

Outputs:
  data/rna_combat_corrected.tsv   — merged count matrix (genes × samples), HC corrected
  data/sample_info.tsv            — SampleID, OrigBatch, Group, FinalBatch, Deduplicated
  figures/pca_before_after.png
  figures/pca_scree_comparison.png
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

RNA_ROOT = "/BiO/Research/Infectomics/LongCOVID/Resources/RNA"
BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
FIG_DIR  = os.path.join(BASE_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

BATCHES = {
    "AC": os.path.join(RNA_ROOT, "Acute_phase", "4_expmtx",
                       "expression_matrix_genes.results_expected_count.tsv"),
    "RC": os.path.join(RNA_ROOT, "Recovered_phase", "4_expmtx",
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
    "HC-WelfareGenome": os.path.join(RNA_ROOT, "Healthy_controls", "WelfareGenome",
                                     "4_expmtx",
                                     "expression_matrix_genes.results_expected_count.tsv"),
}

BATCH_TO_GROUP = {b: ("HC" if b.startswith("HC") else b) for b in BATCHES}
HC_BATCHES     = [b for b in BATCHES if b.startswith("HC")]

PALETTE_BATCH = {
    "AC":               "#E63946",
    "RC":               "#F4A261",
    "HC-HealthyCtrl":   "#2A9D8F",
    "HC-Korea10K_1":    "#457B9D",
    "HC-Korea10K_2":    "#1D3557",
    "HC-WelfareGenome": "#A8DADC",
}

# ── 1. Load matrices with batch-prefixed column names ────────────────────────
print("Loading expression matrices...")
dfs_prefixed = {}           # batch → DataFrame (genes × samples, prefixed cols)
for batch, fpath in BATCHES.items():
    df = pd.read_csv(fpath, sep="\t", index_col=0)
    df.columns = [f"{batch}__{c}" for c in df.columns]   # unique col names
    dfs_prefixed[batch] = df
    print(f"  {batch}: {df.shape[1]} samples")

# Inner join on common genes
print("\nMerging all batches (inner join on genes)...")
all_dfs  = list(dfs_prefixed.values())
merged_full = pd.concat(all_dfs, axis=1, join="inner")
print(f"  Common genes: {merged_full.shape[0]}, total columns: {merged_full.shape[1]}")

# ── 2. Deduplicate HC samples (same raw sample ID in multiple HC batches) ────
# Build per-column metadata (strip prefix to recover original sample ID)
col_meta = []  # list of dicts: prefixed_col, orig_id, batch, group
for batch in BATCHES:
    for col in dfs_prefixed[batch].columns:
        orig_id = col.split("__", 1)[1]
        col_meta.append({"prefixed": col, "orig_id": orig_id,
                          "batch": batch, "group": BATCH_TO_GROUP[batch]})

# Within HC batches only: keep first occurrence of each orig_id
seen_hc = set()
keep_flags = []
for m in col_meta:
    if m["batch"] in HC_BATCHES:
        if m["orig_id"] in seen_hc:
            keep_flags.append(False)   # duplicate — drop
        else:
            seen_hc.add(m["orig_id"])
            keep_flags.append(True)
    else:
        keep_flags.append(True)        # AC / RC — always keep

col_meta_kept = [m for m, k in zip(col_meta, keep_flags) if k]
keep_cols     = [m["prefixed"] for m in col_meta_kept]
merged        = merged_full[keep_cols]

n_dropped = sum(1 for k in keep_flags if not k)
print(f"  Dropped {n_dropped} duplicate HC samples → {merged.shape[1]} unique columns")

# ── 3. Low-expression filter ─────────────────────────────────────────────────
row_sum = merged.sum(axis=1)
merged  = merged.loc[row_sum > 10]
print(f"  After low-expression filter (sum > 10): {merged.shape[0]} genes")

# Round to integer counts
merged = merged.round().astype(int)

# ── 4. Build HC-only matrix for ComBat-seq ────────────────────────────────────
hc_meta  = [m for m in col_meta_kept if m["batch"] in HC_BATCHES]
hc_cols  = [m["prefixed"] for m in hc_meta]
hc_batch = [m["batch"] for m in hc_meta]

hc_counts = merged[hc_cols]
print(f"\nHC samples for ComBat-seq: {hc_counts.shape[1]}")
for b in HC_BATCHES:
    print(f"  {b}: {hc_batch.count(b)}")

# ── 5. Run ComBat-seq on HC sub-batches ────────────────────────────────────────
print("\nRunning ComBat-seq on HC sub-batches...")
hc_corrected = pycombat_seq(counts=hc_counts, batch=hc_batch)
print(f"  Done. Output shape: {hc_corrected.shape}")

# Strip batch prefix from corrected HC column names → original sample IDs
hc_corrected.columns = [c.split("__", 1)[1] for c in hc_corrected.columns]

# ── 6. Merge corrected HC + AC + RC ───────────────────────────────────────────
non_hc_meta = [m for m in col_meta_kept if m["batch"] not in HC_BATCHES]
non_hc_cols = [m["prefixed"] for m in non_hc_meta]
non_hc      = merged[non_hc_cols].copy()
non_hc.columns = [c.split("__", 1)[1] for c in non_hc.columns]

corrected = pd.concat([non_hc, hc_corrected], axis=1)
print(f"Final merged matrix: {corrected.shape}")

# ── 7. Save corrected matrix ──────────────────────────────────────────────────
out_path = os.path.join(DATA_DIR, "rna_combat_corrected.tsv")
corrected.to_csv(out_path, sep="\t")
print(f"Corrected matrix saved → {out_path}")

# ── 8. Save sample info ───────────────────────────────────────────────────────
info_rows = []
for m in col_meta_kept:
    info_rows.append({
        "SampleID":   m["orig_id"],
        "OrigBatch":  m["batch"],
        "Group":      m["group"],
        "FinalBatch": "HC" if m["batch"] in HC_BATCHES else m["batch"],
    })
pd.DataFrame(info_rows).to_csv(os.path.join(DATA_DIR, "sample_info.tsv"),
                                sep="\t", index=False)
print(f"Sample info saved ({len(info_rows)} rows)")

# ── 9. PCA before vs after ────────────────────────────────────────────────────
def run_pca(mat_df, n_top=5000, n_components=10):
    log_mat = np.log2(mat_df.values.astype(float) + 1)
    log_df  = pd.DataFrame(log_mat, index=mat_df.index, columns=mat_df.columns)
    log_df  = log_df.loc[log_df.mean(axis=1) > 0.5]
    top_idx = log_df.var(axis=1).nlargest(n_top).index
    X = log_df.loc[top_idx].T.values
    X = X - X.mean(axis=0)
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(X)
    return pcs, pca.explained_variance_ratio_ * 100

# For "before" PCA, use the same deduplicated/filtered merged matrix (raw counts)
batch_labels = [m["batch"] for m in col_meta_kept]

print("\nRunning PCA on raw (before correction)...")
pcs_before, var_before = run_pca(merged)
print(f"  PC1: {var_before[0]:.1f}%  PC2: {var_before[1]:.1f}%")

print("Running PCA on corrected matrix...")
pcs_after, var_after = run_pca(corrected)
print(f"  PC1: {var_after[0]:.1f}%  PC2: {var_after[1]:.1f}%")

labels = np.array(batch_labels)

# ── 10. Plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 14))
fig.suptitle("RNA-seq Batch Correction: Before vs After HC ComBat-seq\n"
             "(log2 count+1, top 5,000 variable genes)",
             fontsize=14, y=1.01)

for col_idx, (pcs, var_exp, title) in enumerate([
        (pcs_before, var_before, "Before ComBat-seq"),
        (pcs_after,  var_after,  "After HC ComBat-seq"),
]):
    for row_idx, (pcx, pcy) in enumerate([(0, 1), (1, 2)]):
        ax = axes[row_idx][col_idx]
        for batch, color in PALETTE_BATCH.items():
            mask = labels == batch
            if mask.sum() == 0:
                continue
            ax.scatter(pcs[mask, pcx], pcs[mask, pcy],
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
fig_path = os.path.join(FIG_DIR, "pca_before_after.png")
plt.savefig(fig_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {fig_path}")

# Scree comparison
fig2, axes2 = plt.subplots(1, 2, figsize=(12, 4))
for ax2, (var_exp, title) in zip(axes2, [
        (var_before, "Before ComBat-seq"),
        (var_after,  "After HC ComBat-seq"),
]):
    ax2.bar(range(1, 11), var_exp[:10], color="#457B9D", edgecolor="white")
    ax2.set_xlabel("Principal Component")
    ax2.set_ylabel("Explained Variance (%)")
    ax2.set_title(title)
    ax2.set_xticks(range(1, 11))
plt.suptitle("RNA-seq PCA Scree Comparison", fontsize=13)
plt.tight_layout()
scree_path = os.path.join(FIG_DIR, "pca_scree_comparison.png")
plt.savefig(scree_path, dpi=150, bbox_inches="tight")
print(f"Saved → {scree_path}")
print("\nAll done.")
