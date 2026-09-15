"""
Methylation batch-effect PCA.

Strategy (memory-efficient):
  1. Read per-sample CpG TSV files (chr, start, coverage, freqC).
  2. Coverage filter: keep CpGs with coverage >= MIN_COV per sample.
  3. Restrict to chr1 (fast proxy; ~200K CpGs, representative of genome-wide pattern).
  4. Collect positions present in >= MIN_SAMPLE_FRAC of samples.
  5. Build methylation matrix (samples × positions), NaN for missing.
  6. Select top 5,000 most variable positions (by std, ignoring NaN).
  7. Impute remaining NaN with column mean.
  8. PCA, plot coloured by Group (AC / RC / HC).

Outputs:
  figures/pca_methyl_batch.png
  figures/pca_methyl_scree.png
"""

import os
import glob
import re
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

METHYL_ROOT = "/BiO/Research/Infectomics/LongCOVID/Resources/Methyl"
OUT_DIR     = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                             "..", "figures"))
os.makedirs(OUT_DIR, exist_ok=True)

MIN_COV          = 10    # minimum read coverage per CpG per sample
MIN_SAMPLE_FRAC  = 0.70  # CpG must be covered in >=70% of samples
N_TOP_CPGS       = 5000  # top variable CpGs for PCA
CHROM            = "chr1"  # restrict to one chromosome for speed

PALETTE = {"AC": "#E63946", "RC": "#F4A261", "HC": "#457B9D"}

# ── Collect sample files ──────────────────────────────────────────────────────
def collect_samples(subdir, group):
    """Return list of (sample_id, file_path, group).
    Prefer pair_merged file (present in all samples); fall back to raw."""
    samples = []
    base = os.path.join(METHYL_ROOT, subdir)
    for sid in sorted(os.listdir(base)):
        tsv_pair = os.path.join(base, sid, f"{sid}.pair_merged.methyl_cpg_min.tsv")
        tsv_raw  = os.path.join(base, sid, f"{sid}.methyl_cpg_min.tsv")
        if os.path.isfile(tsv_pair):
            samples.append((sid, tsv_pair, group))
        elif os.path.isfile(tsv_raw):
            samples.append((sid, tsv_raw, group))
    return samples

all_samples = (collect_samples("Acute_phase",    "AC") +
               collect_samples("Recovered_phase", "RC") +
               collect_samples("Healthy_controls","HC"))

print(f"Total samples found: {len(all_samples)}")
for g in ["AC", "RC", "HC"]:
    n = sum(1 for _, _, grp in all_samples if grp == g)
    print(f"  {g}: {n}")

# ── Pass 1: collect covered positions per sample ──────────────────────────────
print(f"\nPass 1: reading {CHROM} CpGs (coverage >= {MIN_COV}) ...")
# position_counts[pos] = number of samples with coverage >= MIN_COV
position_counts = {}
sample_data = {}  # sid → dict of pos → freqC

for i, (sid, fpath, grp) in enumerate(all_samples):
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(all_samples)}", flush=True)
    pos_meth = {}
    with open(fpath) as f:
        next(f)  # skip header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if parts[0] != CHROM:
                continue
            cov = float(parts[3])
            if cov < MIN_COV:
                continue
            pos = int(parts[1])
            pos_meth[pos] = float(parts[4])
    # Update position coverage counts
    for pos in pos_meth:
        position_counts[pos] = position_counts.get(pos, 0) + 1
    sample_data[sid] = pos_meth

n_samples = len(all_samples)
min_samples = int(MIN_SAMPLE_FRAC * n_samples)
common_pos = sorted(pos for pos, cnt in position_counts.items() if cnt >= min_samples)
print(f"CpGs in >= {MIN_SAMPLE_FRAC*100:.0f}% of samples: {len(common_pos)}")

# ── Build matrix ──────────────────────────────────────────────────────────────
print("\nBuilding methylation matrix...")
common_pos_set = set(common_pos)
mat = np.full((n_samples, len(common_pos)), np.nan, dtype=np.float32)
pos_idx = {p: i for i, p in enumerate(common_pos)}

sample_ids = []
sample_groups = []
for row_i, (sid, fpath, grp) in enumerate(all_samples):
    sample_ids.append(sid)
    sample_groups.append(grp)
    for pos, val in sample_data[sid].items():
        if pos in pos_idx:
            mat[row_i, pos_idx[pos]] = val

del sample_data  # free memory

# ── Select top variable CpGs ──────────────────────────────────────────────────
col_std = np.nanstd(mat, axis=0)
top_idx = np.argsort(col_std)[::-1][:N_TOP_CPGS]
mat_top = mat[:, top_idx]
print(f"Selected top {N_TOP_CPGS} variable CpGs; "
      f"NaN rate in top set: {np.isnan(mat_top).mean()*100:.1f}%")

# Impute NaN with column mean
col_means = np.nanmean(mat_top, axis=0)
nan_mask  = np.isnan(mat_top)
mat_top[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

# Mean-center per CpG
mat_top -= mat_top.mean(axis=0)

# ── PCA ───────────────────────────────────────────────────────────────────────
print("Running PCA...")
pca = PCA(n_components=10, random_state=42)
pcs = pca.fit_transform(mat_top)
var_exp = pca.explained_variance_ratio_ * 100
print(f"  PC1: {var_exp[0]:.1f}%  PC2: {var_exp[1]:.1f}%  PC3: {var_exp[2]:.1f}%")

labels = np.array(sample_groups)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle(f"Methylation Batch Effect PCA\n"
             f"({CHROM}, coverage≥{MIN_COV}, top {N_TOP_CPGS} variable CpGs, "
             f"present in ≥{MIN_SAMPLE_FRAC*100:.0f}% samples)",
             fontsize=12, y=1.01)

for ax, (pcx, pcy) in zip(axes, [(0, 1), (1, 2)]):
    for grp, color in PALETTE.items():
        mask = labels == grp
        if mask.sum() == 0:
            continue
        ax.scatter(pcs[mask, pcx], pcs[mask, pcy],
                   c=color, label=grp, s=15, alpha=0.7, linewidths=0)
    ax.set_xlabel(f"PC{pcx+1} ({var_exp[pcx]:.1f}%)", fontsize=11)
    ax.set_ylabel(f"PC{pcy+1} ({var_exp[pcy]:.1f}%)", fontsize=11)
    ax.set_title(f"PC{pcx+1} vs PC{pcy+1}")
    ax.axhline(0, color="grey", lw=0.4, ls="--")
    ax.axvline(0, color="grey", lw=0.4, ls="--")

handles = [mpatches.Patch(color=c, label=g) for g, c in PALETTE.items()]
fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=11,
           bbox_to_anchor=(0.5, -0.06), frameon=False)

plt.tight_layout()
out_path = os.path.join(OUT_DIR, "pca_methyl_batch.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"\nSaved → {out_path}")

# Scree plot
fig2, ax2 = plt.subplots(figsize=(6, 4))
ax2.bar(range(1, 11), var_exp[:10], color="#457B9D", edgecolor="white")
ax2.set_xlabel("Principal Component")
ax2.set_ylabel("Explained Variance (%)")
ax2.set_title("Methylation PCA Scree Plot")
ax2.set_xticks(range(1, 11))
plt.tight_layout()
scree_path = os.path.join(OUT_DIR, "pca_methyl_scree.png")
plt.savefig(scree_path, dpi=150, bbox_inches="tight")
print(f"Saved → {scree_path}")
