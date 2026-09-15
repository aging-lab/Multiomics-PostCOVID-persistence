"""
CpG methylation matrix builder.

Input : per-sample pair_merged.methyl_cpg_min.tsv files (AC / RC / HC)

Filters (per study design):
  - Coverage >= 10× per CpG per sample
  - CpG present in >= 90% of all samples (after excluding QC-failed samples)
  - Exclude: U10K-00446 (QC failed — n_cpg_cov10 z=-8.8)

Output:
  data/cpg_matrix_beta.npz     — β-value matrix + row/col index (compressed numpy)
  data/cpg_matrix_mval.npz     — M-value matrix + row/col index
  data/cpg_positions.tsv       — CpG position index (chr, start, cpg_id)
  data/cpg_sample_info.tsv     — sample metadata aligned to matrix columns
  figures/qc_coverage_hist.png — distribution of per-sample covered CpG count after filter

M-value = log2(β / (1 − β)), clipped to avoid ±Inf (β clipped to [0.001, 0.999]).
β-value = freqC / 100.

Strategy (memory-efficient):
  Pass 1: for each sample, record which (chr, start) positions pass cov>=10.
           Count occurrences across all samples → find positions in >=90%.
  Pass 2: for each sample, load only the retained positions → build matrix.
"""

import os, re
import numpy as np
import pandas as pd

METHYL_ROOT = "/BiO/Research/Infectomics/LongCOVID/Resources/Methyl"
BASE   = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA   = os.path.join(BASE, "data")
FIGS   = os.path.join(BASE, "figures")
os.makedirs(DATA, exist_ok=True); os.makedirs(FIGS, exist_ok=True)

MIN_COV        = 10
MIN_SAMPLE_FRAC= 0.90
EXCLUDE        = {"U10K-00446"}    # QC failed

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = {"AC": "#E63946", "RC": "#F4A261", "HC": "#457B9D"}

# ── Collect samples ───────────────────────────────────────────────────────────
def collect(subdir, group):
    rows = []
    base = os.path.join(METHYL_ROOT, subdir)
    for sid in sorted(os.listdir(base)):
        if sid in EXCLUDE:
            continue
        tsv = os.path.join(base, sid, f"{sid}.pair_merged.methyl_cpg_min.tsv")
        if os.path.isfile(tsv):
            rows.append({"SampleID": sid, "Group": group, "FilePath": tsv})
    return rows

samples = (collect("Acute_phase",    "AC") +
           collect("Recovered_phase","RC") +
           collect("Healthy_controls","HC"))
n = len(samples)
print(f"Samples: {n}  "
      f"(AC={sum(1 for s in samples if s['Group']=='AC')}, "
      f"RC={sum(1 for s in samples if s['Group']=='RC')}, "
      f"HC={sum(1 for s in samples if s['Group']=='HC')})")
print(f"Excluded: {EXCLUDE}")
print(f"Presence threshold: >= {MIN_SAMPLE_FRAC*100:.0f}% = {int(np.ceil(MIN_SAMPLE_FRAC*n))} samples")

# ── Pass 1: count position occurrences ───────────────────────────────────────
print("\nPass 1: counting covered positions per sample...")
pos_count = {}   # (chr, start) → n_samples_covered
per_sample_n = []   # n covered positions per sample (for QC plot)

for i, s in enumerate(samples):
    if (i+1) % 100 == 0:
        print(f"  {i+1}/{n}", flush=True)
    covered = set()
    with open(s["FilePath"]) as f:
        next(f)
        for line in f:
            p = line.split("\t")
            if float(p[3]) >= MIN_COV:
                covered.add((p[0], int(p[1])))
    for pos in covered:
        pos_count[pos] = pos_count.get(pos, 0) + 1
    per_sample_n.append(len(covered))

min_n = int(np.ceil(MIN_SAMPLE_FRAC * n))
keep_pos = sorted((chrom, start) for (chrom, start), cnt in pos_count.items()
                  if cnt >= min_n)
print(f"  CpGs passing >= {MIN_SAMPLE_FRAC*100:.0f}% filter: {len(keep_pos):,}")
del pos_count

# ── Save CpG position index ───────────────────────────────────────────────────
chrom_order = ([f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"])
cpg_df = pd.DataFrame(keep_pos, columns=["chr", "start"])
cpg_df["cpg_id"] = cpg_df["chr"] + "_" + cpg_df["start"].astype(str)
cpg_df.to_csv(os.path.join(DATA, "cpg_positions.tsv"), sep="\t", index=False)
print(f"  CpG position index saved ({len(cpg_df):,} rows)")

pos_idx = {pos: i for i, pos in enumerate(keep_pos)}

# ── Pass 2: build matrix ──────────────────────────────────────────────────────
print("\nPass 2: building β-value matrix...")
n_cpg = len(keep_pos)
beta_mat = np.full((n_cpg, n), np.nan, dtype=np.float32)

for col_i, s in enumerate(samples):
    if (col_i+1) % 100 == 0:
        print(f"  {col_i+1}/{n}", flush=True)
    with open(s["FilePath"]) as f:
        next(f)
        for line in f:
            p = line.split("\t")
            pos = (p[0], int(p[1]))
            if pos not in pos_idx:
                continue
            if float(p[3]) >= MIN_COV:
                beta_mat[pos_idx[pos], col_i] = float(p[4]) / 100.0

print(f"  NaN rate: {np.isnan(beta_mat).mean()*100:.2f}%")

# ── M-value conversion ────────────────────────────────────────────────────────
beta_clipped = np.clip(beta_mat, 0.001, 0.999)
mval_mat = np.log2(beta_clipped / (1 - beta_clipped)).astype(np.float32)
mval_mat[np.isnan(beta_mat)] = np.nan

# ── Save matrices ─────────────────────────────────────────────────────────────
sample_ids = np.array([s["SampleID"] for s in samples])
cpg_ids    = cpg_df["cpg_id"].values

np.savez_compressed(os.path.join(DATA, "cpg_matrix_beta.npz"),
                    matrix=beta_mat, cpg_ids=cpg_ids, sample_ids=sample_ids)
np.savez_compressed(os.path.join(DATA, "cpg_matrix_mval.npz"),
                    matrix=mval_mat, cpg_ids=cpg_ids, sample_ids=sample_ids)
print(f"\nSaved β-value matrix  → {DATA}/cpg_matrix_beta.npz")
print(f"Saved M-value matrix  → {DATA}/cpg_matrix_mval.npz")

# Sample info aligned to matrix
sample_meta = pd.DataFrame(samples)[["SampleID","Group"]].copy()
sample_meta.to_csv(os.path.join(DATA, "cpg_sample_info.tsv"), sep="\t", index=False)

# ── QC plot: covered CpG count per sample ─────────────────────────────────────
groups = [s["Group"] for s in samples]
fig, ax = plt.subplots(figsize=(8, 4))
for grp, color in PALETTE.items():
    vals = [v for v, g in zip(per_sample_n, groups) if g == grp]
    ax.hist(vals, bins=30, alpha=0.6, color=color, label=grp, density=True)
ax.axvline(min_n, color="black", lw=1, ls="--", label=f"90% threshold ({min_n} samples)")
ax.set_xlabel("CpGs with coverage ≥ 10 per sample")
ax.set_ylabel("Density")
ax.set_title("Per-sample covered CpG count (Pass 1)")
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIGS, "qc_coverage_hist.png"), dpi=150, bbox_inches="tight")
print(f"Saved → {FIGS}/qc_coverage_hist.png")

print(f"\nFinal matrix: {n_cpg:,} CpGs × {n} samples")
print("Done.")
