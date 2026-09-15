"""
Per-sample methylation QC.

Metrics computed per sample (from pair_merged.methyl_cpg_min.tsv):
  n_cpg_total     : total CpG sites in file (genome-wide breadth)
  n_cpg_cov10     : CpGs with coverage >= 10
  pct_cpg_cov10   : n_cpg_cov10 / n_cpg_total * 100
  mean_cov_depth  : mean coverage depth among covered CpGs
  global_meth_pct : mean freqC (%) among covered CpGs

Speed: samples every SAMPLE_EVERY-th line; estimates scaled back to full genome.
Outlier flagging: samples > N_SD standard deviations from group mean on any metric.

Outputs:
  data/methyl_qc_metrics.tsv
  figures/qc_violin.png       — violin plots of 3 key metrics per group
  figures/qc_scatter.png      — n_cpg_cov10 vs global_meth (outliers labelled)
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

METHYL_ROOT  = "/BiO/Research/Infectomics/LongCOVID/Resources/Methyl"
BASE_DIR     = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
DATA_DIR     = os.path.join(BASE_DIR, "data")
FIG_DIR      = os.path.join(BASE_DIR, "figures")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR,  exist_ok=True)

MIN_COV      = 10
SAMPLE_EVERY = 10    # read 1 in every N lines (10% sample); counts scaled ×N
N_SD_OUTLIER = 3     # flag samples beyond ±N_SD from group mean

PALETTE = {"AC": "#E63946", "RC": "#F4A261", "HC": "#457B9D"}

# ── Collect samples ───────────────────────────────────────────────────────────
def collect(subdir, group):
    records = []
    base = os.path.join(METHYL_ROOT, subdir)
    for sid in sorted(os.listdir(base)):
        tsv = os.path.join(base, sid, f"{sid}.pair_merged.methyl_cpg_min.tsv")
        if os.path.isfile(tsv):
            records.append({"SampleID": sid, "Group": group, "FilePath": tsv})
    return records

all_samples = (collect("Acute_phase",    "AC") +
               collect("Recovered_phase","RC") +
               collect("Healthy_controls","HC"))
print(f"Total samples: {len(all_samples)}  (AC={sum(1 for s in all_samples if s['Group']=='AC')}, "
      f"RC={sum(1 for s in all_samples if s['Group']=='RC')}, "
      f"HC={sum(1 for s in all_samples if s['Group']=='HC')})")

# ── Compute per-sample QC stats ───────────────────────────────────────────────
def compute_qc(fpath, sample_every=SAMPLE_EVERY, min_cov=MIN_COV):
    n_total = 0
    n_cov   = 0
    sum_cov = 0.0
    sum_meth= 0.0
    with open(fpath) as f:
        next(f)  # skip header
        for i, line in enumerate(f):
            if i % sample_every != 0:
                continue
            parts = line.split("\t")
            cov = float(parts[3])
            n_total += 1
            if cov >= min_cov:
                n_cov   += 1
                sum_cov += cov
                sum_meth+= float(parts[4])
    # Scale back to full-genome estimates
    return {
        "n_cpg_total":    n_total   * sample_every,
        "n_cpg_cov10":    n_cov     * sample_every,
        "pct_cpg_cov10":  (n_cov / n_total * 100) if n_total > 0 else 0,
        "mean_cov_depth": (sum_cov / n_cov)  if n_cov > 0 else 0,
        "global_meth_pct":(sum_meth / n_cov) if n_cov > 0 else 0,
    }

print(f"\nComputing QC stats (1/{SAMPLE_EVERY} line sampling)...")
rows = []
for i, s in enumerate(all_samples):
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(all_samples)}", flush=True)
    qc = compute_qc(s["FilePath"])
    rows.append({**s, **qc})

df = pd.DataFrame(rows).drop(columns=["FilePath"])
df.to_csv(os.path.join(DATA_DIR, "methyl_qc_metrics.tsv"), sep="\t", index=False)
print(f"QC metrics saved ({len(df)} rows)")

# ── Outlier detection (per group, per metric) ─────────────────────────────────
metrics = ["n_cpg_cov10", "pct_cpg_cov10", "mean_cov_depth", "global_meth_pct"]
df["is_outlier"] = False
df["outlier_reason"] = ""

for grp in ["AC", "RC", "HC"]:
    mask = df["Group"] == grp
    for m in metrics:
        vals = df.loc[mask, m]
        mu, sd = vals.mean(), vals.std()
        if sd == 0:
            continue
        z = (df.loc[mask, m] - mu) / sd
        out = z.abs() > N_SD_OUTLIER
        df.loc[mask & out, "is_outlier"] = True
        for idx in df.loc[mask & out].index:
            reason = f"{m}(z={z[idx]:.1f})"
            df.at[idx, "outlier_reason"] += (", " if df.at[idx, "outlier_reason"] else "") + reason

outliers = df[df["is_outlier"]]
print(f"\nOutliers flagged (>{N_SD_OUTLIER} SD from group mean): {len(outliers)}")
for _, row in outliers.iterrows():
    print(f"  {row['SampleID']} [{row['Group']}] — {row['outlier_reason']}")

# Save flagged table
df.to_csv(os.path.join(DATA_DIR, "methyl_qc_metrics.tsv"), sep="\t", index=False)

# ── Plot 1: Violin plots ──────────────────────────────────────────────────────
plot_metrics = [
    ("n_cpg_cov10",    "CpGs with coverage ≥ 10 (×10⁶)", 1e6),
    ("mean_cov_depth", "Mean coverage depth (×)", 1),
    ("global_meth_pct","Global methylation (%)", 1),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Methylation Sample QC — Per-group Distribution", fontsize=13)

for ax, (metric, ylabel, scale) in zip(axes, plot_metrics):
    data_by_group = [df.loc[df["Group"] == g, metric].values / scale for g in ["AC", "RC", "HC"]]
    vp = ax.violinplot(data_by_group, positions=[1, 2, 3], showmedians=True, showextrema=True)
    for pc, color in zip(vp["bodies"], PALETTE.values()):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    # Overlay outlier points
    for xi, grp in enumerate(["AC", "RC", "HC"], 1):
        out_vals = df.loc[(df["Group"] == grp) & df["is_outlier"], metric].values / scale
        if len(out_vals):
            ax.scatter([xi] * len(out_vals), out_vals, color="black", s=40, zorder=5,
                       marker="x", linewidths=1.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["AC", "RC", "HC"])
    ax.set_ylabel(ylabel)
    ax.set_title(metric)

handles = [mpatches.Patch(color=c, label=g, alpha=0.7) for g, c in PALETTE.items()]
handles.append(plt.Line2D([0], [0], marker="x", color="black", linestyle="",
                           markersize=8, markeredgewidth=1.5, label=f"Outlier (>{N_SD_OUTLIER}SD)"))
fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=10,
           bbox_to_anchor=(0.5, -0.08), frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "qc_violin.png"), dpi=150, bbox_inches="tight")
print(f"\nSaved → {os.path.join(FIG_DIR, 'qc_violin.png')}")

# ── Plot 2: Scatter n_cpg_cov10 vs global_meth ───────────────────────────────
fig2, ax2 = plt.subplots(figsize=(9, 6))
for grp, color in PALETTE.items():
    sub = df[df["Group"] == grp]
    norm = sub[~sub["is_outlier"]]
    out  = sub[sub["is_outlier"]]
    ax2.scatter(norm["n_cpg_cov10"] / 1e6, norm["global_meth_pct"],
                c=color, s=15, alpha=0.6, linewidths=0, label=grp)
    if len(out):
        ax2.scatter(out["n_cpg_cov10"] / 1e6, out["global_meth_pct"],
                    c=color, s=60, alpha=1, linewidths=1.2,
                    edgecolors="black", marker="D", zorder=5)
        for _, row in out.iterrows():
            ax2.annotate(row["SampleID"], (row["n_cpg_cov10"] / 1e6, row["global_meth_pct"]),
                         fontsize=6, ha="left", va="bottom",
                         xytext=(3, 3), textcoords="offset points")

ax2.set_xlabel("CpGs with coverage ≥ 10 (×10⁶)", fontsize=11)
ax2.set_ylabel("Global methylation (%)", fontsize=11)
ax2.set_title("Methylation QC: Coverage breadth vs Global methylation level", fontsize=11)
handles2 = [mpatches.Patch(color=c, label=g, alpha=0.7) for g, c in PALETTE.items()]
handles2.append(plt.Line2D([0], [0], marker="D", color="black", linestyle="",
                             markersize=7, label=f"Outlier (>{N_SD_OUTLIER}SD)"))
ax2.legend(handles=handles2, fontsize=10, frameon=False)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "qc_scatter.png"), dpi=150, bbox_inches="tight")
print(f"Saved → {os.path.join(FIG_DIR, 'qc_scatter.png')}")

# ── Summary stats ─────────────────────────────────────────────────────────────
print("\n" + "─" * 55)
print("Group-level summary (median):")
print(df.groupby("Group")[metrics].median().round(2).to_string())
