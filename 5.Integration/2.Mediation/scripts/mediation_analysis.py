"""
Causal Mediation Analysis — delayed CpG → persistent gene expression

설계:
  Exposure (X):  Group binary (HC=0, RC=1)
  Mediator (M):  Delayed CpG M-value
  Outcome (Y):   Persistent gene logCPM
  Covariates:    Age, Sex, B, NK, CD4T, CD8T, Mono (methyl cell fractions)

대상 페어:
  COX8A   × chr11_64304866, chr11_64304843
  C4orf48 × chr4_2468860, chr4_2468855, chr4_2468842, chr4_2468846

방법: Baron & Kenny + Bootstrap ACME/ADE (n=1000)
  Step 1. Mediator model:  M ~ X + covariates
  Step 2. Outcome model:   Y ~ X + M + covariates
  ACME = X→M 계수 × M→Y|X 계수
  ADE  = X→Y|M 계수 (direct)
  Total = ACME + ADE
  Proportion mediated = ACME / Total
"""

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import scipy.sparse as sp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
warnings.filterwarnings("ignore")

BASE    = "/BiO/Research/Infectomics/LongCOVID/Analysis/5.Integration/2.Mediation"
METHYL  = "/BiO/Research/Infectomics/LongCOVID/Analysis/2.Data_preparation/5.Methyl_matrix"
RNA_DIR = "/BiO/Research/Infectomics/LongCOVID/Analysis/2.Data_preparation/4.RNA_normalization/data"
META_C  = "/BiO/Research/Infectomics/LongCOVID/Analysis/1.Metadata/COVID-19_sample_metadata.tsv"
META_H  = "/BiO/Research/Infectomics/LongCOVID/Analysis/1.Metadata/HC_sample_metadata.tsv"
CF_DIR  = "/BiO/Research/Infectomics/LongCOVID/Analysis/2.Data_preparation/6.Deconvolution/data"

N_BOOT  = 1000
SEED    = 42
rng     = np.random.default_rng(SEED)

# ── 1. 샘플 정의 (HC + RC, 공유 RNA+Methyl) ──────────────────────────────────
print("1. 샘플 로드 중...")
meta_c = pd.read_csv(META_C, sep="\t")
meta_h = pd.read_csv(META_H, sep="\t")
cpg_info = pd.read_csv(f"{METHYL}/data/cpg_sample_info.tsv", sep="\t")
rna_info = pd.read_csv(
    "/BiO/Research/Infectomics/LongCOVID/Analysis/2.Data_preparation/"
    "2.RNA_batch_correction/data/sample_info_v2.tsv", sep="\t")

shared = set(cpg_info["SampleID"]) & set(rna_info["SampleID"])

# HC + RC 공유 샘플
meta_all = pd.concat([
    meta_c[["SampleID","Group","Timepoint","Age","Sex"]],
    meta_h[["SampleID","Group","Age","Sex"]].assign(Timepoint=None)
], ignore_index=True)
meta_use = meta_all[
    meta_all["SampleID"].isin(shared) &
    meta_all["Group"].isin(["HC","RC"])
].copy()
meta_use["X"] = (meta_use["Group"] == "RC").astype(int)  # HC=0, RC=1
meta_use["Sex_bin"] = (meta_use["Sex"] == "M").astype(int)
meta_use["Age"] = pd.to_numeric(meta_use["Age"], errors="coerce")
print(f"   HC: {(meta_use['Group']=='HC').sum()}, RC: {(meta_use['Group']=='RC').sum()}")

# ── 2. CpG M-value 추출 ──────────────────────────────────────────────────────
print("2. CpG M-value 로드 중...")
npz = np.load(f"{METHYL}/data/cpg_matrix_mval.npz", allow_pickle=True)
cpg_mat    = npz["matrix"]           # (CpGs × samples)
cpg_ids    = npz["cpg_ids"]
cpg_sids   = npz["sample_ids"]

# 샘플 순서 인덱스
cpg_sid_list  = list(cpg_sids)
use_sids      = meta_use["SampleID"].tolist()
cpg_col_idx   = [cpg_sid_list.index(s) for s in use_sids if s in cpg_sid_list]
use_sids_filt = [s for s in use_sids if s in cpg_sid_list]
meta_use = meta_use[meta_use["SampleID"].isin(use_sids_filt)].set_index("SampleID").loc[use_sids_filt].reset_index()

TARGET_CPGS = {
    "chr11_64304866": "COX8A_CpG1",
    "chr11_64304843": "COX8A_CpG2",
    "chr4_2468860":   "C4orf48_CpG1",
    "chr4_2468855":   "C4orf48_CpG2",
    "chr4_2468842":   "C4orf48_CpG3",
    "chr4_2468846":   "C4orf48_CpG4",
}

cpg_data = {}
x_mask = meta_use["X"].values  # numpy array for safe boolean indexing
for cpg_id, label in TARGET_CPGS.items():
    row_idx = np.where(cpg_ids == cpg_id)[0][0]
    vals = cpg_mat[row_idx, :]  # extract full row first
    if sp.issparse(vals):
        vals = np.array(vals.todense()).flatten()
    else:
        vals = np.array(vals).flatten()
    vals = vals.astype(float)
    # select only our samples in correct order
    vals_use = np.array([vals[cpg_sid_list.index(s)] for s in use_sids_filt], dtype=float)
    vals_use[vals_use == 0] = np.nan  # 0 = no coverage → NaN
    cpg_data[cpg_id] = {"label": label, "mval": vals_use}
    print(f"   {label} ({cpg_id}): mean HC={np.nanmean(vals_use[x_mask==0]):.3f}, RC={np.nanmean(vals_use[x_mask==1]):.3f}")

# ── 3. RNA logCPM 로드 ───────────────────────────────────────────────────────
print("3. RNA logCPM 로드 중...")
TARGET_GENES = {
    "ENSG00000176340": "COX8A",
    "ENSG00000243449": "C4orf48",
}

rna = pd.read_csv(f"{RNA_DIR}/rna_logcpm_v2.tsv.gz", sep="\t", index_col=0)
rna.index = [g.split(".")[0].split("_")[0] for g in rna.index]

gene_data = {}
for ensg, name in TARGET_GENES.items():
    vals = rna.loc[ensg, use_sids_filt].values.astype(float)
    gene_data[ensg] = {"name": name, "expr": vals}
    print(f"   {name}: mean HC={vals[meta_use['X']==0].mean():.3f}, RC={vals[meta_use['X']==1].mean():.3f}")

# ── 4. 공변량 로드 ───────────────────────────────────────────────────────────
print("4. 공변량 로드 중...")
cf = pd.read_csv(f"{CF_DIR}/methyl_cell_fractions.tsv", sep="\t")
cf = cf[cf["SampleID"].isin(use_sids_filt)].set_index("SampleID").loc[use_sids_filt]
cf_cols = [c for c in cf.columns if c != "SampleID"]
print(f"   Cell fractions: {cf_cols}")

# 공변량 행렬 구성
cov = np.column_stack([
    meta_use["Age"].fillna(meta_use["Age"].median()).astype(float).values,
    meta_use["Sex_bin"].values,
    cf[cf_cols].values
])
cov_names = ["Age", "Sex"] + cf_cols

# ── 5. Bootstrap Mediation 함수 ──────────────────────────────────────────────
def mediation_bootstrap(X, M, Y, covariates, n_boot=1000, rng=None):
    """
    Bootstrap ACME/ADE 추정
    Returns: dict with point estimates + bootstrap CI
    """
    n = len(X)
    if rng is None:
        rng = np.random.default_rng(42)

    def point_estimate(X, M, Y, cov):
        # ensure float
        X = np.asarray(X, dtype=float)
        M = np.asarray(M, dtype=float)
        Y = np.asarray(Y, dtype=float)
        cov = np.asarray(cov, dtype=float)
        # Mediator model: M ~ X + cov
        Zm = add_constant(np.column_stack([X, cov]))
        med_fit = OLS(M, Zm).fit()
        alpha = med_fit.params[1]   # X → M

        # Outcome model: Y ~ X + M + cov
        Zy = add_constant(np.column_stack([X, M, cov]))
        out_fit = OLS(Y, Zy).fit()
        gamma = out_fit.params[2]   # M → Y | X
        delta = out_fit.params[1]   # X → Y | M (ADE)

        acme  = alpha * gamma
        ade   = delta
        total = acme + ade
        prop  = acme / total if abs(total) > 1e-10 else np.nan
        return acme, ade, total, prop, alpha, gamma

    # Point estimate
    acme0, ade0, tot0, prop0, a0, g0 = point_estimate(X, M, Y, covariates)

    # Bootstrap
    boot_acme, boot_ade, boot_tot, boot_prop = [], [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        Xb, Mb, Yb, Cb = X[idx], M[idx], Y[idx], covariates[idx]
        try:
            a, d, t, p, _, _ = point_estimate(Xb, Mb, Yb, Cb)
            boot_acme.append(a); boot_ade.append(d)
            boot_tot.append(t);  boot_prop.append(p)
        except Exception:
            pass

    def ci(arr, alpha=0.05):
        arr = np.array(arr)
        arr = arr[~np.isnan(arr)]
        return np.percentile(arr, [100*alpha/2, 100*(1-alpha/2)])

    return {
        "acme": acme0, "acme_ci": ci(boot_acme),
        "ade":  ade0,  "ade_ci":  ci(boot_ade),
        "total": tot0, "total_ci": ci(boot_tot),
        "prop":  prop0, "prop_ci": ci(boot_prop),
        "alpha": a0, "gamma": g0,
        "n": n, "n_boot": len(boot_acme),
    }

# ── 6. 페어별 Mediation 실행 ─────────────────────────────────────────────────
print(f"5. Mediation 분석 실행 (Bootstrap n={N_BOOT})...")

PAIRS = [
    ("ENSG00000176340", "chr11_64304866"),
    ("ENSG00000176340", "chr11_64304843"),
    ("ENSG00000243449", "chr4_2468860"),
    ("ENSG00000243449", "chr4_2468855"),
    ("ENSG00000243449", "chr4_2468842"),
    ("ENSG00000243449", "chr4_2468846"),
]

X_arr = meta_use["X"].values.astype(float)
results = []

for ensg, cpg_id in PAIRS:
    gene_name = TARGET_GENES[ensg]
    cpg_label = TARGET_CPGS[cpg_id]
    M_arr = cpg_data[cpg_id]["mval"]
    Y_arr = gene_data[ensg]["expr"]

    # NaN 제거
    valid = ~(np.isnan(M_arr) | np.isnan(Y_arr))
    res = mediation_bootstrap(X_arr[valid], M_arr[valid], Y_arr[valid],
                               cov[valid], n_boot=N_BOOT, rng=rng)
    res.update({"gene": gene_name, "cpg": cpg_id, "cpg_label": cpg_label,
                "pair": f"{gene_name} × {cpg_label}"})

    # 유의성: CI가 0을 포함하지 않으면 유의
    acme_sig  = not (res["acme_ci"][0] <= 0 <= res["acme_ci"][1])
    total_sig = not (res["total_ci"][0] <= 0 <= res["total_ci"][1])

    print(f"\n  [{gene_name} × {cpg_label}]  n={res['n']}")
    print(f"    Total effect   : {res['total']:+.4f} (95%CI [{res['total_ci'][0]:+.4f}, {res['total_ci'][1]:+.4f}]) {'*' if total_sig else 'n.s.'}")
    print(f"    ACME (mediated): {res['acme']:+.4f} (95%CI [{res['acme_ci'][0]:+.4f}, {res['acme_ci'][1]:+.4f}]) {'*' if acme_sig else 'n.s.'}")
    print(f"    ADE  (direct)  : {res['ade']:+.4f} (95%CI [{res['ade_ci'][0]:+.4f}, {res['ade_ci'][1]:+.4f}])")
    print(f"    Prop. mediated : {res['prop']:+.3f} ({100*res['prop']:.1f}%)")
    print(f"    X→M (alpha)    : {res['alpha']:+.4f}")
    print(f"    M→Y|X (gamma)  : {res['gamma']:+.4f}")

    results.append({
        "gene": gene_name, "cpg": cpg_id, "cpg_label": cpg_label,
        "n": res["n"],
        "total": res["total"], "total_ci_lo": res["total_ci"][0], "total_ci_hi": res["total_ci"][1],
        "acme": res["acme"], "acme_ci_lo": res["acme_ci"][0], "acme_ci_hi": res["acme_ci"][1],
        "ade": res["ade"],   "ade_ci_lo": res["ade_ci"][0],   "ade_ci_hi": res["ade_ci"][1],
        "prop_mediated": res["prop"],
        "prop_ci_lo": res["prop_ci"][0], "prop_ci_hi": res["prop_ci"][1],
        "alpha_XtoM": res["alpha"], "gamma_MtoY": res["gamma"],
        "acme_sig": acme_sig, "total_sig": total_sig,
    })

res_df = pd.DataFrame(results)
res_df.to_csv(f"{BASE}/data/mediation_results.tsv", sep="\t", index=False)

# ── 7. 시각화 ────────────────────────────────────────────────────────────────
print("\n6. 시각화 중...")
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.55, wspace=0.42)

GENE_COLORS = {"COX8A": "#d62728", "C4orf48": "#1f77b4"}

# Panel A & B: Forest plot — ACME per pair (split by gene)
for gi, (gene_name, ax_idx) in enumerate([("COX8A", gs[0, :2]), ("C4orf48", gs[1, :2])]):
    ax = fig.add_subplot(ax_idx)
    gene_res = res_df[res_df["gene"] == gene_name].reset_index(drop=True)
    y_pos = np.arange(len(gene_res))
    color = GENE_COLORS[gene_name]

    for i, row in gene_res.iterrows():
        sig_mark = "●" if row["acme_sig"] else "○"
        ax.errorbar(row["acme"], i, xerr=[[row["acme"]-row["acme_ci_lo"]], [row["acme_ci_hi"]-row["acme"]]],
                    fmt="none", color=color, capsize=4, linewidth=2)
        ax.scatter(row["acme"], i, s=90, color=color if row["acme_sig"] else "white",
                   edgecolors=color, zorder=3, linewidths=2)

    ax.axvline(0, color="gray", linestyle="--", linewidth=1.2)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['cpg_label']}\n(prop={r['prop_mediated']:+.2f})" for _, r in gene_res.iterrows()], fontsize=9)
    ax.set_xlabel("ACME (Mediated Effect)", fontsize=10)
    ax.set_title(f"{gene_name} — ACME per CpG pair\n(Bootstrap 95% CI, ●=significant)", fontsize=10, fontweight="bold", color=color)
    ax.invert_yaxis()

# Panel C: Path diagram summary — best pair per gene
ax_c = fig.add_subplot(gs[0, 2])
ax_c.axis("off")
best_cox = res_df[res_df["gene"]=="COX8A"].sort_values("acme").iloc[0]
best_c4  = res_df[res_df["gene"]=="C4orf48"].sort_values("acme").iloc[0]

def fmt_path(row, color):
    return (
        f"{'●' if row['total_sig'] else '○'} {row['gene']} × {row['cpg_label']}\n"
        f"  X→M (α):  {row['alpha_XtoM']:+.3f}\n"
        f"  M→Y (γ):  {row['gamma_MtoY']:+.3f}\n"
        f"  Total:    {row['total']:+.4f}\n"
        f"  ACME:     {row['acme']:+.4f} {'*' if row['acme_sig'] else 'n.s.'}\n"
        f"  ADE:      {row['ade']:+.4f}\n"
        f"  Prop:     {100*row['prop_mediated']:.1f}%"
    )

ax_c.text(0.05, 0.95, "Best pair per gene\n(lowest ACME)", transform=ax_c.transAxes,
          fontsize=10, fontweight="bold", va="top")
ax_c.text(0.05, 0.80, fmt_path(best_cox, GENE_COLORS["COX8A"]),
          transform=ax_c.transAxes, fontsize=9, va="top", family="monospace",
          color=GENE_COLORS["COX8A"],
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#fff0f0", edgecolor=GENE_COLORS["COX8A"]))
ax_c.text(0.05, 0.42, fmt_path(best_c4, GENE_COLORS["C4orf48"]),
          transform=ax_c.transAxes, fontsize=9, va="top", family="monospace",
          color=GENE_COLORS["C4orf48"],
          bbox=dict(boxstyle="round,pad=0.4", facecolor="#f0f4ff", edgecolor=GENE_COLORS["C4orf48"]))

# Panel D: Proportion mediated bar chart
ax_d = fig.add_subplot(gs[1, 2])
colors_bar = [GENE_COLORS[g] for g in res_df["gene"]]
bars = ax_d.barh(range(len(res_df)), res_df["prop_mediated"]*100,
                  color=colors_bar, alpha=0.75, edgecolor="white")
ax_d.axvline(0, color="gray", linestyle="--", linewidth=1.2)
ax_d.set_yticks(range(len(res_df)))
ax_d.set_yticklabels([f"{r['gene']}\n{r['cpg_label']}" for _, r in res_df.iterrows()], fontsize=8)
ax_d.set_xlabel("Proportion Mediated (%)", fontsize=10)
ax_d.set_title("Proportion of Total Effect\nMediated by CpG", fontsize=10, fontweight="bold")
ax_d.invert_yaxis()
for i, row in res_df.iterrows():
    ax_d.text(row["prop_mediated"]*100 + (0.5 if row["prop_mediated"] >= 0 else -0.5),
              i, f"{row['prop_mediated']*100:.1f}%", va="center", ha="left" if row["prop_mediated"] >= 0 else "right", fontsize=8)

plt.suptitle("Causal Mediation Analysis — Delayed CpG Mediates Persistent Gene Expression?\n(HC vs RC, n=173, Bootstrap n=1000)",
             fontsize=12, fontweight="bold", y=1.01)
plt.savefig(f"{BASE}/figures/mediation_results.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"   저장: {BASE}/figures/mediation_results.png")

# ── 8. 요약 출력 ─────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("Mediation Analysis 요약")
print("=" * 60)
print(f"{'Pair':30s} {'Total':>8} {'ACME':>8} {'ADE':>8} {'Prop%':>7} {'Sig'}")
print("-" * 60)
for _, r in res_df.iterrows():
    sig = "* ACME" if r["acme_sig"] else ("* Total" if r["total_sig"] else "n.s.")
    print(f"{r['gene']+' × '+r['cpg_label']:30s} {r['total']:>8.4f} {r['acme']:>8.4f} {r['ade']:>8.4f} {r['prop_mediated']*100:>7.1f}% {sig}")
print()
print(f"저장: {BASE}/data/mediation_results.tsv")
print(f"그림: {BASE}/figures/mediation_results.png")
