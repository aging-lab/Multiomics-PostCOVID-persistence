"""
cis-eQTM Analysis
=================
Persistent/Delayed CpG markers × Persistent/Delayed RNA genes (±1 Mb)
Partial Spearman correlation — covariates: Group, Age, Sex, methyl cell fractions

입력:
  - rna_combat_corrected_v2.tsv   : ComBat-corrected raw counts (43006 genes × ~1003 samples)
  - cpg_matrix_mval.npz           : CpG M-values (1,615,056 CpGs × 630 samples)
  - cpg_positions.tsv             : CpG 염색체 좌표
  - gene_tss.tsv                  : 대상 유전자 TSS 좌표 (사전 추출)
  - rna_dual_gate_markers_v2.tsv  : persistent / delayed RNA 마커
  - methyl_dual_gate_markers.tsv  : persistent / delayed CpG 마커
  - COVID-19_sample_metadata.tsv  : 임상 메타데이터 (Age, Sex, BMI, MM_SC)
  - HC_sample_metadata.tsv        : HC 메타데이터
  - methyl_cell_fractions.tsv     : Methyl EpiDISH cell fractions
  - sample_info_v2.tsv            : RNA sample group 정보
  - cpg_sample_info.tsv           : Methyl sample group 정보

출력:
  - data/eqtm_all_pairs.tsv       : 전체 CpG-gene 테스트 결과
  - data/eqtm_significant.tsv     : FDR < 0.05 유의 pair
  - figures/eqtm_manhattan.png    : -log10(p) by CpG position
  - figures/eqtm_top_pairs.png    : Top pair scatter plots
"""

import os, warnings
import numpy as np
import pandas as pd
import scipy.sparse as ssp
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ── 경로 ─────────────────────────────────────────────────────────────────────
BASE      = "/BiO/Research/Infectomics/LongCOVID/Analysis"
OUT_DIR   = f"{BASE}/5.Integration/1.cis_eQTM"
OUT_D     = f"{OUT_DIR}/data"
OUT_F     = f"{OUT_DIR}/figures"
WINDOW    = 1_000_000  # ±1 Mb

RNA_MATRIX   = f"{BASE}/2.Data_preparation/2.RNA_batch_correction/data/rna_combat_corrected_v2.tsv"
RNA_INFO     = f"{BASE}/2.Data_preparation/2.RNA_batch_correction/data/sample_info_v2.tsv"
CPG_MVAL_NPZ = f"{BASE}/2.Data_preparation/5.Methyl_matrix/data/cpg_matrix_mval.npz"
CPG_POS      = f"{BASE}/2.Data_preparation/5.Methyl_matrix/data/cpg_positions.tsv"
CPG_INFO     = f"{BASE}/2.Data_preparation/5.Methyl_matrix/data/cpg_sample_info.tsv"
META_COVID   = f"{BASE}/1.Metadata/COVID-19_sample_metadata.tsv"
HC_META      = f"{BASE}/1.Metadata/HC_sample_metadata.tsv"
METHYL_CF    = f"{BASE}/2.Data_preparation/6.Deconvolution/data/methyl_cell_fractions.tsv"
RNA_MARKERS  = f"{BASE}/3.Discovery/1.RNA_DE_ver2/data/rna_dual_gate_markers_v2.tsv"
METHYL_DM    = f"{BASE}/3.Discovery/2.Methyl_DM/data/methyl_dual_gate_markers.tsv"
GENE_TSS     = f"{OUT_D}/gene_tss.tsv"


# ── 유틸 ─────────────────────────────────────────────────────────────────────
def partial_spearman_fast(x, y, Z):
    """
    Partial Spearman correlation of x,y given covariates Z (n×k).
    Residuals of rank(x) ~ Z  and rank(y) ~ Z → Pearson of residuals.
    Returns (rho, pval).
    """
    n = len(x)
    rx = stats.rankdata(x).reshape(-1, 1)
    ry = stats.rankdata(y).reshape(-1, 1)
    # OLS residuals: e = y - Z(Z'Z)^{-1}Z'y
    Zc = np.column_stack([np.ones(n), Z])
    try:
        coef_x = np.linalg.lstsq(Zc, rx, rcond=None)[0]
        coef_y = np.linalg.lstsq(Zc, ry, rcond=None)[0]
    except Exception:
        return np.nan, np.nan
    ex = (rx - Zc @ coef_x).ravel()
    ey = (ry - Zc @ coef_y).ravel()
    if ex.std() == 0 or ey.std() == 0:
        return np.nan, np.nan
    r, p = stats.pearsonr(ex, ey)
    return r, p


def logcpm(counts_arr):
    lib_size = counts_arr.sum(axis=0, keepdims=True)
    lib_size = np.where(lib_size == 0, 1, lib_size)
    return np.log2(counts_arr / lib_size * 1e6 + 1)


# ── 메타데이터 로드 ────────────────────────────────────────────────────────────
print("Loading metadata ...")
meta_covid = pd.read_csv(META_COVID, sep="\t")
hc_meta    = pd.read_csv(HC_META,    sep="\t")
methyl_cf  = pd.read_csv(METHYL_CF,  sep="\t")

# 공유 공변량: Age, Sex (M=0/F=1), Group(AC=1,RC=2,HC=0)
cov_covid = meta_covid[["SampleID","Age","Sex"]].drop_duplicates("SampleID")
cov_covid["Sex_bin"] = (cov_covid["Sex"] == "F").astype(int)

# HC 메타에서 Age/Sex 추출 (컬럼명 확인)
hc_meta_cols = hc_meta.columns.tolist()
hc_sid_col   = "SampleID" if "SampleID" in hc_meta_cols else hc_meta_cols[0]
hc_age_col   = next((c for c in hc_meta_cols if "age" in c.lower()), None)
hc_sex_col   = next((c for c in hc_meta_cols if "sex" in c.lower() or "gender" in c.lower()), None)

cov_hc = pd.DataFrame()
if hc_age_col and hc_sex_col:
    cov_hc = hc_meta[[hc_sid_col, hc_age_col, hc_sex_col]].copy()
    cov_hc.columns = ["SampleID", "Age", "Sex"]
    cov_hc["Sex_bin"] = cov_hc["Sex"].apply(lambda s: 1 if str(s).upper() in ("F","FEMALE","2") else 0)

cov_all = pd.concat([cov_covid[["SampleID","Age","Sex_bin"]],
                     cov_hc[["SampleID","Age","Sex_bin"]] if len(cov_hc) else pd.DataFrame()],
                    ignore_index=True).drop_duplicates("SampleID")


# ── RNA 데이터 로드 ───────────────────────────────────────────────────────────
print("Loading RNA matrix ...")
rna_raw  = pd.read_csv(RNA_MATRIX, sep="\t", index_col=0)
rna_info = pd.read_csv(RNA_INFO, sep="\t")
group_map = rna_info.set_index("SampleID")["Group"].to_dict()

# logCPM
rna_lcpm = pd.DataFrame(
    logcpm(rna_raw.values.astype(float)),
    index=rna_raw.index, columns=rna_raw.columns
)

# 대상 RNA 유전자 (persistent + delayed)
rna_markers = pd.read_csv(RNA_MARKERS, sep="\t")
target_rna  = rna_markers[rna_markers["category"].isin(["persistent","delayed"])]["gene_id"].tolist()
target_rna  = [g for g in target_rna if g in rna_lcpm.index]
print(f"  Target RNA genes in matrix: {len(target_rna)}")

rna_sub = rna_lcpm.loc[target_rna]  # shape: (n_genes, n_samples)


# ── CpG 데이터 로드 ───────────────────────────────────────────────────────────
print("Loading CpG M-value matrix ...")
cpg_npz  = np.load(CPG_MVAL_NPZ)
cpg_mval_data  = cpg_npz["data"] if "data" in cpg_npz else cpg_npz[cpg_npz.files[0]]
cpg_pos_df     = pd.read_csv(CPG_POS, sep="\t")
cpg_info_df    = pd.read_csv(CPG_INFO, sep="\t")

# Check if sparse matrix
if ssp.issparse(cpg_mval_data):
    cpg_mval_data = cpg_mval_data.toarray()

print(f"  CpG matrix shape: {cpg_mval_data.shape}")
print(f"  CpG positions: {len(cpg_pos_df)}")
print(f"  CpG samples: {len(cpg_info_df)}")

# 대상 CpG (persistent + delayed in Gate2 = RC vs HC)
methyl_dm = pd.read_csv(METHYL_DM, sep="\t")
target_cpg = methyl_dm[methyl_dm["category"].isin(["persistent","delayed"])]["cpg_id"].tolist()
print(f"  Target CpGs (persistent+delayed Gate2): {len(target_cpg)}")

# CpG 인덱스 매핑
cpg_id_to_idx = {cid: i for i, cid in enumerate(cpg_pos_df["cpg_id"])}
target_cpg = [c for c in target_cpg if c in cpg_id_to_idx]
print(f"  Target CpGs in matrix: {len(target_cpg)}")

target_cpg_idx = [cpg_id_to_idx[c] for c in target_cpg]


# ── 공유 샘플 확보 ─────────────────────────────────────────────────────────────
print("Identifying overlapping samples ...")
rna_samples   = set(rna_sub.columns)
cpg_samples   = set(cpg_info_df["SampleID"])
shared        = sorted(rna_samples & cpg_samples)
print(f"  Shared samples (RNA+Methyl): {len(shared)}")

# 공유 샘플에서 Group numeric
group_num = np.array([{"HC": 0, "AC": 1, "RC": 2}.get(group_map.get(s, "HC"), 0) for s in shared])

# RNA 서브셋 (공유 샘플만)
rna_shared = rna_sub[shared].values.T  # (n_shared, n_genes)

# CpG 서브셋 (공유 샘플 × 대상 CpG)
cpg_sample_order = list(cpg_info_df["SampleID"])
cpg_shared_idx   = [cpg_sample_order.index(s) for s in shared]
cpg_shared       = cpg_mval_data[np.ix_(target_cpg_idx, cpg_shared_idx)].T  # (n_shared, n_cpg)
print(f"  CpG shared matrix: {cpg_shared.shape}")

# 공변량 행렬 Z (group, age, sex, methyl cell fractions)
cov_df = pd.DataFrame({"SampleID": shared, "Group": group_num})
cov_df = cov_df.merge(cov_all, on="SampleID", how="left")
cov_df = cov_df.merge(methyl_cf[["SampleID","B","NK","CD4T","CD8T","Mono"]], on="SampleID", how="left")

# NaN 채우기 (중앙값)
for col in ["Age","Sex_bin","B","NK","CD4T","CD8T","Mono"]:
    if col in cov_df.columns:
        cov_df[col] = cov_df[col].fillna(cov_df[col].median())

Z_cols = ["Group","Sex_bin","Age","B","NK","CD4T","CD8T","Mono"]
Z_cols = [c for c in Z_cols if c in cov_df.columns]
Z = cov_df[Z_cols].values.astype(float)
print(f"  Covariate matrix Z: {Z.shape}  cols={Z_cols}")


# ── 유전자 TSS 로드 & CpG 좌표 ────────────────────────────────────────────────
gene_tss = pd.read_csv(GENE_TSS, sep="\t")
gene_tss = gene_tss.set_index("gene_id")  # index = full gene_id like ENSG...ver_SYMBOL

cpg_pos_sub = cpg_pos_df[cpg_pos_df["cpg_id"].isin(target_cpg)].set_index("cpg_id")


# ── CpG-Gene 페어링 (±1 Mb) ────────────────────────────────────────────────
print("Pairing CpGs with genes (±1 Mb) ...")
pairs = []
for gene_id in target_rna:
    if gene_id not in gene_tss.index:
        continue
    row = gene_tss.loc[gene_id]
    chrom = row["chrom"]
    tss   = int(row["tss"])
    cat_rna = rna_markers.set_index("gene_id").loc[gene_id, "category"] if gene_id in rna_markers["gene_id"].values else "unknown"
    for cpg_id in target_cpg:
        if cpg_id not in cpg_pos_sub.index:
            continue
        cp = cpg_pos_sub.loc[cpg_id]
        if cp["chr"] != chrom:
            continue
        dist = int(cp["start"]) - tss
        if abs(dist) <= WINDOW:
            cat_cpg = methyl_dm.set_index("cpg_id").loc[cpg_id, "category"]
            pairs.append({"gene_id": gene_id, "cpg_id": cpg_id,
                          "dist": dist, "cat_rna": cat_rna, "cat_cpg": cat_cpg})

print(f"  CpG-Gene pairs within ±1 Mb: {len(pairs)}")

if len(pairs) == 0:
    print("No pairs found! Check chromosome naming consistency.")
    import sys; sys.exit(1)


# ── Partial Spearman ──────────────────────────────────────────────────────────
print("Computing partial Spearman correlations ...")
gene_to_col  = {g: i for i, g in enumerate(target_rna)}
cpg_to_col   = {c: i for i, c in enumerate(target_cpg)}

results = []
n_pairs = len(pairs)
for i, p in enumerate(pairs):
    if i % 100 == 0:
        print(f"  {i}/{n_pairs} ...", end="\r")
    g_idx = gene_to_col.get(p["gene_id"])
    c_idx = cpg_to_col.get(p["cpg_id"])
    if g_idx is None or c_idx is None:
        continue
    rna_vec = rna_shared[:, g_idx]
    cpg_vec = cpg_shared[:, c_idx]
    # Remove NaN rows
    mask = np.isfinite(rna_vec) & np.isfinite(cpg_vec)
    if mask.sum() < 30:
        continue
    rho, pval = partial_spearman_fast(rna_vec[mask], cpg_vec[mask], Z[mask])
    p["rho"] = rho
    p["pval"] = pval
    p["n"] = int(mask.sum())
    results.append(p)

print(f"\n  Tested pairs: {len(results)}")

res_df = pd.DataFrame(results)
res_df = res_df.dropna(subset=["pval"])

# FDR correction
_, fdr, _, _ = multipletests(res_df["pval"].values, method="fdr_bh")
res_df["fdr"] = fdr
res_df = res_df.sort_values("pval")

# 저장
res_df.to_csv(f"{OUT_D}/eqtm_all_pairs.tsv", sep="\t", index=False)
sig = res_df[res_df["fdr"] < 0.05]
sig.to_csv(f"{OUT_D}/eqtm_significant.tsv", sep="\t", index=False)
print(f"  Significant pairs (FDR<0.05): {len(sig)}")
print(sig[["cpg_id","gene_id","rho","pval","fdr","dist","cat_rna","cat_cpg"]].head(10).to_string())


# ── 시각화 ───────────────────────────────────────────────────────────────────
print("Plotting ...")

# 1. Manhattan-style: -log10(p) by CpG index
fig, ax = plt.subplots(figsize=(12, 4))
res_df["log10p"] = -np.log10(res_df["pval"].clip(1e-300))
colors = {"persistent": "#c0392b", "delayed": "#3498db", "acute_only": "#e67e22"}
for cat, grp in res_df.groupby("cat_cpg"):
    ax.scatter(range(len(grp)), grp["log10p"], s=10, alpha=0.5,
               color=colors.get(cat, "#7f8c8d"), label=cat, rasterized=True)
threshold = -np.log10(res_df[res_df["fdr"] < 0.05]["pval"].max()) if len(sig) > 0 else 4
ax.axhline(threshold, color="red", lw=1, ls="--", label=f"FDR<0.05 (−log10p={threshold:.1f})")
ax.set_xlabel("CpG-Gene pair index")
ax.set_ylabel("−log₁₀(p)")
ax.set_title(f"cis-eQTM: {len(res_df)} pairs tested, {len(sig)} FDR<0.05 significant")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT_F}/eqtm_manhattan.png", dpi=150)
plt.close()

# 2. Top pair scatter plots
n_top = min(6, len(sig))
if n_top > 0:
    top = sig.head(n_top)
    ncols = 3
    nrows = (n_top + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    axes = np.array(axes).ravel()
    for ax_i, (_, row) in enumerate(top.iterrows()):
        ax = axes[ax_i]
        g_idx = gene_to_col[row["gene_id"]]
        c_idx = cpg_to_col[row["cpg_id"]]
        rna_v = rna_shared[:, g_idx]
        cpg_v = cpg_shared[:, c_idx]
        mask  = np.isfinite(rna_v) & np.isfinite(cpg_v)
        grp_v = group_num[mask]
        group_colors = {0: "#95a5a6", 1: "#e74c3c", 2: "#3498db"}
        for gval, glabel in [(0,"HC"),(1,"AC"),(2,"RC")]:
            m2 = grp_v == gval
            ax.scatter(cpg_v[mask][m2], rna_v[mask][m2], s=15, alpha=0.6,
                       color=group_colors[gval], label=glabel, rasterized=True)
        gene_sym = row["gene_id"].split("_", 1)[-1] if "_" in row["gene_id"] else row["gene_id"]
        ax.set_xlabel(f"CpG M-value\n({row['cpg_id']})", fontsize=8)
        ax.set_ylabel(f"RNA logCPM\n({gene_sym})", fontsize=8)
        ax.set_title(f"ρ={row['rho']:.3f}  FDR={row['fdr']:.2e}\n"
                     f"dist={row['dist']:,} bp | {row['cat_cpg']} CpG × {row['cat_rna']} gene", fontsize=7)
        ax.legend(fontsize=7)
    for ax_i in range(n_top, len(axes)):
        axes[ax_i].set_visible(False)
    plt.suptitle("Top cis-eQTM pairs (partial Spearman)", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig(f"{OUT_F}/eqtm_top_pairs.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved top {n_top} pair scatter plot.")

# 3. Distance distribution of significant pairs
if len(sig) > 0:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(sig["dist"] / 1000, bins=40, color="#3498db", edgecolor="white", alpha=0.8)
    ax.axvline(0, color="red", lw=1.5, ls="--", label="TSS")
    ax.set_xlabel("Distance from TSS (kb)")
    ax.set_ylabel("Number of significant pairs")
    ax.set_title(f"Distance distribution of {len(sig)} significant cis-eQTM pairs (FDR<0.05)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUT_F}/eqtm_distance_dist.png", dpi=150)
    plt.close()

print("Done. Outputs written to:", OUT_D, OUT_F)
