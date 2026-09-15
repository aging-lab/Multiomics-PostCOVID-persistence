"""
Methylation batch/technical-shift sensitivity (reviewer comment #4) — Tier B.
============================================================================
No explicit sequencing-batch label exists for the methylome, and batch is
collinear with condition (RC and HC are different cohorts), so a batch covariate
cannot be fit directly. As a surrogate we remove any per-sample GLOBAL methylation
offset (the classic signature of a technical/batch shift): we subtract each
sample's genome-wide mean M-value, then re-fit the identical RC-vs-HC OLS. If the
delayed hypermethylation were a uniform per-sample/batch offset it would collapse;
if it is CpG-specific it should survive.

Reproduces methyl_discovery_cleancohort.py's clean RC+HC selection and design
(~Sex+Age+B+NK+CD4T+CD8T+Mono+condition), then reports, for the delayed sets,
mean delta-beta and the number of CpGs retaining |delta-beta|>0.05 & p<0.05
BEFORE vs AFTER per-sample global centering.
Run with .venv_deseq python.
"""
import numpy as np, pandas as pd, time, os, warnings
warnings.filterwarnings("ignore")
from scipy.stats import t as t_dist

R   = "PROJECT_ROOT/research/LongCOVID/Results"
NEW = R + "_updated_260714"
PREP = f"{R}/2.Data_preparation"
NPZ_BETA = f"{PREP}/5.Methyl_matrix/data/cpg_matrix_beta.npz"
CPG_SI   = f"{PREP}/5.Methyl_matrix/data/cpg_sample_info.tsv"
COVID_F  = f"{R}/1.Metadata/COVID/COVID-19_sample_metadata.tsv"
HC_F     = f"{R}/1.Metadata/HC/HC_sample_metadata.tsv"
CF_METH_F= f"{PREP}/6.Deconvolution/data/methyl_cell_fractions.tsv"
MDM = f"{NEW}/3.Discovery/2.Methyl_DM/data"
PROMHYP = f"{NEW}/4.Trajectory/4.Methyl_trajectory/data/methyl_cpg_trajectory_clean.tsv"
OUT = f"{NEW}/addResult_by_review_0813/data"
os.makedirs(OUT, exist_ok=True)
FDR_CUT=0.05; DB_CUT=0.05; FLOOR=28; CHUNK=200_000

print("[1] load beta matrix ...", flush=True)
z = np.load(NPZ_BETA, allow_pickle=True)
cpg_ids = np.asarray([str(c) for c in z['cpg_ids']])
samp_ids = np.asarray([str(s) for s in z['sample_ids']])
samp2idx = {s:i for i,s in enumerate(samp_ids)}
cpg2idx  = {c:i for i,c in enumerate(cpg_ids)}

print("[2] metadata + clean RC/HC selection ...", flush=True)
cpg_si = pd.read_csv(CPG_SI, sep="\t")
covid  = pd.read_csv(COVID_F, sep="\t")
hc     = pd.read_csv(HC_F, sep="\t")
cf_m   = pd.read_csv(CF_METH_F, sep="\t", index_col=0)
covid_sub = covid[['SampleID','Age','Sex','Recovery_days']]
meta_c = cpg_si[cpg_si['Group'].isin(['AC','RC'])].merge(covid_sub, on='SampleID', how='left')
hcm = cpg_si[cpg_si['Group']=='HC'].merge(hc[['DataID_Methyl','Age','Sex']],
             left_on='SampleID', right_on='DataID_Methyl', how='left')
hcm['Recovery_days'] = np.nan
meta = pd.concat([meta_c[['SampleID','Group','Age','Sex','Recovery_days']],
                  hcm[['SampleID','Group','Age','Sex','Recovery_days']]], ignore_index=True)
meta['Age'] = pd.to_numeric(meta['Age'], errors='coerce'); meta['Age'] = meta['Age'].fillna(meta['Age'].median())
meta['Sex'] = meta['Sex'].map({'M':0,'F':1}).fillna(0).astype(int)
cf_use = cf_m[['B','NK','CD4T','CD8T','Mono']].copy(); cf_use.index.name='SampleID'
meta = meta.merge(cf_use.reset_index(), on='SampleID', how='left')
cf_cols=['B','NK','CD4T','CD8T','Mono']; meta[cf_cols]=meta[cf_cols].fillna(meta[cf_cols].median())
# restrict to methyl samples present in matrix
meta = meta[meta['SampleID'].isin(samp_ids)].copy()
meta['recd'] = pd.to_numeric(meta['Recovery_days'], errors='coerce')
is_rc = meta['Group']=='RC'; is_L1 = meta['SampleID'].str.endswith('-L1'); is_short = is_rc & (meta['recd']<FLOOR)
excl = is_rc & (is_L1 | is_short)
meta = meta.loc[~excl].copy()
rc_ids = meta[meta['Group']=='RC']['SampleID'].tolist()
hc_ids = meta[meta['Group']=='HC']['SampleID'].tolist()
print(f"    clean methyl RC={len(rc_ids)}  HC={len(hc_ids)}", flush=True)
g2 = meta[meta['Group'].isin(['RC','HC'])].copy().reset_index(drop=True)
g2['condition'] = np.where(g2['Group']=='RC',1,0)
col_idx = [samp2idx[s] for s in g2['SampleID']]

print("[3] subset beta cols, reconstruct M (chunked): per-sample global mean + target rows ...", flush=True)
t0=time.time()
beta = z['matrix'][:, col_idx].astype(np.float32)   # CpG x n_sub
del z
ncpg, nsub = beta.shape
# target rows: delayed(303) + promoter-hyper(155) union
mk = pd.read_csv(f"{MDM}/methyl_markers_cleancohort_clean.tsv", sep="\t")
delayed303 = mk[mk['category']=='delayed']['cpg_id'].tolist()
promhyp155 = pd.read_csv(PROMHYP, sep="\t")['cpg_id'].tolist()
target = [c for c in dict.fromkeys(delayed303 + promhyp155) if c in cpg2idx]
tset303 = set(delayed303); tset155 = set(promhyp155)
tgt_pos = {c: cpg2idx[c] for c in target}
# accumulate per-sample sum of imputed M over ALL CpGs
samp_sum = np.zeros(nsub, dtype=np.float64)
tgt_M = {}   # cpg -> M vector (n_sub)
for a in range(0, ncpg, CHUNK):
    b = min(a+CHUNK, ncpg)
    bc = beta[a:b]
    nanm = np.isnan(bc)
    bclip = np.clip(bc, 0.001, 0.999)
    M = np.log2(bclip/(1.0-bclip)).astype(np.float32)
    M[nanm] = np.nan
    rm = np.nanmean(M, axis=1, keepdims=True)
    M[nanm] = np.broadcast_to(rm, M.shape)[nanm]     # impute NaN by row mean
    samp_sum += M.sum(axis=0)
    for c, gi in tgt_pos.items():
        if a <= gi < b:
            tgt_M[c] = M[gi-a].astype(np.float64).copy()
    if a % (CHUNK*3) == 0:
        print(f"      {b:,}/{ncpg:,}  ({time.time()-t0:.0f}s)", flush=True)
samp_global_mean = samp_sum / ncpg
print(f"    per-sample global mean M: RC mean={samp_global_mean[g2['condition'].values==1].mean():.4f}  "
      f"HC mean={samp_global_mean[g2['condition'].values==0].mean():.4f}", flush=True)
del beta

import patsy
design = patsy.dmatrix('~Sex + Age + B + NK + CD4T + CD8T + Mono + condition', data=g2, return_type='dataframe')
X = design.values.astype(np.float64); n_s, p = X.shape; df = n_s - p
XtX_inv = np.linalg.inv(X.T @ X); cond_i = list(design.columns).index('condition')
XtXinv_Xt = XtX_inv @ X.T
g1v = g2['condition'].values==1; g0v = g2['condition'].values==0

def fit_targets(center):
    rows=[]
    for c in target:
        M = tgt_M[c].copy()
        if center:
            M = M - samp_global_mean           # remove per-sample global offset
        Bcoef = XtXinv_Xt @ M
        resid = M - X @ Bcoef
        sig2 = (resid**2).sum()/df
        se = np.sqrt(sig2 * XtX_inv[cond_i, cond_i])
        tstat = Bcoef[cond_i]/(se+1e-300)
        pval = 2*t_dist.sf(abs(tstat), df=df)
        Bimp = (2.0**M)/(1.0+2.0**M)
        dbeta = Bimp[g1v].mean() - Bimp[g0v].mean()
        rows.append(dict(cpg_id=c, delta_beta=dbeta, pval=pval,
                         in303=c in tset303, in155=c in tset155))
    return pd.DataFrame(rows)

print("[4] fit target CpGs BEFORE vs AFTER per-sample global centering ...", flush=True)
pre = fit_targets(False); post = fit_targets(True)

def summarize(dfr, label):
    for setname, mask in [("delayed_303", dfr['in303']), ("promhyp_155", dfr['in155'])]:
        s = dfr[mask]
        mean_db = s['delta_beta'].mean()
        keep = int(((s['delta_beta'].abs()>DB_CUT) & (s['pval']<0.05)).sum())
        print(f"    [{label}] {setname}: n={len(s)}  mean delta-beta={mean_db:+.5f}  "
              f"survive(|db|>0.05 & p<0.05)={keep}/{len(s)}")
        yield dict(state=label, cpg_set=setname, n=len(s), mean_delta_beta=float(mean_db),
                   n_survive=keep)
print("\n=== BEFORE centering (should reproduce discovery ~+0.074 on 155) ===")
rows = list(summarize(pre, "uncentered"))
print("=== AFTER per-sample global centering (batch-shift surrogate removed) ===")
rows += list(summarize(post, "centered"))
outdf = pd.DataFrame(rows)
outdf.to_csv(f"{OUT}/methyl_batch_tierB_centering.tsv", sep="\t", index=False)
# also save per-CpG before/after for the 155
merged = pre[['cpg_id','delta_beta','pval','in155']].rename(columns={'delta_beta':'db_uncentered','pval':'p_uncentered'}) \
    .merge(post[['cpg_id','delta_beta','pval']].rename(columns={'delta_beta':'db_centered','pval':'p_centered'}), on='cpg_id')
merged.to_csv(f"{OUT}/methyl_batch_tierB_perCpG.tsv", sep="\t", index=False)
print(f"\n[saved] {OUT}/methyl_batch_tierB_centering.tsv")
print(f"[saved] {OUT}/methyl_batch_tierB_perCpG.tsv")
