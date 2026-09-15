"""
Methylation discovery on the CLEAN convalescent cohort (v3) — gate2 only.
========================================================================
Only the Convalescent gate (RC vs HC) is re-run; Acute gate (AC_T0 vs HC) does
not involve RC -> reuse saved methyl gate1 sig set.
Clean RC (methyl): V1 only, recovery_days>=28, exclude -L1  (from the 110 methyl RC).

M-values reconstructed from local beta matrix (mval npz is 0 bytes):
  M = log2(beta/(1-beta)); NaN imputed by row mean on M (matches original).
Vectorized OLS in CpG chunks (memory-safe). Design/covariates identical to
original methyl_dm.py:  ~Sex+Age+B+NK+CD4T+CD8T+Mono + condition.
Run with .venv_deseq python.
"""
import numpy as np, pandas as pd, time, warnings, os
warnings.filterwarnings("ignore")
VALIDATE = os.environ.get("MODE")=="validate"   # keep ALL 110 RC to reproduce original
SFX = "_validate" if VALIDATE else "_clean"
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests
import patsy

R="PROJECT_ROOT/research/LongCOVID/Results"
PREP=f"{R}/2.Data_preparation"; MDM=f"{R}/3.Discovery/2.Methyl_DM"
OUT="PROJECT_ROOT/research/LongCOVID/Results_updated_260714/3.Discovery/2.Methyl_DM"
NPZ_BETA=f"{PREP}/5.Methyl_matrix/data/cpg_matrix_beta.npz"
CPG_SI=f"{PREP}/5.Methyl_matrix/data/cpg_sample_info.tsv"
COVID_F=f"{R}/1.Metadata/COVID-19_sample_metadata.tsv"; HC_F=f"{R}/1.Metadata/HC_sample_metadata.tsv"
CF_METH_F=f"{PREP}/6.Deconvolution/data/methyl_cell_fractions.tsv"
FDR_CUT=0.05; DB_CUT=0.05; FLOOR=28; CHUNK=200_000

print("[1] load beta matrix ...")
z=np.load(NPZ_BETA,allow_pickle=True)
cpg_ids=np.asarray([str(c) for c in z['cpg_ids']])
samp_ids=np.asarray([str(s) for s in z['sample_ids']])
samp2idx={s:i for i,s in enumerate(samp_ids)}

print("[2] build metadata (Age/Sex/cell fractions) ...")
cpg_si=pd.read_csv(CPG_SI,sep="\t"); covid=pd.read_csv(COVID_F,sep="\t"); hc=pd.read_csv(HC_F,sep="\t")
cf_m=pd.read_csv(CF_METH_F,sep="\t",index_col=0)
covid_sub=covid[['SampleID','Timepoint','Age','Sex','Recovery_days']]
meta_c=cpg_si[cpg_si['Group'].isin(['AC','RC'])].merge(covid_sub,on='SampleID',how='left')
hcm=cpg_si[cpg_si['Group']=='HC'].reset_index(drop=True).copy()
hcm['BaseID']=hcm['SampleID'].str.replace(r'-V\d+.*$','',regex=True).str.replace(r'-(CT|CR)\d+.*$','',regex=True)
s1=hcm.merge(hc[['DataID_Methyl','Age','Sex']],left_on='SampleID',right_on='DataID_Methyl',how='left')
nan=s1[s1['Age'].isna()].index
s2=hcm.loc[nan].merge(hc[['KU10K_ID','Age','Sex']],left_on='BaseID',right_on='KU10K_ID',how='left')
s1.loc[nan,'Age']=s2['Age'].values; s1.loc[nan,'Sex']=s2['Sex'].values
meta_hc=s1[['SampleID','Group','Age','Sex']].copy(); meta_hc['Recovery_days']=np.nan
meta=pd.concat([meta_c[['SampleID','Group','Age','Sex','Recovery_days']],meta_hc],ignore_index=True)
meta['Age']=pd.to_numeric(meta['Age'],errors='coerce'); meta['Age']=meta['Age'].fillna(meta['Age'].median())
meta['Sex']=meta['Sex'].map({'M':0,'F':1}).fillna(0).astype(int)
cf_use=cf_m[['B','NK','CD4T','CD8T','Mono']].copy(); cf_use.index.name='SampleID'
meta=meta.merge(cf_use.reset_index(),on='SampleID',how='left')
cf_cols=['B','NK','CD4T','CD8T','Mono']; meta[cf_cols]=meta[cf_cols].fillna(meta[cf_cols].median())

# ---- CLEAN COHORT: RC = V1, recovery>=28, no L1 ; restricted to methyl samples ----
meta=meta[meta['SampleID'].isin(samp_ids)].copy()
meta['recd']=pd.to_numeric(meta['Recovery_days'],errors='coerce')
is_rc=meta['Group']=='RC'; is_L1=meta['SampleID'].str.endswith('-L1'); is_short=is_rc&(meta['recd']<FLOOR)
excl=(is_rc&False) if VALIDATE else (is_rc&(is_L1|is_short))
if VALIDATE: print("    [VALIDATE MODE] keeping ALL RC (should reproduce original ~399/396)")
excluded=meta.loc[excl,'SampleID'].tolist()
meta=meta.loc[~excl].copy()
rc_ids=meta[meta['Group']=='RC']['SampleID'].tolist(); hc_ids=meta[meta['Group']=='HC']['SampleID'].tolist()
print(f"    methyl RC: kept={len(rc_ids)}  excluded={len(excluded)} (L1={int((is_rc&is_L1).sum())}, V1<{FLOOR}d={int(is_short.sum())})  HC={len(hc_ids)}")
pd.Series(excluded,name='excluded_SampleID').to_csv(f"{OUT}/data/excluded_samples{SFX}.tsv",index=False,sep="\t")

g2_ids=rc_ids+hc_ids
mg=meta[meta['SampleID'].isin(g2_ids)].copy().reset_index(drop=True)
mg['condition']=np.where(mg['Group']=='RC',1,0)
col_idx=[samp2idx[s] for s in mg['SampleID']]

print("[3] subset beta cols + reconstruct M-values (match build_cpg_matrix.py) ...")
t0=time.time()
beta=z['matrix'][:,col_idx].astype(np.float32)   # CpG x n_sub (raw, may have NaN)
del z
# reconstruct M-value EXACTLY like the matrix builder: clip beta to [0.001,0.999]
nanm=np.isnan(beta)
bclip=np.clip(beta,0.001,0.999); del beta
mval=np.log2(bclip/(1.0-bclip)).astype(np.float32); del bclip
mval[nanm]=np.nan
# impute NaN on M by row mean (matches methyl_dm.py)
rm=np.nanmean(mval,axis=1,keepdims=True); mval[nanm]=np.broadcast_to(rm,mval.shape)[nanm]
print(f"    subset {mval.shape}, imputed {int(nanm.sum()):,} ({time.time()-t0:.1f}s)")

design=patsy.dmatrix('~Sex + Age + B + NK + CD4T + CD8T + Mono + condition',data=mg,return_type='dataframe')
cond_key=[c for c in design.columns if 'condition' in c][0]
X=design.values.astype(np.float64); n_s,p=X.shape; df=n_s-p
XtX_inv=np.linalg.inv(X.T@X); cond_i=list(design.columns).index(cond_key)
g1v=mg['condition'].values==1; g0v=mg['condition'].values==0
print(f"[4] chunked OLS: {mval.shape[0]:,} CpGs, n={n_s}, df={df}, chunk={CHUNK} ...")

ncpg=mval.shape[0]; pval=np.empty(ncpg); dbeta=np.empty(ncpg); dmv=np.empty(ncpg)
XtXinv_Xt = XtX_inv @ X.T   # p x n_s
for a in range(0,ncpg,CHUNK):
    b=min(a+CHUNK,ncpg)
    Mc=mval[a:b].astype(np.float64)               # nc x n_s (M-value)
    Bc=(2.0**Mc)/(1.0+2.0**Mc)                     # beta from imputed M (for Δβ), matches original
    Bcoef=(XtXinv_Xt @ Mc.T)                       # p x nc
    resid=Mc.T - X@Bcoef                           # n_s x nc
    sig2=(resid**2).sum(0)/df
    se=np.sqrt(sig2*XtX_inv[cond_i,cond_i])
    tstat=Bcoef[cond_i,:]/(se+1e-300)
    pval[a:b]=2*t_dist.sf(np.abs(tstat),df=df)
    dmv[a:b]=Bcoef[cond_i,:]
    dbeta[a:b]=Bc[:,g1v].mean(1)-Bc[:,g0v].mean(1)
    if a% (CHUNK*3)==0: print(f"      {b:,}/{ncpg:,}")
del mval
_,padj,_,_=multipletests(pval,method='fdr_bh')
res=pd.DataFrame({'cpg_id':cpg_ids,'delta_mval':dmv,'delta_beta':dbeta,'abs_delta_beta':np.abs(dbeta),
                  'pvalue':pval,'adj_pvalue':padj})
res.to_csv(f"{OUT}/data/gate2_rc_vs_hc{SFX}.tsv",sep="\t",index=False)
d_persist=set(res[(res['adj_pvalue']<FDR_CUT)&(res['abs_delta_beta']>=DB_CUT)]['cpg_id'])

# reuse original methyl gate1 sig set (AC_T0 vs HC unchanged)
g1=pd.read_csv(f"{MDM}/data/gate1_ac_t0_vs_hc.tsv",sep="\t")
d_acute=set(g1[(g1['adj_pvalue']<FDR_CUT)&(g1['abs_delta_beta']>=DB_CUT)]['cpg_id'])
orig=pd.read_csv(f"{MDM}/data/methyl_dual_gate_markers.tsv",sep="\t",index_col=0)
orig_del=set(orig[orig['category']=='delayed'].index); orig_pers=set(orig[orig['category']=='persistent'].index)

persistent=d_acute&d_persist; delayed=d_persist-d_acute
print("\n=== CLEAN COHORT METHYL DISCOVERY (gate2 re-run) ===")
print(f"  Convalescent gate sig (RC vs HC): {len(d_persist)}   (orig 399)")
print(f"  Persistent (A∩B): {len(persistent)}   (orig {len(orig_pers)})")
print(f"  Delayed    (B\\A): {len(delayed)}    (orig {len(orig_del)})")
print(f"  delayed ∩ orig396 = {len(delayed&orig_del)}  (lost {len(orig_del-delayed)}, gained {len(delayed-orig_del)})")

allc=list(d_acute|d_persist); ri=res.set_index('cpg_id')
m=pd.DataFrame(index=allc)
m['delta_beta_gate2']=ri['delta_beta'].reindex(allc); m['padj_gate2']=ri['adj_pvalue'].reindex(allc)
m['in_gate1']=m.index.isin(d_acute); m['in_gate2']=m.index.isin(d_persist)
m['category']=np.select([m['in_gate1']&m['in_gate2'],~m['in_gate1']&m['in_gate2']],['persistent','delayed'],default='acute_only')
m['in_orig_delayed']=m.index.isin(orig_del)
m.index.name='cpg_id'; m.to_csv(f"{OUT}/data/methyl_markers_cleancohort{SFX}.tsv",sep="\t")
print(f"\n[saved] {OUT}/data/methyl_markers_cleancohort{SFX}.tsv , gate2_rc_vs_hc{SFX}.tsv , excluded_samples{SFX}.tsv")
