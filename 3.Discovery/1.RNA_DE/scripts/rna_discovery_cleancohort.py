"""
RNA discovery on the CLEAN convalescent cohort — LongCOVID re-analysis (v3).
==========================================================================
Canonical convalescent RC cohort (decided 2026-07):
  - V1 samples only (1 per subject -> no pseudoreplication)
  - recovery_days >= 28  (4-week convalescent floor; excludes acute-window overlap)
  - -L1 (~1yr) fully excluded from discovery (used only in recovery-time analysis)
  => RC excluded = 10 (-L1) + 10 (V1 <28d) = 20 ; RC kept = 138

Replicates the gate2 half of rna_de_v2.py (design/globin/CPM filter/DESeq2).
Acute gate (AC_T0 vs HC) is UNCHANGED -> reuse saved sig_gate1 set.
persistent = d_acute ∩ d_persist(clean).  Outputs a full marker table.
Run with .venv_deseq python (inmoose).
"""
import numpy as np, pandas as pd, warnings, re
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from inmoose.deseq2 import DESeqDataSet, DESeq
from inmoose.deseq2.results import results_dds

R="PROJECT_ROOT/research/LongCOVID/Results"
PREP=f"{R}/2.Data_preparation"; DISC2=f"{R}/3.Discovery/1.RNA_DE_ver2"
OUT="PROJECT_ROOT/research/LongCOVID/Results_updated_260714/3.Discovery/1.RNA_DE"
COUNT_F=f"{PREP}/2.RNA_batch_correction/data/rna_combat_corrected_v2.tsv"
LOGCPM_F=f"{PREP}/4.RNA_normalization/data/rna_logcpm_v2.tsv.gz"
SINFO_F=f"{PREP}/2.RNA_batch_correction/data/sample_info_v2.tsv"
COVID_F=f"{R}/1.Metadata/COVID-19_sample_metadata.tsv"
HC_F=f"{R}/1.Metadata/HC_sample_metadata.tsv"
CF_F=f"{PREP}/6.Deconvolution/data/rna_cell_fractions_v2.tsv"
GLOB={'HBB':'ENSG00000244734','HBA1':'ENSG00000206172','HBA2':'ENSG00000188536','HBM':'ENSG00000206177',
 'HBQ1':'ENSG00000086506','HBG2':'ENSG00000196565','HBD':'ENSG00000223609','HBZ':'ENSG00000130656','HBG1':'ENSG00000213934'}
FLOOR=28  # recovery_days floor

print("[1] load counts + metadata ...")
counts_raw=pd.read_csv(COUNT_F,sep="\t",index_col=0)
si=pd.read_csv(SINFO_F,sep="\t"); covid=pd.read_csv(COVID_F,sep="\t"); hc=pd.read_csv(HC_F,sep="\t")
covid_sub=covid[['SampleID','Timepoint','Age','Sex','BMI','MM_SC','Recovery_days']]
mc=si[si['Group'].isin(['AC','RC'])].merge(covid_sub,on='SampleID',how='left')
mh=si[si['Group']=='HC'].merge(hc[['DataID_RNA','Age','Sex']],left_on='SampleID',right_on='DataID_RNA',how='left').drop(columns=['DataID_RNA'])
meta=pd.concat([mc,mh],ignore_index=True)
meta['Age']=pd.to_numeric(meta['Age'],errors='coerce'); meta['Age']=meta['Age'].fillna(meta['Age'].median())
meta['Sex']=meta['Sex'].fillna(meta['Sex'].mode()[0]).map({'M':0,'F':1}).fillna(0).astype(int)

# ---- CLEAN COHORT exclusion ----
meta['recd']=pd.to_numeric(meta['Recovery_days'],errors='coerce')
is_rc=meta['Group']=='RC'
is_L1=meta['SampleID'].str.endswith('-L1')
is_short=is_rc & (meta['recd']<FLOOR)   # V1 <28d (L1 recd is NaN -> not caught here)
excl_rc = is_rc & (is_L1 | is_short)
excluded=meta.loc[excl_rc,'SampleID'].tolist()
meta_clean=meta.loc[~excl_rc].copy()
n_rc_before=int(is_rc.sum()); n_rc_after=int((meta_clean['Group']=='RC').sum())
print(f"    RC before={n_rc_before}  excluded={len(excluded)} (L1={int((is_rc&is_L1).sum())}, V1<{FLOOR}d={int(is_short.sum())})  RC after={n_rc_after}")
pd.Series(excluded,name='excluded_SampleID').to_csv(f"{OUT}/data/excluded_samples.tsv",index=False,sep="\t")

print("[2] cell fractions + globin ...")
cf=pd.read_csv(CF_F,sep="\t",index_col=0)
agg=pd.DataFrame(index=cf.index)
agg['cf_B']=cf[['B cells naive','B cells memory','Plasma cells']].sum(axis=1)
agg['cf_CD4T']=cf[['T cells CD4 naive','T cells CD4 memory resting','T cells CD4 memory activated','T cells follicular helper','T cells regulatory (Tregs)','T cells gamma delta']].sum(axis=1)
agg['cf_CD8T']=cf['T cells CD8']; agg['cf_NK']=cf[['NK cells resting','NK cells activated']].sum(axis=1); agg['cf_Mono']=cf['Monocytes']
agg['cf_Other']=cf[['Macrophages M0','Macrophages M1','Macrophages M2','Dendritic cells resting','Dendritic cells activated','Mast cells resting','Mast cells activated','Eosinophils']].sum(axis=1)
meta_clean=meta_clean.merge(agg.reset_index().rename(columns={'index':'SampleID'}),on='SampleID',how='left')
cf_cols=['cf_B','cf_CD4T','cf_CD8T','cf_NK','cf_Mono','cf_Other']
lc=pd.read_csv(LOGCPM_F,sep="\t",index_col=0)
gr={s:lc.loc[[g for g in lc.index if g.startswith(e)][0]] for s,e in GLOB.items() if [g for g in lc.index if g.startswith(e)]}
gdf=pd.DataFrame(gr)
pc1=PCA(n_components=1).fit_transform(StandardScaler().fit_transform(gdf.fillna(gdf.mean())))
gs=pd.Series(pc1.flatten(),index=gdf.index,name='globin_score')
meta_clean=meta_clean.merge(gs.reset_index().rename(columns={'index':'SampleID'}),on='SampleID',how='left')
meta_clean['globin_score']=meta_clean['globin_score'].fillna(meta_clean['globin_score'].median())
for c in cf_cols: meta_clean[c]=meta_clean[c].fillna(meta_clean[c].median())
del lc

# reuse Acute gate sig set (unchanged)
g1=pd.read_csv(f"{DISC2}/data/gate1_ac_t0_vs_hc.tsv",sep="\t",index_col=0)
d_acute=set(g1[g1['sig_gate1']==True].index)
orig=pd.read_csv(f"{DISC2}/data/rna_dual_gate_markers_v2.tsv",sep="\t",index_col=0)
orig_pers=set(orig[orig['category']=='persistent'].index); orig_del=set(orig[orig['category']=='delayed'].index)
print(f"    d_acute(reused)={len(d_acute)}  orig persistent={len(orig_pers)} delayed={len(orig_del)}")

DESIGN='~Sex + Age + globin_score + cf_B + cf_CD4T + cf_CD8T + cf_NK + cf_Mono + cf_Other + condition'
cov=['Sex','Age','globin_score']+cf_cols
hc_ids=meta_clean[meta_clean['Group']=='HC']['SampleID'].tolist()
rc_ids=meta_clean[meta_clean['Group']=='RC']['SampleID'].tolist()
sub=counts_raw[rc_ids+hc_ids]; cpm=sub.div(sub.sum(0),axis=1)*1e6
passg=counts_raw.index[(cpm>1).sum(1)>=int(0.5*len(rc_ids+hc_ids))].tolist()
clin=meta_clean[meta_clean['Group'].isin(['RC','HC'])].set_index('SampleID')[cov+['Group']].rename(columns={'Group':'condition'})
clin['condition']=pd.Categorical(clin['condition'].astype(str),categories=['HC','RC'])
cnt=counts_raw.loc[passg,clin.index.tolist()].T.loc[clin.index.tolist()]
print(f"[3] gate2 DESeq2: {cnt.shape[1]} genes x {cnt.shape[0]} samples (RC={len(rc_ids)}, HC={len(hc_ids)}) ...")
dds=DESeq(DESeqDataSet(countData=cnt,clinicalData=clin,design=DESIGN),fitType='mean',quiet=True)
res=results_dds(dds,contrast=['condition','RC','HC'],alpha=0.05)
g2=pd.DataFrame({'log2FoldChange':np.asarray(res['log2FoldChange']),'adj_pvalue':np.asarray(res['adj_pvalue'])},index=list(res.index))
g2.index.name='gene_id'; g2.to_csv(f"{OUT}/data/gate2_rc_vs_hc_clean.tsv",sep="\t")

d_persist=set(g2[(g2['adj_pvalue']<0.05)&(g2['log2FoldChange'].abs()>=1)].index)
persistent=d_acute & d_persist; delayed=d_persist-d_acute; acute_only=d_acute-d_persist
print("\n=== CLEAN COHORT RNA DISCOVERY ===")
print(f"  Convalescent gate sig (RC vs HC): {len(d_persist)}   (orig 345)")
print(f"  Persistent (A∩B): {len(persistent)}   (orig {len(orig_pers)})")
print(f"  Delayed    (B\\A): {len(delayed)}    (orig {len(orig_del)})")
print(f"  Acute-only      : {len(acute_only)}")
print(f"  persistent ∩ orig263 = {len(persistent&orig_pers)}  (lost {len(orig_pers-persistent)}, gained {len(persistent-orig_pers)})")

# full marker table
allg=list(d_acute|d_persist)
m=pd.DataFrame(index=allg)
m['log2FC_gate2_clean']=g2['log2FoldChange'].reindex(allg); m['padj_gate2_clean']=g2['adj_pvalue'].reindex(allg)
m['log2FC_gate1']=g1['log2FoldChange'].reindex(allg); m['padj_gate1']=g1['adj_pvalue'].reindex(allg)
m['in_gate1']=m.index.isin(d_acute); m['in_gate2']=m.index.isin(d_persist)
m['category']=np.select([m['in_gate1']&m['in_gate2'],~m['in_gate1']&m['in_gate2']],['persistent','delayed'],default='acute_only')
m['in_orig_263']=m.index.isin(orig_pers)
m.index.name='gene_id'; m.to_csv(f"{OUT}/data/rna_markers_cleancohort.tsv",sep="\t")
print(f"\n[saved] {OUT}/data/rna_markers_cleancohort.tsv  (persistent list included)")
print(f"[saved] {OUT}/data/gate2_rc_vs_hc_clean.tsv , excluded_samples.tsv")
