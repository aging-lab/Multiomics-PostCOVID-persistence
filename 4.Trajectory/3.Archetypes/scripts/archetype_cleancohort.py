"""
5-Archetype / Recovery Index recompute on CLEAN cohort (v3).
Replicates gene_trajectory_gmm_v2.py RI logic: per-gene OLS with one-hot
timepoint dummies + covariates -> adjusted mean mu[t] -> RI=(mu_RC-mu_HC)/(mu_T0-mu_HC)
-> 5-archetype by RI rule. CLEAN: persistent=268; RC restricted to clean
convalescent (exclude -L1 and V1<28d) so mu_RC reflects the clean cohort.
Run with .venv_deseq python.
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

R="PROJECT_ROOT/research/LongCOVID/Results"
NEW=R+"_updated_260714"
RNA_MATRIX=f"{R}/2.Data_preparation/2.RNA_batch_correction/data/rna_combat_corrected_v2.tsv"
SAMPLE_INFO=f"{R}/2.Data_preparation/2.RNA_batch_correction/data/sample_info_v2.tsv"
META_COVID=f"{R}/1.Metadata/COVID-19_sample_metadata.tsv"
HC_F=f"{R}/1.Metadata/HC_sample_metadata.tsv"
CF_F=f"{R}/2.Data_preparation/6.Deconvolution/data/rna_cell_fractions_v2.tsv"
MARKERS_F=f"{NEW}/3.Discovery/1.RNA_DE/data/rna_markers_cleancohort.tsv"
OUT=f"{NEW}/4.Trajectory/3.Archetypes/data"

GLOBIN=["HBB","HBA1","HBA2","HBM","HBQ1","HBG2","HBD","HBZ","HBG1"]
TP_ORDER=['HC','T0','T1','T2','T3','RC']; FLOOR=28
def parse_symbol(g):
    p=g.split("_",1)
    if len(p)==2: return p[0].split(".")[0] if p[1].startswith("ENSG") else p[1]
    return g.split(".")[0]

print("[1] metadata + clean RC set ...")
si=pd.read_csv(SAMPLE_INFO,sep="\t"); meta=pd.read_csv(META_COVID,sep="\t"); hc=pd.read_csv(HC_F,sep="\t")
hc_ids=si[si["Group"]=="HC"]["SampleID"].tolist()
meta_ac=meta[meta["Has_RNA"]=="Y"].copy(); meta_ac["Visit_date"]=pd.to_datetime(meta_ac["Visit_date"])
t0d=meta_ac[meta_ac["Timepoint"]=="T0"].set_index("SubjectID")["Visit_date"]
meta_ac["days_from_T0"]=(meta_ac["Visit_date"]-meta_ac["SubjectID"].map(t0d)).dt.days.fillna(0)
meta_ac=meta_ac[(meta_ac["days_from_T0"]<=60)&(meta_ac["Timepoint"].isin(["T0","T1","T2","T3"]))]
# clean convalescent RC: V1, recovery>=28, no L1
rcm=meta[meta["Group"]=="RC"].copy(); rcm["recd"]=pd.to_numeric(rcm["Recovery_days"],errors="coerce")
clean_rc=rcm[(~rcm["SampleID"].str.endswith("-L1"))&(rcm["recd"]>=FLOOR)]["SampleID"].tolist()
print(f"    HC={len(hc_ids)}  AC_tp={meta_ac['Timepoint'].value_counts().to_dict()}  clean RC={len(clean_rc)}")

# per-sample Sex/Age
covid_sub=meta[['SampleID','Sex','Age']]
hc_sub=si[si['Group']=='HC'].merge(hc[['DataID_RNA','Age','Sex']],left_on='SampleID',right_on='DataID_RNA',how='left')[['SampleID','Age','Sex']]
allmeta=pd.concat([covid_sub,hc_sub],ignore_index=True).drop_duplicates('SampleID').set_index('SampleID')
# cell fractions (aggregate 6, use 5)
cf=pd.read_csv(CF_F,sep="\t",index_col=0)
cf6=pd.DataFrame(index=cf.index)
cf6['cf_B']=cf[['B cells naive','B cells memory','Plasma cells']].sum(1)
cf6['cf_CD4T']=cf[['T cells CD4 naive','T cells CD4 memory resting','T cells CD4 memory activated','T cells follicular helper','T cells regulatory (Tregs)','T cells gamma delta']].sum(1)
cf6['cf_CD8T']=cf['T cells CD8']; cf6['cf_NK']=cf[['NK cells resting','NK cells activated']].sum(1); cf6['cf_Mono']=cf['Monocytes']

print("[2] logCPM + globin PC1 ...")
markers=pd.read_csv(MARKERS_F,sep="\t"); markers['symbol']=markers['gene_id'].apply(parse_symbol)
persistent=markers[markers['category']=='persistent']['symbol'].tolist()
rna=pd.read_csv(RNA_MATRIX,sep="\t",index_col=0); rna.index=[parse_symbol(g) for g in rna.index]; rna=rna.groupby(level=0).mean()
logcpm=np.log2(rna.divide(rna.sum(0),axis=1)*1e6+1)
gg=[g for g in GLOBIN if g in logcpm.index]
pc1=PCA(1).fit_transform(StandardScaler().fit_transform(logcpm.loc[gg].T.values)).ravel()
Xp=np.column_stack([np.ones(len(pc1)),pc1]); LT=logcpm.values.T; bg=np.linalg.lstsq(Xp,LT,rcond=None)[0]
resid=LT-Xp@bg; hc_idx=[i for i,s in enumerate(logcpm.columns) if s in hc_ids]
logcpm_adj=pd.DataFrame((resid+resid[hc_idx].mean(0)).T,index=logcpm.index,columns=logcpm.columns)
persist_avail=[g for g in persistent if g in logcpm_adj.index]
expr=logcpm_adj.loc[persist_avail]; print(f"    persistent avail: {len(persist_avail)}/{len(persistent)}")

print("[3] per-gene OLS -> mu -> RI ...")
tp_label={s:'HC' for s in hc_ids}
for _,r in meta_ac.iterrows(): tp_label[r['SampleID']]=r['Timepoint']
for s in clean_rc: tp_label[s]='RC'          # only clean convalescent RC
valid=[s for s in expr.columns if s in tp_label and s in allmeta.index and s in cf6.index]
mo=allmeta.loc[valid].copy(); mo['tp']=[tp_label[s] for s in valid]
mo['Sex_bin']=(mo['Sex']=='F').astype(float)
mo['Age']=pd.to_numeric(mo['Age'],errors='coerce'); mo['Age']=mo['Age'].fillna(mo['Age'].median())
mo['Age_z']=(mo['Age']-mo['Age'].mean())/mo['Age'].std()
mo=mo.join(cf6,how='left')
mo['glob_score']=pc1[[list(logcpm.columns).index(s) for s in valid]]
tpd=pd.get_dummies(mo['tp'],prefix='tp').reindex(columns=[f'tp_{t}' for t in TP_ORDER],fill_value=0)
cov=['Sex_bin','Age_z','glob_score','cf_B','cf_CD4T','cf_CD8T','cf_NK','cf_Mono']
mo[cov]=mo[cov].fillna(mo[cov].median())
X=np.column_stack([tpd.values,mo[cov].values]).astype(np.float64); Y=expr[valid].values
print(f"    design {X.shape}, genes {Y.shape}; RC in OLS={ (mo['tp']=='RC').sum() }")
beta=np.linalg.lstsq(X,Y.T,rcond=None)[0]
mu={t:beta[i,:] for i,t in enumerate(TP_ORDER)}
denom=mu['T0']-mu['HC']
ri=np.where(np.abs(denom)>0.01,(mu['RC']-mu['HC'])/denom,0.0); ri_clip=np.clip(ri,-4,4)
dirmap=markers.drop_duplicates('symbol').set_index('symbol')['log2FC_gate2_clean']
dirs=np.array(['up' if dirmap.get(g,1)>=0 else 'down' for g in persist_avail])

def assign(r):
    if r>1.5: return 'Escalated'
    if r>=0.7: return 'Plateau'
    if r>=0.2: return 'Partial'
    if r>=-0.5: return 'Recovered'
    return 'Overshoot'
arch=[assign(r) for r in ri_clip]
tv=pd.DataFrame({'gene_id':persist_avail,'mu_HC':mu['HC'],'mu_T0':mu['T0'],'mu_T1':mu['T1'],
                 'mu_T2':mu['T2'],'mu_T3':mu['T3'],'mu_RC':mu['RC'],'RI':ri,'RI_clip':ri_clip,
                 'direction':dirs,'archetype':arch}).set_index('gene_id')
os.makedirs(OUT,exist_ok=True); tv.to_csv(f"{OUT}/gene_archetypes_clean.tsv",sep="\t")

n=len(arch)
print("\n=== 5-Archetype (clean cohort, persistent=%d) ==="%n)
for a in ['Escalated','Plateau','Partial','Recovered','Overshoot']:
    c=arch.count(a); print(f"  {a:10} {c:3}  ({100*c/n:.1f}%)")
esc_sus=arch.count('Escalated')+arch.count('Plateau')
print(f"  Escalated+Plateau (non-recovering): {esc_sus} ({100*esc_sus/n:.1f}%)")
print(f"\n[saved] {OUT}/gene_archetypes_clean.tsv")
