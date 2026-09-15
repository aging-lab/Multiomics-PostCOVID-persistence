"""
Methyl delayed-promoter-hyper CpG trajectory on CLEAN cohort (v3) -> Fig 3B headline.
Target = clean delayed CpGs (303) that are hyper (clean Δβ_gate2>0) AND promoter
(genomic feature from existing annotation; cohort-independent). 37 hyper CpGs are
not in the existing annotation file -> pending full genomic annotation (caveat).
Mean β per timepoint (HC/T0/T2/T3/RC), RC = clean convalescent methyl cohort
(exclude -L1 and V1<28d). Headline: Δβ(T0-HC) ~0 vs Δβ(RC-HC) = max (temporal cascade).
Run with .venv_deseq python.
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")

R="PROJECT_ROOT/research/LongCOVID/Results"
NEW=R+"_updated_260714"
BETA=f"{R}/2.Data_preparation/5.Methyl_matrix/data/cpg_matrix_beta.npz"
ANNO=f"{R}/3.Discovery/2.Methyl_DM/3.Functional/data/methyl_dm_annotated.tsv"
CLEAN_MARK=f"{NEW}/3.Discovery/2.Methyl_DM/data/methyl_markers_cleancohort_clean.tsv"
META=f"{R}/1.Metadata/COVID-19_sample_metadata.tsv"
HC_F=f"{R}/1.Metadata/HC_sample_metadata.tsv"
CPG_SI=f"{R}/2.Data_preparation/5.Methyl_matrix/data/cpg_sample_info.tsv"
OUT=f"{NEW}/4.Trajectory/4.Methyl_trajectory/data"; os.makedirs(OUT,exist_ok=True)
FLOOR=28

print("[1] target: clean delayed promoter-hyper CpGs ...")
anno=pd.read_csv(ANNO,sep="\t").set_index('cpg_id')
nd=pd.read_csv(CLEAN_MARK,sep="\t"); nd=nd[nd['category']=='delayed'].copy()
nd['feature']=nd['cpg_id'].map(anno['feature'])
target=nd[(nd['delta_beta_gate2']>0)&(nd['feature']=='promoter')]['cpg_id'].tolist()
print(f"    delayed={len(nd)}  promoter-hyper(annotated)={len(target)}  (37 hyper unannotated pending)")

print("[2] beta matrix subset ...")
z=np.load(BETA,allow_pickle=True)
allc=[str(c) for c in z['cpg_ids']]; alls=[str(s) for s in z['sample_ids']]
cidx={c:i for i,c in enumerate(allc)}
tidx=[cidx[c] for c in target if c in cidx]; avail=[c for c in target if c in cidx]
beta=z['matrix'][tidx,:].astype(np.float32); del z
print(f"    {beta.shape} ({len(avail)} CpGs)")

print("[3] timepoint assignment (RC=clean convalescent) ...")
meta=pd.read_csv(META,sep="\t"); hc=pd.read_csv(HC_F,sep="\t"); si=pd.read_csv(CPG_SI,sep="\t")
hc_methyl=set(hc['DataID_Methyl'].dropna().tolist()) | set(si[si['Group']=='HC']['SampleID'])
meta['recd']=pd.to_numeric(meta['Recovery_days'],errors='coerce')
tp={}
for s in alls:
    if s in hc_methyl: tp[s]='HC'
for _,r in meta.iterrows():
    s=r['SampleID']
    if r['Group']=='AC' and r['Timepoint'] in ['T0','T2','T3']: tp[s]=r['Timepoint']
    elif r['Group']=='RC':
        if s.endswith('-L1'): continue                      # exclude L1
        if pd.notna(r['recd']) and r['recd']<FLOOR: continue # exclude short
        tp[s]='RC'
TP=['HC','T0','T2','T3','RC']
idxmap={t:[i for i,s in enumerate(alls) if tp.get(s)==t] for t in TP}
print("    counts:",{t:len(idxmap[t]) for t in TP})

print("[4] mean beta per timepoint ...")
mu={t:(np.nanmean(beta[:,idxmap[t]],axis=1) if idxmap[t] else np.full(len(avail),np.nan)) for t in TP}
res=pd.DataFrame(mu,index=avail); res.index.name='cpg_id'
res['delta_HC_to_RC']=res['RC']-res['HC']
res.to_csv(f"{OUT}/methyl_cpg_trajectory_clean.tsv",sep="\t")
print(f"\n=== Delayed promoter-hyper trajectory (clean, n={len(avail)}) ===")
for t in TP:
    m=np.nanmean(mu[t]); print(f"    {t}: mean β={m:.4f}  Δβ(vs HC)={m-np.nanmean(mu['HC']):+.4f}")
print(f"\n  HEADLINE  Δβ(T0-HC)={np.nanmean(mu['T0'])-np.nanmean(mu['HC']):+.4f}"
      f"   Δβ(RC-HC)={np.nanmean(mu['RC'])-np.nanmean(mu['HC']):+.4f}   (orig: +0.009 -> +0.074)")
print(f"[saved] {OUT}/methyl_cpg_trajectory_clean.tsv")
