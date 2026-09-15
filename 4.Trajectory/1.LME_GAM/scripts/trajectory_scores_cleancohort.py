"""
Gene-set z-score recompute on CLEAN cohort (v3) — foundation for downstream.
Replicates trajectory_v2.py score logic (globin-PC1-removed logCPM, HC-anchored
per-gene z, mean over gene set) but with the CLEAN dual-gate markers
(rna_markers_cleancohort.tsv: persistent 268 / delayed / acute_only) and the
clean gate2 lfc for UP/DOWN direction. Per-sample scores are cohort-independent
(HC-anchored); RC samples get a `stratum` flag so downstream uses clean
convalescent (V1, recovery>=28) and treats L1(~1yr)/short separately.
Run with .venv_deseq python (has sklearn).
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

R="PROJECT_ROOT/research/LongCOVID/Results"
NEW=R+"_updated_260714"
RNA_MATRIX=f"{R}/2.Data_preparation/2.RNA_batch_correction/data/rna_combat_corrected_v2.tsv"
SAMPLE_INFO=f"{R}/2.Data_preparation/2.RNA_batch_correction/data/sample_info_v2.tsv"
META_COVID=f"{R}/1.Metadata/COVID-19_sample_metadata.tsv"
MARKERS_F=f"{NEW}/3.Discovery/1.RNA_DE/data/rna_markers_canonical_filtered.tsv"  # canonical, IG/pseudo removed (2026-07-30)
GATE2_F=f"{NEW}/3.Discovery/1.RNA_DE/data/gate2_rc_vs_hc_clean.tsv"
GATE1_F=f"{R}/3.Discovery/1.RNA_DE_ver2/data/gate1_ac_t0_vs_hc.tsv"
OUT=f"{NEW}/4.Trajectory/1.LME_GAM/data"

GLOBIN=["HBB","HBA1","HBA2","HBM","HBQ1","HBG2","HBD","HBZ","HBG1"]
TP_DAYS={"HC":-30,"T0":0,"T1":6,"T2":14,"T3":22,"RC":100}
L1_DAYS={"C19-R001":462,"C19-R010":477,"C19-R024":467,"C19-R028":472,"C19-R030":450,
         "C19-R078":391,"C19-R081":372,"C19-R091":370,"C19-R092":334,"C19-R096":336}
FLOOR=28

def parse_symbol(g):
    p=g.split("_",1)
    if len(p)==2: return p[0].split(".")[0] if p[1].startswith("ENSG") else p[1]
    return g.split(".")[0]

print("[1] metadata + markers ...")
si=pd.read_csv(SAMPLE_INFO,sep="\t"); meta=pd.read_csv(META_COVID,sep="\t")
hc_ids=si[si["Group"]=="HC"]["SampleID"].tolist()
rc_ids=si[si["Group"]=="RC"]["SampleID"].tolist()
meta_ac=meta[meta["Has_RNA"]=="Y"].copy()
meta_ac["Visit_date"]=pd.to_datetime(meta_ac["Visit_date"])
t0d=meta_ac[meta_ac["Timepoint"]=="T0"].set_index("SubjectID")["Visit_date"]
meta_ac["T0_date"]=meta_ac["SubjectID"].map(t0d)
meta_ac["days_from_T0"]=(meta_ac["Visit_date"]-meta_ac["T0_date"]).dt.days.fillna(0)
meta_ac=meta_ac[(meta_ac["days_from_T0"]<=60)&(meta_ac["Timepoint"].isin(["T0","T1","T2","T3"]))]
rc_meta=meta[meta["Group"]=="RC"][["SampleID","Recovery_days"]].copy()
rc_meta["recd"]=pd.to_numeric(rc_meta["Recovery_days"],errors="coerce")

markers=pd.read_csv(MARKERS_F,sep="\t"); markers["symbol"]=markers["gene_id"].apply(parse_symbol)
gene_sets={c:markers[markers["category"]==c]["symbol"].tolist() for c in ["persistent","delayed","acute_only"]}
for c,g in gene_sets.items(): print(f"    {c}: {len(g)}")

print("[2] logCPM + globin PC1 removal ...")
rna=pd.read_csv(RNA_MATRIX,sep="\t",index_col=0)
rna.index=[parse_symbol(g) for g in rna.index]; rna=rna.groupby(level=0).mean()
lib=rna.sum(0); logcpm=np.log2(rna.divide(lib,axis=1)*1e6+1)
gg=[g for g in GLOBIN if g in logcpm.index]
pc1=PCA(1).fit_transform(StandardScaler().fit_transform(logcpm.loc[gg].T.values)).ravel()
Xp=np.column_stack([np.ones(len(pc1)),pc1]); LT=logcpm.values.T
b=np.linalg.lstsq(Xp,LT,rcond=None)[0]; resid=LT-Xp@b
hc_idx=[i for i,s in enumerate(logcpm.columns) if s in hc_ids]
logcpm_adj=pd.DataFrame((resid+resid[hc_idx].mean(0)).T,index=logcpm.index,columns=logcpm.columns)

print("[3] gate lfc (direction) ...")
def lfc_series(f):
    d=pd.read_csv(f,sep="\t"); d["symbol"]=d["gene_id"].apply(parse_symbol)
    return d.drop_duplicates("symbol").set_index("symbol")["log2FoldChange"]
gate2=lfc_series(GATE2_F); gate1=lfc_series(GATE1_F)

def geneset_score(genes):
    avail=[g for g in genes if g in logcpm_adj.index]
    if not avail: return pd.Series(np.nan,index=logcpm_adj.columns),[]
    mat=logcpm_adj.loc[avail]; hcm=mat.columns.isin(hc_ids)
    mu=mat.loc[:,hcm].mean(1); sd=mat.loc[:,hcm].std(1).replace(0,np.nan)
    return mat.sub(mu,axis=0).div(sd,axis=0).mean(0),avail
def updown(av,cat):
    ref=gate2 if cat in ("persistent","delayed") else gate1
    return [g for g in av if g in ref.index and ref[g]>0],[g for g in av if g in ref.index and ref[g]<0]

scores={}; scores_up={}; scores_dn={}
for cat,genes in gene_sets.items():
    sc,av=geneset_score(genes); scores[cat]=sc
    up,dn=updown(av,cat); scores_up[cat]=geneset_score(up)[0]; scores_dn[cat]=geneset_score(dn)[0]
    print(f"    {cat}: {len(av)}/{len(genes)}  UP{len(up)}/DN{len(dn)}")

print("[4] build trajectory_scores ...")
def rc_stratum(sid):
    subj=sid.rsplit('-',1)[0]; suf=sid.rsplit('-',1)[1]
    if suf=="L1": return "long_term_1yr"
    rd=rc_meta.set_index("SampleID")["recd"].get(sid,np.nan)
    return "acute_overlap" if (pd.notna(rd) and rd<FLOOR) else "convalescent"
rows=[]
for sid in hc_ids:
    rows.append(dict(SampleID=sid,SubjectID=sid,Group="HC",Timepoint="HC",days_from_T0=TP_DAYS["HC"],stratum="HC",
                     **{f"score_{c}":scores[c].get(sid,np.nan) for c in scores}))
for _,m in meta_ac.iterrows():
    sid=m["SampleID"]
    if sid not in logcpm_adj.columns: continue
    rows.append(dict(SampleID=sid,SubjectID=m["SubjectID"],Group="AC",Timepoint=m["Timepoint"],
                     days_from_T0=m["days_from_T0"],stratum="AC",
                     **{f"score_{c}":scores[c].get(sid,np.nan) for c in scores}))
for sid in rc_ids:
    if sid not in logcpm_adj.columns: continue
    subj=sid.rsplit('-',1)[0]
    rd=rc_meta.set_index("SampleID")["recd"].get(sid,np.nan)
    if pd.isna(rd) and sid.endswith("-L1"): rd=L1_DAYS.get(subj,np.nan)
    rows.append(dict(SampleID=sid,SubjectID=sid,Group="RC",Timepoint="RC",
                     days_from_T0=rd if pd.notna(rd) else 100,stratum=rc_stratum(sid),
                     **{f"score_{c}":scores[c].get(sid,np.nan) for c in scores}))
traj=pd.DataFrame(rows)
for c in ["persistent","delayed","acute_only"]:
    traj[f"score_{c}_up"]=traj["SampleID"].map(scores_up[c])
    traj[f"score_{c}_down"]=traj["SampleID"].map(scores_dn[c])
import os; os.makedirs(OUT,exist_ok=True)
traj.to_csv(f"{OUT}/trajectory_scores_clean.tsv",sep="\t",index=False)
print(f"[saved] {OUT}/trajectory_scores_clean.tsv  {traj.shape}")
print("\nRC stratum counts:",traj[traj.Group=='RC']['stratum'].value_counts().to_dict())
print("\nMean score by timepoint (clean convalescent RC only):")
tt=traj[(traj.stratum!='acute_overlap')&(traj.stratum!='long_term_1yr')]
print(tt.groupby("Timepoint")[["score_persistent","score_delayed"]].mean().round(3).reindex(["HC","T0","T1","T2","T3","RC"]).to_string())
