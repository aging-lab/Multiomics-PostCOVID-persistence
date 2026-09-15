"""
Severity (MM vs SC) recompute on CLEAN cohort (v3) -> Fig 5.
Uses clean trajectory_scores; RC restricted to clean convalescent (stratum).
Cohen's d + Mann-Whitney U per timepoint per gene-set score (matches
severity_trajectory.py). Headline: T1 max gap -> RC convergence (severity-independent).
Run with .venv_deseq python.
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
from scipy import stats

R="PROJECT_ROOT/research/LongCOVID/Results"
NEW=R+"_updated_260714"
TRAJ=f"{NEW}/4.Trajectory/1.LME_GAM/data/trajectory_scores_clean.tsv"
META=f"{R}/1.Metadata/COVID-19_sample_metadata.tsv"
OUT=f"{NEW}/5.Integration/3.Severity/data"; os.makedirs(OUT,exist_ok=True)
SCORES=["score_persistent_up","score_persistent_down","score_delayed_up","score_delayed_down","score_acute_only_up","score_acute_only_down"]

sc=pd.read_csv(TRAJ,sep="\t")
meta=pd.read_csv(META,sep="\t")[["SampleID","MM_SC"]].drop_duplicates("SampleID")
df=sc.merge(meta,on="SampleID",how="left")
# RC: clean convalescent only
df=df[~((df["Group"]=="RC")&(df["stratum"]!="convalescent"))]

def cohend(a,b): return (a.mean()-b.mean())/np.sqrt((a.std()**2+b.std()**2)/2)
rows=[]
for tp,sub in [("T0",df[df.Timepoint=="T0"]),("T1",df[df.Timepoint=="T1"]),
               ("T2",df[df.Timepoint=="T2"]),("T3",df[df.Timepoint=="T3"]),
               ("RC",df[df.Group=="RC"])]:
    mm=sub[sub.MM_SC=="MM"]; scg=sub[sub.MM_SC=="SC"]
    for col in SCORES:
        a=mm[col].dropna().values; b=scg[col].dropna().values
        if len(a)<3 or len(b)<3: continue
        u,p=stats.mannwhitneyu(a,b,alternative="two-sided")
        rows.append(dict(timepoint=tp,score=col,n_MM=len(a),n_SC=len(b),
                         mean_MM=a.mean(),mean_SC=b.mean(),cohen_d=cohend(a,b),pval=p))
res=pd.DataFrame(rows)
# FDR within timepoint
from statsmodels.stats.multitest import multipletests
out=[]
for tp,g in res.groupby("timepoint"):
    g=g.copy(); g["fdr"]=multipletests(g["pval"],method="fdr_bh")[1]; out.append(g)
res=pd.concat(out)
res.to_csv(f"{OUT}/severity_mwu_clean.tsv",sep="\t",index=False)

print("=== Severity MM vs SC (clean cohort) — key scores ===")
for col in ["score_persistent_up","score_delayed_up"]:
    print(f"\n{col}:")
    for tp in ["T0","T1","T2","T3","RC"]:
        r=res[(res.timepoint==tp)&(res.score==col)]
        if len(r):
            r=r.iloc[0]; sig="*" if r.fdr<0.05 else ""
            print(f"  {tp}: d={r.cohen_d:+.3f}  p={r.pval:.3f}  fdr={r.fdr:.3f}{sig}  (MM={r.n_MM},SC={r.n_SC})")
print(f"\n[saved] {OUT}/severity_mwu_clean.tsv")
print("orig headline: T1 persistent-UP d=0.96, delayed-UP d=1.04 ; RC persistent-UP d=0.105, delayed-UP d=-0.07 (n.s.)")
