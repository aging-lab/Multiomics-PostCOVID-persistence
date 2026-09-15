"""
Supplementary Figure — Long-term (~1 year) durability of the molecular scar.
Within-subject paired change from convalescence (V1) to ~1 year (L1) in the 10
re-contacted subjects (clean cohort; paired => no wave confound).
  A: RNA gene-set scores — Persistent (UP/DOWN) sustained; Delayed (UP/DOWN) resolves.
  B: Methylation delayed Δβ — declines toward baseline (n.s.).
Exploratory (n=10, non-random re-contacted subset). No suptitle.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats

R="PROJECT_ROOT/research/LongCOVID/Results"
NEW=R+"_updated_260714"
RNA_F=f"{NEW}/4.Trajectory/1.LME_GAM/data/trajectory_scores_clean.tsv"
METH_F=f"{R}/8.Recovery_duration/data/rc_recovery_methyl_scores.tsv"
OUT=f"{NEW}/manuscript/figures/figS7_longterm"

V1_COL="#457B9D"; L1_COL="#8B0000"
plt.rcParams.update({"font.family":"Arial","font.size":11,"axes.linewidth":0.9,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
LABEL_FS=11; TICK_FS=9.5; ANNOT_FS=9; LEGEND_FS=9; PANEL_FS=15

# ── RNA paired ──
r=pd.read_csv(RNA_F,sep="\t"); r=r[r.Group=="RC"]
by={}
for _,x in r.iterrows():
    b=x.SampleID.rsplit("-",1)[0]; s=x.SampleID.rsplit("-",1)[1]; by.setdefault(b,{})[s]=x
pairs=[(d["V1"],d["L1"]) for d in by.values() if "V1" in d and "L1" in d]
RNA_SETS=[("score_persistent_up","Persistent\nUP"),("score_persistent_down","Persistent\nDOWN"),
          ("score_delayed_up","Delayed\nUP"),("score_delayed_down","Delayed\nDOWN")]

# ── Methyl paired ──
m=pd.read_csv(METH_F,sep="\t"); mby={}
for _,x in m.iterrows(): mby.setdefault(x["subj"],{})[x["suf"]]=x["mscore"]
mpairs=[(d["V1"],d["L1"]) for d in mby.values() if "V1" in d and "L1" in d]

fig,(axA,axB)=plt.subplots(1,2,figsize=(12,5),gridspec_kw={"width_ratios":[3,1]})

# Panel A — RNA
for xc,(k,lab) in enumerate(RNA_SETS):
    v1=np.array([float(p[0][k]) for p in pairs]); l1=np.array([float(p[1][k]) for p in pairs])
    _,pw=stats.wilcoxon(v1,l1)
    for a,b in zip(v1,l1):
        axA.plot([xc-.16,xc+.16],[a,b],color="#bbb",lw=1,alpha=.6,zorder=1)
    axA.scatter([xc-.16]*len(v1),v1,c=V1_COL,s=26,zorder=2)
    axA.scatter([xc+.16]*len(l1),l1,c=L1_COL,s=26,zorder=2)
    axA.plot([xc-.16,xc+.16],[v1.mean(),l1.mean()],color="k",lw=3,zorder=3)
    star="**" if pw<0.01 else "*" if pw<0.05 else "n.s."
    ytop=max(v1.max(),l1.max())
    axA.text(xc,ytop+0.12,f"p={pw:.3f}\n{star}",ha="center",va="bottom",fontsize=ANNOT_FS-0.5,
             color="#c0392b" if pw<0.05 else "#555",fontweight="bold" if pw<0.05 else "normal")
axA.axhline(0,color="#bbb",lw=1,ls=":")
axA.set_xticks(range(4)); axA.set_xticklabels([s[1] for s in RNA_SETS],fontsize=TICK_FS)
axA.set_ylabel("Gene-set score (HC-anchored z)",fontsize=LABEL_FS)
axA.set_xlim(-.5,3.5); axA.set_ylim(top=axA.get_ylim()[1]+0.4)
# group brackets
axA.annotate("",xy=(0-.16,axA.get_ylim()[0]+0.1),xytext=(1+.16,axA.get_ylim()[0]+0.1),arrowprops=dict(arrowstyle="-",color="#888"))
axA.text(0.5,axA.get_ylim()[0]+0.02,"Persistent → sustained",ha="center",va="bottom",fontsize=ANNOT_FS,color="#333")
axA.text(2.5,axA.get_ylim()[0]+0.02,"Delayed → resolves",ha="center",va="bottom",fontsize=ANNOT_FS,color="#c0392b")
axA.legend(handles=[mlines.Line2D([],[],marker='o',ls='',mfc=V1_COL,mec='w',ms=8,label="Convalescent (V1)"),
                    mlines.Line2D([],[],marker='o',ls='',mfc=L1_COL,mec='w',ms=8,label="~1 year (L1)")],
           fontsize=LEGEND_FS,frameon=False,loc="upper right")
axA.tick_params(labelsize=TICK_FS)

# Panel B — Methyl
v1=np.array([p[0] for p in mpairs]); l1=np.array([p[1] for p in mpairs]); _,pw=stats.wilcoxon(v1,l1)
for a,b in zip(v1,l1): axB.plot([0-.16,0+.16],[a,b],color="#bbb",lw=1,alpha=.6,zorder=1)
axB.scatter([0-.16]*len(v1),v1,c=V1_COL,s=26,zorder=2); axB.scatter([0+.16]*len(l1),l1,c=L1_COL,s=26,zorder=2)
axB.plot([0-.16,0+.16],[v1.mean(),l1.mean()],color="k",lw=3,zorder=3)
axB.text(0,max(v1.max(),l1.max())+0.006,f"p={pw:.3f}\nn.s.",ha="center",va="bottom",fontsize=ANNOT_FS-0.5,color="#555")
axB.axhline(0,color="#bbb",lw=1,ls=":")
axB.set_xticks([0]); axB.set_xticklabels(["Delayed\npromoter-hyper"],fontsize=TICK_FS)
axB.set_ylabel("Methylation Δβ (vs HC)",fontsize=LABEL_FS); axB.set_xlim(-.6,.6)
axB.tick_params(labelsize=TICK_FS)

for ax,lab in [(axA,"A"),(axB,"B")]:
    ax.annotate(lab,xy=(0,1),xycoords="axes fraction",xytext=(-42,12),textcoords="offset points",
                fontsize=PANEL_FS,fontweight="bold",va="bottom",ha="left")
plt.tight_layout()
plt.savefig(OUT+".png",dpi=300,bbox_inches="tight"); plt.savefig(OUT+".pdf",bbox_inches="tight")
print("saved",OUT+".png",f"| RNA paired n={len(pairs)}, methyl paired n={len(mpairs)}")
