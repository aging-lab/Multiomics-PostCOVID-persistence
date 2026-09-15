"""
Supplementary Figure S8 — COX8A cis-eQTM scatter (CpG methylation × RNA expression).
The lead persistent–delayed pair: COX8A (ENSG00000176340) × chr11_64304843
(ρ=−0.18, nominal p=0.002, FDR n.s.). Per-sample methylation β vs RNA logCPM.
Exploratory (companion to Fig S9 mediation).
"""
import numpy as np, pandas as pd, warnings, re
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

R="PROJECT_ROOT/research/LongCOVID/Results"
NEW=R+"_updated_260714"; FIG=f"{NEW}/manuscript/figures"
plt.rcParams.update({"font.family":"Arial","font.size":11,"axes.linewidth":0.9,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42})
CPG="chr11_64304843"; ENSG="ENSG00000176340"
GC={"AC":"#e74c3c","RC":"#3498db","HC":"#95a5a6"}

print("[1] RNA COX8A logCPM ...")
lc=pd.read_csv(f"{R}/2.Data_preparation/4.RNA_normalization/data/rna_logcpm_v2.tsv.gz",sep="\t",index_col=0)
row=[g for g in lc.index if g.startswith(ENSG)]
rna=lc.loc[row[0]]; del lc

print("[2] methyl CpG beta ...")
z=np.load(f"{R}/2.Data_preparation/5.Methyl_matrix/data/cpg_matrix_beta.npz",allow_pickle=True)
cids=[str(c) for c in z['cpg_ids']]; sids=[str(s) for s in z['sample_ids']]
ci=cids.index(CPG) if CPG in cids else None
beta=pd.Series(z['matrix'][ci,:].astype(float),index=sids); del z

print("[3] match samples + group ...")
si=pd.read_csv(f"{R}/2.Data_preparation/5.Methyl_matrix/data/cpg_sample_info.tsv",sep="\t").set_index("SampleID")["Group"]
common=[s for s in beta.index if s in rna.index]
df=pd.DataFrame({"beta":beta[common],"rna":rna[common]}).dropna()
df["grp"]=si.reindex(df.index).fillna("HC")
df=df[df["beta"].notna()&(df["rna"]>0)]
rho,p=stats.spearmanr(df["beta"],df["rna"])
print(f"    n={len(df)}  Spearman rho={rho:+.3f} p={p:.4f}")

fig,ax=plt.subplots(figsize=(6.5,5.5))
for g in ["AC","RC","HC"]:
    s=df[df.grp==g]
    ax.scatter(s["beta"],s["rna"],c=GC[g],s=22,alpha=.6,edgecolors="none",label=f"{g} (n={len(s)})")
# regression line
b=np.polyfit(df["beta"],df["rna"],1); xs=np.linspace(df["beta"].min(),df["beta"].max(),40)
ax.plot(xs,np.polyval(b,xs),color="k",lw=1.6,ls="--")
ax.set_xlabel(f"CpG methylation β  ({CPG})"); ax.set_ylabel("COX8A expression (logCPM)")
ax.set_title(f"COX8A cis-eQTM (ρ={rho:+.2f}, nominal p={p:.3f}, FDR n.s.)",fontsize=11,fontweight="bold")
ax.legend(frameon=False,fontsize=9)
fig.tight_layout()
fig.savefig(f"{FIG}/figS10_cox8a.png",dpi=300,bbox_inches="tight"); fig.savefig(f"{FIG}/figS10_cox8a.pdf",bbox_inches="tight")
print("saved figS10_cox8a")
