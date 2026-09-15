"""
Generates Supplementary Figures S1–S7 (clean cohort where applicable).
Naming: figS{N}_*.png/pdf in manuscript/figures/. Run with system python3 or venv.
  S1 QC             — RNA library size, methyl coverage (n_cpgs), deconvolution RMSE
  S2 Globin artifact— persistent v1(no globin) 501 vs v2(globin, clean) 268; globin gene logCPM
  S3 GSEA           — top pathways (Acute + Convalescent contrast) NES barplot
  S4 GMM BIC        — BIC/AIC vs K (K*=3) + entropy
  S5 Archetype genes— example gene trajectories per archetype (clean)
  S6 EpiDISH        — methylation cell fractions by group
  S7 cis-eQTM       — nominal pairs (bubble: rho vs -log10 p)
"""
import numpy as np, pandas as pd, warnings, gzip
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

R="PROJECT_ROOT/research/LongCOVID/Results"
NEW=R+"_updated_260714"
FIG=f"{NEW}/manuscript/figures"
plt.rcParams.update({"font.family":"Arial","font.size":11,"axes.linewidth":0.9,
    "axes.spines.top":False,"axes.spines.right":False,"pdf.fonttype":42,"figure.dpi":150})
GC={"AC":"#e74c3c","RC":"#3498db","HC":"#95a5a6"}
def save(fig,n):
    fig.savefig(f"{FIG}/{n}.png",dpi=300,bbox_inches="tight"); fig.savefig(f"{FIG}/{n}.pdf",bbox_inches="tight"); plt.close(fig); print("saved",n)
def parse_sym(g):
    p=str(g).split("_",1); return p[1] if len(p)>1 and not p[1].startswith("ENSG") else p[0].split(".")[0]

# ══ S1 — QC ══
def s1():
    si=pd.read_csv(f"{R}/2.Data_preparation/2.RNA_batch_correction/data/sample_info_v2.tsv",sep="\t",index_col=0)
    cnt=pd.read_csv(f"{R}/2.Data_preparation/2.RNA_batch_correction/data/rna_combat_corrected_v2.tsv",sep="\t",index_col=0)
    lib=np.log10(cnt.sum(0)); grp=si["Group"].reindex(lib.index)
    md=pd.read_csv(f"{R}/2.Data_preparation/6.Deconvolution/data/methyl_deconv_stats.tsv",sep="\t")
    csi=pd.read_csv(f"{R}/2.Data_preparation/5.Methyl_matrix/data/cpg_sample_info.tsv",sep="\t")
    md=md.merge(csi,on="SampleID",how="left")
    fig,ax=plt.subplots(1,3,figsize=(14,4.2))
    for i,g in enumerate(["AC","RC","HC"]):
        ax[0].boxplot([lib[grp==g].dropna()],positions=[i],widths=.6,patch_artist=True,
                      boxprops=dict(facecolor=GC[g],alpha=.7),medianprops=dict(color="k"))
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels(["AC","RC","HC"]); ax[0].set_ylabel("RNA library size (log10 counts)")
    ax[0].set_title("RNA sequencing depth",fontsize=11,fontweight="bold")
    for i,g in enumerate(["AC","RC","HC"]):
        v=md.loc[md.Group==g,"n_cpgs"].dropna()/1e6
        if len(v): ax[1].boxplot([v],positions=[i],widths=.6,patch_artist=True,boxprops=dict(facecolor=GC[g],alpha=.7),medianprops=dict(color="k"))
    ax[1].set_xticks(range(3)); ax[1].set_xticklabels(["AC","RC","HC"]); ax[1].set_ylabel("Covered CpGs (millions)")
    ax[1].set_title("Methylation coverage breadth",fontsize=11,fontweight="bold")
    for i,g in enumerate(["AC","RC","HC"]):
        v=md.loc[md.Group==g,"RMSE"].dropna()
        if len(v): ax[2].boxplot([v],positions=[i],widths=.6,patch_artist=True,boxprops=dict(facecolor=GC[g],alpha=.7),medianprops=dict(color="k"))
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels(["AC","RC","HC"]); ax[2].set_ylabel("EpiDISH deconvolution RMSE")
    ax[2].set_title("Methylation deconvolution fit",fontsize=11,fontweight="bold")
    for a,l in zip(ax,"ABC"): a.annotate(l,xy=(0,1),xycoords="axes fraction",xytext=(-38,12),textcoords="offset points",fontsize=14,fontweight="bold")
    fig.tight_layout(); save(fig,"figS1_qc")

# ══ S2 — Globin artifact ══
def s2():
    cmp=pd.read_csv(f"{R}/3.Discovery/1.RNA_DE_ver2/data/comparison_v1_v2.tsv",sep="\t").set_index("metric")
    # clean persistent = 268 (v2 col is original 263; use clean)
    v1p=int(cmp.loc["Persistent","v1"]); v2p=268
    lc=pd.read_csv(f"{R}/2.Data_preparation/4.RNA_normalization/data/rna_logcpm_v2.tsv.gz",sep="\t",index_col=0)
    glob={"HBA2":"ENSG00000188536","HBB":"ENSG00000244734","HBA1":"ENSG00000206172","HBG2":"ENSG00000196565","HBD":"ENSG00000223609"}
    means={}
    for s,e in glob.items():
        row=[g for g in lc.index if g.startswith(e)]
        if row: means[s]=lc.loc[row[0]].mean()
    fig,ax=plt.subplots(1,2,figsize=(10,4.3))
    ax[0].bar([0,1],[v1p,v2p],color=["#c0392b","#457B9D"],width=.55,edgecolor="white")
    for x,v in zip([0,1],[v1p,v2p]): ax[0].text(x,v+6,str(v),ha="center",fontweight="bold")
    ax[0].set_xticks([0,1]); ax[0].set_xticklabels(["v1\n(no globin corr.)","v2 clean\n(globin PC1 corr.)"])
    ax[0].set_ylabel("Persistent genes (n)"); ax[0].set_title("Globin PC1 correction removes artifact",fontsize=11,fontweight="bold")
    gs=sorted(means,key=means.get,reverse=True)
    ax[1].barh(range(len(gs)),[means[g] for g in gs],color="#e67e22",edgecolor="white")
    ax[1].set_yticks(range(len(gs))); ax[1].set_yticklabels(gs); ax[1].invert_yaxis()
    ax[1].set_xlabel("Mean logCPM"); ax[1].set_title("Hemoglobin genes dominate expression",fontsize=11,fontweight="bold")
    for a,l in zip(ax,"AB"): a.annotate(l,xy=(0,1),xycoords="axes fraction",xytext=(-40,12),textcoords="offset points",fontsize=14,fontweight="bold")
    fig.tight_layout(); save(fig,"figS2_globin")

# ══ S3 — GSEA ══
def s3():
    g=pd.read_csv(f"{R}/3.Discovery/1.RNA_DE_ver2/4.Functional/data/gsea/gsea_all_results.tsv",sep="\t")
    fig,ax=plt.subplots(1,2,figsize=(15,5.5))
    for k,(gate,title) in enumerate([("gate1_AC_T0","Acute contrast (AC_T0 vs HC)"),("gate2_RC","Convalescent contrast (RC vs HC)")]):
        sub=g[g.gate==gate].copy().sort_values("NES")
        sub=pd.concat([sub.head(8),sub.tail(8)]).drop_duplicates("Term").reset_index(drop=True)
        for yi,(_,r) in enumerate(sub.iterrows()):
            col="#457B9D" if r.NES<0 else "#E63946"
            ax[k].barh(yi,r.NES,color=col,edgecolor=col,linewidth=.6,
                       alpha=1.0 if r["FDR q-val"]<0.05 else .40,
                       hatch=None if r["FDR q-val"]<0.05 else "///")
        ax[k].set_yticks(range(len(sub))); ax[k].set_yticklabels([t[:42] for t in sub.Term],fontsize=7.5)
        ax[k].axvline(0,color="#444",lw=.8); ax[k].set_xlabel("NES"); ax[k].set_title(title,fontsize=11,fontweight="bold")
    for a,l in zip(ax,"AB"): a.annotate(l,xy=(0,1),xycoords="axes fraction",xytext=(-44,14),textcoords="offset points",fontsize=14,fontweight="bold")
    fig.tight_layout(); save(fig,"figS3_gsea")

# ══ S4 — GMM BIC ══
def s4():
    b=pd.read_csv(f"{R}/4.Trajectory_ver2/2.lcmm/data/gmm_bic_v2.tsv",sep="\t")
    fig,ax=plt.subplots(1,2,figsize=(11,4.3))
    ax[0].plot(b.K,b.BIC,"-o",color="#2980b9",label="BIC"); ax[0].plot(b.K,b.AIC,"-s",color="#e67e22",label="AIC")
    kstar=3; ax[0].axvline(kstar,color="#c0392b",ls="--",lw=1); ax[0].text(kstar,ax[0].get_ylim()[1],"K*=3",color="#c0392b",ha="center",va="top")
    ax[0].set_xlabel("Number of latent classes (K)"); ax[0].set_ylabel("Information criterion"); ax[0].legend(frameon=False)
    ax[0].set_title("GMM class selection",fontsize=11,fontweight="bold")
    ax[1].plot(b.K,b.mean_entropy,"-o",color="#8e44ad"); ax[1].set_xlabel("K"); ax[1].set_ylabel("Mean entropy")
    ax[1].set_title("Class assignment entropy",fontsize=11,fontweight="bold")
    for a,l in zip(ax,"AB"): a.annotate(l,xy=(0,1),xycoords="axes fraction",xytext=(-40,12),textcoords="offset points",fontsize=14,fontweight="bold")
    fig.tight_layout(); save(fig,"figS4_gmm_bic")

# ══ S5 — Archetype example genes (clean) ══
def s5():
    a=pd.read_csv(f"{NEW}/4.Trajectory/3.Archetypes/data/gene_archetypes_filtered.tsv",sep="\t")  # canonical, IG/pseudo removed (2026-07-30)
    a["sym"]=a["gene_id"].apply(parse_sym)
    order=["Escalated","Plateau","Partial","Recovered","Overshoot"]
    DISP={"Escalated":"Escalated","Plateau":"Sustained","Partial":"Partial","Recovered":"Resolving","Overshoot":"Overshoot"}  # render Plateau->Sustained, Recovered->Resolving
    COL={"Escalated":"#9B2226","Plateau":"#E63946","Partial":"#F4A261","Recovered":"#A8DADC","Overshoot":"#457B9D"}
    tps=["mu_HC","mu_T0","mu_T1","mu_T2","mu_T3","mu_RC"]; xpos=[-10,0,7,14,21,33]; xl=["HC","T0","T1","T2","T3","RC"]
    fig,axes=plt.subplots(1,5,figsize=(18,3.8),sharey=True)
    for ax,arch in zip(axes,order):
        sub=a[a.archetype==arch].copy()
        sub["absdev"]=(sub["mu_RC"]-sub["mu_HC"]).abs()
        ex=sub.sort_values("absdev",ascending=False).head(4)
        for _,r in ex.iterrows():
            sign=np.sign(r["mu_T0"]-r["mu_HC"]) or 1
            y=[(r[t]-r["mu_HC"])*sign for t in tps]
            ax.plot(xpos,y,"-o",ms=3,lw=1.3,alpha=.85,label=r["sym"][:12])
        ax.axhline(0,color="#bbb",lw=.7,ls="--"); ax.set_xticks(xpos); ax.set_xticklabels(xl,fontsize=7,rotation=45)
        ax.set_title(DISP[arch],fontsize=11,fontweight="bold",color=COL[arch]); ax.legend(fontsize=6.5,frameon=False)
    axes[0].set_ylabel("Δ z-score,\noriented to acute (HC=0)")
    fig.tight_layout(); save(fig,"figS5_archetype_genes")

# ══ S6 — EpiDISH cell fractions ══
def s6():
    cf=pd.read_csv(f"{R}/2.Data_preparation/6.Deconvolution/data/methyl_cell_fractions.tsv",sep="\t")
    csi=pd.read_csv(f"{R}/2.Data_preparation/5.Methyl_matrix/data/cpg_sample_info.tsv",sep="\t")
    cf=cf.merge(csi,on="SampleID",how="left")
    cells=["Neutro","CD4T","CD8T","NK","B","Mono","Eosino"]
    fig,ax=plt.subplots(figsize=(10,4.6))
    groups=["AC","RC","HC"]; x=np.arange(len(cells)); w=.25
    for i,g in enumerate(groups):
        m=[cf.loc[cf.Group==g,c].mean() for c in cells]
        ax.bar(x+(i-1)*w,m,w,label=g,color=GC[g],edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(cells); ax.set_ylabel("Mean cell fraction (EpiDISH)")
    ax.set_title("Methylation-derived leukocyte composition by group",fontsize=11,fontweight="bold"); ax.legend(frameon=False)
    fig.tight_layout(); save(fig,"figS6_epidish")

# ══ S7 — cis-eQTM nominal pairs ══
def s7():
    e=pd.read_csv(f"{R}/5.Integration/1.cis_eQTM/data/eqtm_nominal_p05.tsv",sep="\t")
    e["sym"]=e["gene_id"].apply(parse_sym); e["mlp"]=-np.log10(e["pval"])
    fig,ax=plt.subplots(figsize=(8.5,5.5))
    sc=ax.scatter(e["rho"],e["mlp"],s=40+120*(e["dist"].abs()<1e5),c=e["rho"],cmap="RdBu_r",vmin=-.3,vmax=.3,edgecolors="k",linewidths=.4)
    from adjustText import adjust_text
    lab=e.sort_values("mlp",ascending=False).drop_duplicates("sym").head(8)   # dedup by gene symbol
    texts=[ax.text(r["rho"],r["mlp"],r["sym"],fontsize=7.5) for _,r in lab.iterrows()]
    adjust_text(texts,ax=ax,expand=(1.5,2.0),arrowprops=dict(arrowstyle="-",color="#888",lw=0.6))
    ax.axvline(0,color="#bbb",lw=.8); ax.axhline(-np.log10(0.05),color="#888",ls="--",lw=.8)
    ax.set_xlabel("Partial Spearman ρ (gene–CpG, cell-adjusted)"); ax.set_ylabel("−log10 nominal p")
    ax.set_title(f"cis-eQTM nominal pairs (n={len(e)}; all FDR>0.05)",fontsize=11,fontweight="bold")
    plt.colorbar(sc,ax=ax,label="ρ")
    fig.tight_layout(); save(fig,"figS9_eqtm")

for f in (s1,s2,s3,s4,s5,s6,s7):
    try: f()
    except Exception as ex: print("FAIL",f.__name__,ex)
print("done S1-S7")
