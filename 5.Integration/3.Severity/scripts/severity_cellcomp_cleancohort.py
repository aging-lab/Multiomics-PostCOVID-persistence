#!/usr/bin/env python3
"""T1 cell-composition ANCOVA on the CLEAN, FILTERED cohort (2026-07-30).
Re-runs severity_cellcomp.py's adjustment with the recomputed gene-set scores
(canonical, IG/pseudo removed -> persistent 208 / delayed 61 / acute_only 953).
Same methodology: d_raw = Cohen's d MM vs SC at T1; d_adj = Cohen's d on OLS
residuals of score ~ cell_fractions; ANCOVA = score ~ MM_SC + cell_fractions.
Writes t1_score_adjusted.tsv (read by fig5 Panel C). Cell-fraction inputs are
unchanged, so t1_cellcomp_comparison.tsv (Fig 5B) is NOT regenerated.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from scipy import stats
import statsmodels.formula.api as smf

R    = "PROJECT_ROOT/research/LongCOVID/Results"
NEW  = R + "_updated_260714"
TRAJ = f"{NEW}/4.Trajectory/1.LME_GAM/data/trajectory_scores_clean.tsv"   # recomputed (filtered sets)
META = f"{R}/1.Metadata/COVID-19_sample_metadata.tsv"
RNA_CF = f"{R}/2.Data_preparation/6.Deconvolution/data/rna_cell_fractions_v2.tsv"
OUT  = f"{NEW}/5.Integration/3.Severity/data/t1_score_adjusted.tsv"

SCORE_COLS = ["score_persistent_up","score_persistent_down","score_delayed_up",
              "score_delayed_down","score_acute_only_up","score_acute_only_down"]
LABELS = {"score_persistent_up":"Persistent UP","score_persistent_down":"Persistent DOWN",
          "score_delayed_up":"Delayed UP","score_delayed_down":"Delayed DOWN",
          "score_acute_only_up":"Acute-only UP","score_acute_only_down":"Acute-only DOWN"}

scores = pd.read_csv(TRAJ, sep="\t")
meta   = pd.read_csv(META, sep="\t")[["SampleID","MM_SC","Age","Sex"]].drop_duplicates("SampleID")
cf     = pd.read_csv(RNA_CF, sep="\t")

df = scores.merge(meta, on="SampleID", how="left").merge(cf, on="SampleID", how="left")
rename_map = {c: c.replace(" ","_").replace("(","").replace(")","") for c in cf.columns if c != "SampleID"}
df = df.rename(columns=rename_map)
cf_cols_all = list(rename_map.values())

t1 = df[(df["Group"]=="AC") & (df["Timepoint"]=="T1")].copy()
t1["MM_SC_bin"] = (t1["MM_SC"]=="SC").astype(float)
cf_cols = [c for c in cf_cols_all if c in t1.columns and t1[c].mean() > 0.005]
print(f"T1: MM n={sum(t1['MM_SC']=='MM')}, SC n={sum(t1['MM_SC']=='SC')}")
print(f"cell-fraction covariates ({len(cf_cols)}): {cf_cols}\n")

def cohend(a,b):
    pooled = np.sqrt((np.var(a,ddof=1)+np.var(b,ddof=1))/2)
    return (np.mean(a)-np.mean(b))/(pooled+1e-9)

rows=[]
for scr in SCORE_COLS:
    sub = t1[["MM_SC","MM_SC_bin",scr]+cf_cols].dropna()
    mm = sub[sub["MM_SC"]=="MM"][scr].values; sc = sub[sub["MM_SC"]=="SC"][scr].values
    _, p_raw = stats.mannwhitneyu(mm, sc, alternative="two-sided"); d_raw = cohend(mm, sc)
    resid = smf.ols(scr+" ~ "+" + ".join(cf_cols), data=sub).fit().resid
    sub = sub.copy(); sub["resid"]=resid.values
    mmr = sub[sub["MM_SC"]=="MM"]["resid"].values; scr_ = sub[sub["MM_SC"]=="SC"]["resid"].values
    _, p_res = stats.mannwhitneyu(mmr, scr_, alternative="two-sided"); d_adj = cohend(mmr, scr_)
    anc = smf.ols(scr+" ~ MM_SC_bin + "+" + ".join(cf_cols), data=sub).fit()
    rows.append(dict(score=scr, label=LABELS[scr], n_MM=len(mm), n_SC=len(sc),
                     mean_MM_raw=mm.mean(), mean_SC_raw=sc.mean(), d_raw=d_raw, p_raw=p_raw,
                     d_adj=d_adj, p_adj_resid=p_res,
                     beta_ANCOVA=anc.params.get("MM_SC_bin",np.nan),
                     p_ANCOVA=anc.pvalues.get("MM_SC_bin",np.nan)))
    pct = 100*(abs(d_adj)-abs(d_raw))/abs(d_raw)
    print(f"{LABELS[scr]:18s} raw d={d_raw:+.3f}  adj d={d_adj:+.3f}  ({pct:+.0f}%)  ANCOVA p={rows[-1]['p_ANCOVA']:.3f}")

pd.DataFrame(rows).to_csv(OUT, sep="\t", index=False)
print(f"\n[saved] {OUT}")
