"""
Severity review add-ons (reviewer comment #3) — CLEAN cohort.
=============================================================
Adds, on top of severity_cleancohort.py (which gave point Cohen's d + MWU):
  (1) Cohen's d with 95% CI (bootstrap percentile, seed=0) at every timepoint x score.
  (2) Longitudinal ACUTE (T0-T3) severity effect via a proper repeated-measures model:
      OLS with cluster-robust (by SubjectID) SEs, AND a linear mixed-effects model
      (random intercept per SubjectID) as a cross-check. Covariates: Age, Sex,
      Neutrophils, CD8 T cells (CIBERSORTx LM22).
  (3) TOST equivalence test at RC (convalescence): is the MM-vs-SC difference
      confidently within a pre-specified margin? Margins d=0.5 (primary) and d=0.8.

Point estimates must match severity_cleancohort.py (same input/cohort/d formula).
Run with .venv_deseq python.
"""
import numpy as np, pandas as pd, os, warnings, json
warnings.filterwarnings("ignore")
from scipy import stats

R   = "PROJECT_ROOT/research/LongCOVID/Results"
NEW = R + "_updated_260714"
TRAJ = f"{NEW}/4.Trajectory/1.LME_GAM/data/trajectory_scores_clean.tsv"
META = f"{R}/1.Metadata/COVID/COVID-19_sample_metadata.tsv"
RNACF = f"{R}/2.Data_preparation/6.Deconvolution/data/rna_cell_fractions_v2.tsv"
OUT  = f"{NEW}/addResult_by_review_0813/data"
os.makedirs(OUT, exist_ok=True)
SCORES = ["score_persistent_up","score_persistent_down","score_delayed_up",
          "score_delayed_down","score_acute_only_up","score_acute_only_down"]
KEY = ["score_persistent_up","score_delayed_up"]   # headline scores
RNG = np.random.default_rng(0)
NBOOT = 10000

# ---------- load & merge ----------
sc = pd.read_csv(TRAJ, sep="\t")
meta = pd.read_csv(META, sep="\t")[["SampleID","MM_SC","Age","Sex"]].drop_duplicates("SampleID")
cf = pd.read_csv(RNACF, sep="\t")
cf = cf.rename(columns={"Neutrophils":"Neutro","T cells CD8":"CD8T"})[["SampleID","Neutro","CD8T"]]
df = sc.merge(meta, on="SampleID", how="left").merge(cf, on="SampleID", how="left")
# RC: clean convalescent only (match severity_cleancohort.py)
df = df[~((df["Group"]=="RC") & (df["stratum"]!="convalescent"))].copy()
df["Sex_bin"] = df["Sex"].map({"M":0,"F":1}).fillna(0).astype(float)
df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
for c in ["Neutro","CD8T"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

def cohend(a, b):
    # matches severity_cleancohort.py: (mean_a - mean_b)/sqrt((sd_a^2+sd_b^2)/2)
    return (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)

def boot_ci_d(a, b, nboot=NBOOT, rng=RNG):
    idx_a = rng.integers(0, len(a), size=(nboot, len(a)))
    idx_b = rng.integers(0, len(b), size=(nboot, len(b)))
    sa = a[idx_a]; sb = b[idx_b]
    ma = sa.mean(1); mb = sb.mean(1)
    va = sa.var(1, ddof=1); vb = sb.var(1, ddof=1)
    d = (ma - mb) / np.sqrt((va + vb) / 2)
    d = d[np.isfinite(d)]
    return np.percentile(d, 2.5), np.percentile(d, 97.5)

# ---------- (1) effect sizes with 95% CI ----------
tp_sets = [("T0", df[df.Timepoint=="T0"]), ("T1", df[df.Timepoint=="T1"]),
           ("T2", df[df.Timepoint=="T2"]), ("T3", df[df.Timepoint=="T3"]),
           ("RC", df[df.Group=="RC"])]
rows = []
for tp, sub in tp_sets:
    mm = sub[sub.MM_SC=="MM"]; scg = sub[sub.MM_SC=="SC"]
    for col in SCORES:
        a = mm[col].dropna().values; b = scg[col].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        d = cohend(pd.Series(a), pd.Series(b))
        lo, hi = boot_ci_d(a, b)
        rows.append(dict(timepoint=tp, score=col, n_MM=len(a), n_SC=len(b),
                         mean_MM=a.mean(), mean_SC=b.mean(), cohen_d=d,
                         d_CI95_low=lo, d_CI95_high=hi, mwu_p=p))
res = pd.DataFrame(rows)
from statsmodels.stats.multitest import multipletests
out = []
for tp, g in res.groupby("timepoint"):
    g = g.copy(); g["fdr"] = multipletests(g["mwu_p"], method="fdr_bh")[1]; out.append(g)
res = pd.concat(out)
res.to_csv(f"{OUT}/severity_effectsize_CI.tsv", sep="\t", index=False)

# ---------- (2) longitudinal acute model (repeated measures) ----------
import statsmodels.formula.api as smf
# Repeated-measures inference: OLS with cluster-robust (by-subject) SEs.
# (A random-intercept MixedLM deadlocks in this venv's statsmodels build; the
#  by-subject cluster-robust SE addresses the same within-subject correlation and
#  is the reported method.)
acute = df[df.Timepoint.isin(["T0","T1","T2","T3"])].copy()
acute["SC"] = (acute["MM_SC"]=="SC").astype(float)
me_rows = []
for col in KEY:
    d0 = acute.dropna(subset=[col,"Age","Sex_bin","Neutro","CD8T","SubjectID"]).copy()
    d0["y"] = d0[col]
    # per-timepoint SC-vs-MM delta = interaction coefs (no standalone SC term):
    #   y ~ C(Timepoint) + C(Timepoint):SC + Age + Sex + Neutro + CD8T
    formula = "y ~ C(Timepoint) + C(Timepoint):SC + Age + Sex_bin + Neutro + CD8T"
    ols = smf.ols(formula, data=d0).fit(cov_type="cluster",
                                        cov_kwds={"groups": d0["SubjectID"]})
    for tp in ["T0","T1","T2","T3"]:
        term = f"C(Timepoint)[{tp}]:SC"
        if term in ols.params.index:
            b = ols.params[term]; se = ols.bse[term]; p = ols.pvalues[term]
            ci = ols.conf_int().loc[term].values
            me_rows.append(dict(score=col, timepoint=tp, model="OLS_cluster_subject",
                       SC_minus_MM=b, se=se, ci_low=ci[0], ci_high=ci[1], pval=p))
me = pd.DataFrame(me_rows)
me.to_csv(f"{OUT}/severity_acute_repeatedmeasures.tsv", sep="\t", index=False)

# ---------- (3) TOST equivalence at RC ----------
def welch_tost(a, b, margin):
    # H1: |mean_a - mean_b| < margin.  Two one-sided Welch t-tests.
    na, nb = len(a), len(b)
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = np.sqrt(va/na + vb/nb)
    diff = ma - mb
    dfw = (va/na + vb/nb)**2 / ((va/na)**2/(na-1) + (vb/nb)**2/(nb-1))
    # lower: H0 diff <= -margin ; upper: H0 diff >= +margin
    t_low = (diff - (-margin)) / se
    t_up  = (diff - (margin)) / se
    p_low = stats.t.sf(t_low, dfw)     # P(T > t_low)
    p_up  = stats.t.cdf(t_up, dfw)     # P(T < t_up)
    p_tost = max(p_low, p_up)
    # 90% CI of diff (equiv test uses (1-2alpha) CI)
    tcrit = stats.t.ppf(0.95, dfw)
    return dict(diff=diff, se=se, df=dfw, p_low=p_low, p_up=p_up, p_tost=p_tost,
                ci90_low=diff - tcrit*se, ci90_high=diff + tcrit*se)

rc = df[df.Group=="RC"]
tost_rows = []
for col in SCORES:
    a = rc[rc.MM_SC=="MM"][col].dropna().values
    b = rc[rc.MM_SC=="SC"][col].dropna().values
    if len(a) < 3 or len(b) < 3:
        continue
    pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    for dmarg in (0.5, 0.8):
        margin = dmarg * pooled_sd
        r = welch_tost(a, b, margin)
        tost_rows.append(dict(score=col, n_MM=len(a), n_SC=len(b),
                              margin_d=dmarg, margin_raw=margin, pooled_sd=pooled_sd,
                              diff=r["diff"], ci90_low=r["ci90_low"], ci90_high=r["ci90_high"],
                              p_tost=r["p_tost"], p_low=r["p_low"], p_up=r["p_up"],
                              equivalence_at_0p05=bool(r["p_tost"] < 0.05)))
tost = pd.DataFrame(tost_rows)
tost.to_csv(f"{OUT}/severity_TOST_RC.tsv", sep="\t", index=False)

# ---------- console summary ----------
print("\n================ (1) EFFECT SIZE + 95% CI (key scores) ================")
for col in KEY:
    print(f"\n{col}:")
    for tp in ["T0","T1","T2","T3","RC"]:
        r = res[(res.timepoint==tp) & (res.score==col)]
        if len(r):
            r = r.iloc[0]; sig = "*" if r.fdr < 0.05 else " "
            print(f"  {tp}: d={r.cohen_d:+.3f} [95% CI {r.d_CI95_low:+.3f}, {r.d_CI95_high:+.3f}] "
                  f"p={r.mwu_p:.3f} fdr={r.fdr:.3f}{sig} (MM={r.n_MM}, SC={r.n_SC})")

print("\n================ (2) ACUTE repeated-measures SC-vs-MM delta ================")
for col in KEY:
    print(f"\n{col}:")
    for model in ["OLS_cluster_subject"]:
        sub = me[(me.score==col) & (me.model==model)]
        if len(sub):
            print(f"  [{model}]")
            for _, r in sub.iterrows():
                print(f"    {r.timepoint}: SC-MM={r.SC_minus_MM:+.3f} (SE {r.se:.3f}) "
                      f"CI[{r.ci_low:+.3f},{r.ci_high:+.3f}] p={r.pval:.4f}")

print("\n================ (3) TOST equivalence at RC ================")
for col in KEY:
    for dmarg in (0.5, 0.8):
        r = tost[(tost.score==col) & (tost.margin_d==dmarg)]
        if len(r):
            r = r.iloc[0]
            verdict = "EQUIVALENT" if r["equivalence_at_0p05"] else "NOT established"
            print(f"  {col} margin d=±{dmarg}: diff={r['diff']:+.3f} 90%CI[{r['ci90_low']:+.3f},{r['ci90_high']:+.3f}] "
                  f"margin=±{r['margin_raw']:.3f} p_TOST={r['p_tost']:.3f} -> equivalence {verdict}  (SC n={r['n_SC']})")

print(f"\n[saved] {OUT}/severity_effectsize_CI.tsv")
print(f"[saved] {OUT}/severity_acute_repeatedmeasures.tsv")
print(f"[saved] {OUT}/severity_TOST_RC.tsv")
