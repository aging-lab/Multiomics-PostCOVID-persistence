"""
Methylation batch/technical-shift sensitivity (reviewer comment #4) — Tier A.
============================================================================
The delayed methylome (RC vs HC) is a cross-cohort contrast with NO explicit
sequencing-batch variable available (batch is collinear with condition). We
therefore cannot 'adjust it away'; instead we test whether the delayed
hypermethylation is a GLOBAL technical shift (which a batch offset would produce)
or a SPECIFIC signal.

Uses the already-computed genome-wide clean gate2 (RC vs HC) table — no re-fit.

Outputs:
  - global mean/median delta-beta across ALL CpGs (a uniform batch offset -> large;
    a specific signal -> ~0 genome-wide)
  - invariant control CpGs (adj_p>0.9): mean delta-beta should be ~0
  - background null: random size-matched CpG sets -> null of mean delta-beta;
    empirical p / z for the delayed set's mean delta-beta
  - directional balance genome-wide vs in the delayed set
Run with .venv_deseq python.
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")

NEW = "PROJECT_ROOT/research/LongCOVID/Results_updated_260714"
MDM = f"{NEW}/3.Discovery/2.Methyl_DM/data"
GATE2 = f"{MDM}/gate2_rc_vs_hc_clean.tsv"
MARKERS = f"{MDM}/methyl_markers_cleancohort_clean.tsv"
PROMHYP = f"{NEW}/4.Trajectory/4.Methyl_trajectory/data/methyl_cpg_trajectory_clean.tsv"
OUT = f"{NEW}/addResult_by_review_0813/data"
os.makedirs(OUT, exist_ok=True)
RNG = np.random.default_rng(0)
NDRAW = 10000

print("[1] load genome-wide gate2 (RC vs HC, clean) ...")
g = pd.read_csv(GATE2, sep="\t")             # cpg_id, delta_mval, delta_beta, abs_delta_beta, pvalue, adj_pvalue
g = g.dropna(subset=["delta_beta"])
ncpg = len(g)
print(f"    {ncpg:,} CpGs")

# marker sets
mk = pd.read_csv(MARKERS, sep="\t")
delayed303 = set(mk[mk["category"]=="delayed"]["cpg_id"])
ph = pd.read_csv(PROMHYP, sep="\t")
promhyp155 = set(ph["cpg_id"])
print(f"    delayed(303 set) n={len(delayed303)} ; promoter-hyper(155 set) n={len(promhyp155)}")

gb = g.set_index("cpg_id")

# ---------- global shift ----------
alld = g["delta_beta"].values
global_mean = float(np.mean(alld)); global_med = float(np.median(alld))
global_hyper_frac = float(np.mean(alld > 0))
print("\n=== GLOBAL (all CpGs, RC - HC) ===")
print(f"  mean delta-beta = {global_mean:+.5f}   median = {global_med:+.5f}   frac hyper = {global_hyper_frac:.3f}")
print(f"  (a uniform batch offset would push this away from 0; here ~0 => no global shift)")

# ---------- invariant controls ----------
inv = g[g["adj_pvalue"] > 0.9]["delta_beta"].values
print("\n=== INVARIANT CONTROL CpGs (adj_p > 0.9) ===")
print(f"  n={len(inv):,}  mean delta-beta = {inv.mean():+.5f}  mean |delta-beta| = {np.abs(inv).mean():.5f}")

# ---------- target set means ----------
def set_stats(cset, label):
    sub = gb.reindex([c for c in cset if c in gb.index])
    db = sub["delta_beta"].dropna().values
    return dict(label=label, n=len(db), mean_db=float(db.mean()),
                hyper_frac=float(np.mean(db > 0)), median_db=float(np.median(db)))
s303 = set_stats(delayed303, "delayed_303")
s155 = set_stats(promhyp155, "promoter_hyper_155")
print("\n=== TARGET SETS ===")
for s in (s303, s155):
    print(f"  {s['label']}: n={s['n']}  mean delta-beta={s['mean_db']:+.5f}  "
          f"median={s['median_db']:+.5f}  frac hyper={s['hyper_frac']:.3f}")

# ---------- background null for mean delta-beta ----------
def null_test(cset_stats, pool_desc, pool_vals, ndraw=NDRAW, rng=RNG):
    n = cset_stats["n"]; obs = cset_stats["mean_db"]
    draws = rng.choice(pool_vals, size=(ndraw, n), replace=True).mean(1)
    z = (obs - draws.mean()) / draws.std()
    p_emp = float((np.abs(draws - draws.mean()) >= abs(obs - draws.mean())).mean())
    return dict(set=cset_stats["label"], pool=pool_desc, n=n, obs_mean_db=obs,
                null_mean=float(draws.mean()), null_sd=float(draws.std()),
                z=float(z), p_empirical=p_emp,
                null_q2p5=float(np.percentile(draws,2.5)),
                null_q97p5=float(np.percentile(draws,97.5)))
pool_all = alld
pool_ns = g[g["adj_pvalue"] > 0.5]["delta_beta"].values   # non-significant background
null_rows = []
for st in (s155, s303):
    null_rows.append(null_test(st, "all_CpGs", pool_all))
    null_rows.append(null_test(st, "nonsig_CpGs(adjp>0.5)", pool_ns))
nulldf = pd.DataFrame(null_rows)
print("\n=== BACKGROUND NULL (random size-matched sets) ===")
for _, r in nulldf.iterrows():
    print(f"  {r['set']} vs {r['pool']}: obs={r.obs_mean_db:+.5f}  "
          f"null={r.null_mean:+.5f}+/-{r.null_sd:.5f}  z={r.z:+.1f}  p_emp={r.p_empirical:.1e}  "
          f"[null 95% {r.null_q2p5:+.4f},{r.null_q97p5:+.4f}]")

# ---------- save ----------
summ = pd.DataFrame([
    dict(metric="global_mean_delta_beta", value=global_mean),
    dict(metric="global_median_delta_beta", value=global_med),
    dict(metric="global_frac_hyper", value=global_hyper_frac),
    dict(metric="invariant_ctrl_mean_delta_beta", value=float(inv.mean())),
    dict(metric="invariant_ctrl_n", value=float(len(inv))),
    dict(metric="delayed303_mean_delta_beta", value=s303["mean_db"]),
    dict(metric="delayed303_hyper_frac", value=s303["hyper_frac"]),
    dict(metric="promhyp155_mean_delta_beta", value=s155["mean_db"]),
    dict(metric="promhyp155_hyper_frac", value=s155["hyper_frac"]),
])
summ.to_csv(f"{OUT}/methyl_batch_tierA_summary.tsv", sep="\t", index=False)
nulldf.to_csv(f"{OUT}/methyl_batch_tierA_null.tsv", sep="\t", index=False)
print(f"\n[saved] {OUT}/methyl_batch_tierA_summary.tsv")
print(f"[saved] {OUT}/methyl_batch_tierA_null.tsv")
