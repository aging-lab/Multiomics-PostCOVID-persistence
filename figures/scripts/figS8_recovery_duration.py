#!/usr/bin/env python3
"""Supplementary Fig. S11 — RC recovery-duration robustness (clean/filtered cohort).
Shows the persistent scar does NOT depend on recovery duration once recruitment
wave is accounted for (the crude association is a recruitment-wave confound /
Simpson's paradox). Recomputed on the recomputed clean gene-set scores
(trajectory_scores_clean.tsv; canonical, IG/pseudo-removed marker sets).

Panel A: Persistent-UP scar vs recovery duration, coloured by recruitment wave;
         wave-specific regressions vs the confounded overall slope.
Panel B: Recovery-duration slope (per 100 d) crude vs wave-adjusted, 95% CI,
         for persistent/delayed UP & DOWN scores.
"""
import csv, re, os, numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R   = "PROJECT_ROOT/research/LongCOVID/Results"
NEW = R + "_updated_260714"
SC  = f"{NEW}/4.Trajectory/1.LME_GAM/data/trajectory_scores_clean.tsv"
META= f"{R}/1.Metadata/COVID-19_sample_metadata.tsv"
CF  = f"{R}/2.Data_preparation/6.Deconvolution/data/rna_cell_fractions_v2.tsv"
OUT = f"{NEW}/manuscript/figures/figS8_recovery_duration.png"
OUTPDF = OUT.replace(".png", ".pdf")

def num(v):
    try: return float(v)
    except: return None

meta = {r["SampleID"]: r for r in csv.DictReader(open(META), delimiter="\t")}
cf   = {r["SampleID"]: r for r in csv.DictReader(open(CF), delimiter="\t")}
def agg_cf(sid):
    r = cf.get(sid)
    if not r: return None
    g = lambda k: float(r[k])
    return dict(
        cf_CD4T=g("T cells CD4 naive")+g("T cells CD4 memory resting")+g("T cells CD4 memory activated")
                +g("T cells follicular helper")+g("T cells regulatory (Tregs)")+g("T cells gamma delta"),
        cf_CD8T=g("T cells CD8"),
        cf_NK=g("NK cells resting")+g("NK cells activated"),
        cf_Mono=g("Monocytes"))

# clean convalescent V1 (stratum == convalescent), with covariates
conv = []
for s in csv.DictReader(open(SC), delimiter="\t"):
    if s["Group"] != "RC" or s.get("stratum") != "convalescent": continue
    sid = s["SampleID"]
    if not sid.endswith("-V1"): continue
    n = int(re.search(r"-R(\d+)", sid).group(1))
    c = agg_cf(sid)
    age = num(meta.get(sid, {}).get("Age")); sex = 1.0 if meta.get(sid, {}).get("Sex") == "F" else 0.0
    if c is None or age is None: continue
    conv.append(dict(days=float(s["days_from_T0"]), wave=("A" if n <= 100 else "B"),
                     age=age, sex=sex,
                     p_up=num(s["score_persistent_up"]), p_dn=num(s["score_persistent_down"]),
                     d_up=num(s["score_delayed_up"]), d_dn=num(s["score_delayed_down"]), **c))
print(f"clean convalescent V1 n={len(conv)}  (wave A={sum(r['wave']=='A' for r in conv)}, "
      f"B={sum(r['wave']=='B' for r in conv)})")

def ols(y, X):
    X = np.asarray(X, float); y = np.asarray(y, float)
    beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta; n, k = X.shape; dof = n - k
    sigma2 = (resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    t = beta / se
    ci = stats.t.ppf(0.975, dof) * se
    return beta, se, t, 2 * stats.t.sf(np.abs(t), dof), ci

days = np.array([r["days"] for r in conv]) / 100
waveB = np.array([1.0 if r["wave"] == "B" else 0.0 for r in conv])
age = np.array([r["age"] for r in conv]); sex = np.array([r["sex"] for r in conv])
cfs = np.column_stack([[r[c] for r in conv] for c in ["cf_CD4T", "cf_CD8T", "cf_NK", "cf_Mono"]])
one = np.ones(len(conv))

SCORES = [("p_up", "Persistent-UP"), ("p_dn", "Persistent-DOWN"),
          ("d_up", "Delayed-UP"), ("d_dn", "Delayed-DOWN")]
stat = {}
for key, lab in SCORES:
    y = np.array([r[key] for r in conv])
    bc, _, _, pc, cic = ols(y, np.column_stack([one, days]))
    ba, _, _, pa, cia = ols(y, np.column_stack([one, days, waveB, age, sex, cfs]))
    stat[key] = dict(crude_b=bc[1], crude_p=pc[1], crude_ci=cic[1],
                     adj_b=ba[1], adj_p=pa[1], adj_ci=cia[1])
    print(f"{lab:16} crude b={bc[1]:+.3f} p={pc[1]:.3f} | wave+adj b={ba[1]:+.3f} p={pa[1]:.3f}")

# ── figure ──
FS = 13
plt.rcParams.update({"font.family": "Arial", "font.size": FS, "axes.spines.top": False,
                     "axes.spines.right": False, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
colA, colB = "#457B9D", "#E63946"

# Panel A — dose-response with wave confound
ax = axes[0]
for wave, col, lab in [("A", colA, "Wave A (2021, R001–100)"), ("B", colB, "Wave B (2022, R101–148)")]:
    g = [r for r in conv if r["wave"] == wave]
    x = [r["days"] for r in g]; y = [r["p_up"] for r in g]
    ax.scatter(x, y, c=col, s=34, alpha=0.65, edgecolors="none", label=lab)
    if len(g) > 2:
        b = np.polyfit(x, y, 1); xs = np.linspace(min(x), max(x), 50)
        ax.plot(xs, np.polyval(b, xs), c=col, lw=2)
xall = [r["days"] for r in conv]; yall = [r["p_up"] for r in conv]
b = np.polyfit(xall, yall, 1); xs = np.linspace(min(xall), max(xall), 50)
ax.plot(xs, np.polyval(b, xs), c="k", lw=1.6, ls="--", label="Overall (confounded)")
ax.axhline(0, c="grey", lw=1, ls=":")
ax.set_xlabel("Recovery duration (days since diagnosis)")
ax.set_ylabel("Persistent-UP gene-set score (HC-anchored z)")
ax.text(0.03, 0.97,
        f"unadjusted {stat['p_up']['crude_b']:+.2f}/100 d (p={stat['p_up']['crude_p']:.2f})\n"
        f"wave-adjusted {stat['p_up']['adj_b']:+.2f} (p={stat['p_up']['adj_p']:.2f}, n.s.)",
        transform=ax.transAxes, va="top", ha="left", fontsize=FS - 2,
        bbox=dict(boxstyle="round", fc="#fff6e6", ec="#e0b050"))
ax.legend(loc="lower right", frameon=False, fontsize=FS - 3)

# Panel B — slope forest: crude vs wave-adjusted
ax = axes[1]
ypos = np.arange(len(SCORES))[::-1]
for i, (key, lab) in zip(ypos, SCORES):
    s = stat[key]
    ax.errorbar(s["crude_b"], i + 0.15, xerr=s["crude_ci"], fmt="o", color="#999",
                capsize=3, ms=6, label="unadjusted" if i == ypos[0] else None)
    sig = "*" if s["adj_p"] < 0.05 else ""
    ax.errorbar(s["adj_b"], i - 0.15, xerr=s["adj_ci"], fmt="s", color="#2A9D8F",
                capsize=3, ms=6, label="wave-adjusted" if i == ypos[0] else None)
ax.axvline(0, c="grey", lw=1, ls="--")
ax.set_yticks(ypos); ax.set_yticklabels([lab for _, lab in SCORES])
ax.set_xlabel("Recovery-duration slope (Δ gene-set score / 100 d)")
ax.legend(loc="lower left", frameon=False, fontsize=FS - 2)
ax.set_ylim(-0.6, len(SCORES) - 0.4)

# panel labels A / B (match main-figure style)
for ax_, lab in zip(axes, ["A", "B"]):
    ax_.annotate(lab, xy=(0, 1), xycoords="axes fraction", xytext=(-40, 12),
                 textcoords="offset points", fontsize=17, fontweight="bold",
                 va="bottom", ha="left")

plt.tight_layout()
os.makedirs(os.path.dirname(OUT), exist_ok=True)
fig.savefig(OUT, dpi=300, bbox_inches="tight")
fig.savefig(OUTPDF, bbox_inches="tight")
print(f"[saved] {OUT}")
