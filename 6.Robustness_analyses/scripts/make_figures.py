"""
Supplementary figures S12 (severity add-ons) and S13 (methyl batch sensitivity)
from the add-on analysis outputs. Style: no figure/panel descriptive titles,
large fonts, panel letters only. Saves PDF + 300-dpi PNG.
Run with .venv_deseq python.
"""
import numpy as np, pandas as pd, os, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 15, "axes.labelsize": 16, "axes.titlesize": 15,
    "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 13,
    "axes.linewidth": 1.1, "figure.dpi": 120, "savefig.bbox": "tight",
    "pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans",
})
NEW = "PROJECT_ROOT/research/LongCOVID/Results_updated_260714"
D   = f"{NEW}/addResult_by_review_0813/data"
FIG = f"{NEW}/addResult_by_review_0813/figures"; os.makedirs(FIG, exist_ok=True)
MDM = f"{NEW}/3.Discovery/2.Methyl_DM/data"
C_PERS, C_DEL = "#2c6fbb", "#e07b39"     # persistent vs delayed
C_GREY = "#8a8f98"
RNG = np.random.default_rng(0)

def panel_letter(ax, s):
    ax.text(-0.02, 1.06, s, transform=ax.transAxes, fontsize=20, fontweight="bold",
            va="top", ha="right")

def save(fig, name):
    fig.savefig(f"{FIG}/{name}.pdf")
    fig.savefig(f"{FIG}/{name}.png", dpi=300)
    print(f"[saved] {FIG}/{name}.pdf / .png")

# ============================================================ Fig S12: severity
ci = pd.read_csv(f"{D}/severity_effectsize_CI.tsv", sep="\t")
me = pd.read_csv(f"{D}/severity_acute_repeatedmeasures.tsv", sep="\t")
tost = pd.read_csv(f"{D}/severity_TOST_RC.tsv", sep="\t")
TPS = ["T0","T1","T2","T3","RC"]
ypos = {tp:i for i,tp in enumerate(TPS[::-1])}   # RC at bottom

fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

# (a) effect size + 95% CI forest
ax = axes[0]
for score, col, off in [("score_persistent_up", C_PERS, +0.12), ("score_delayed_up", C_DEL, -0.12)]:
    sub = ci[ci.score==score].set_index("timepoint")
    for tp in TPS:
        if tp in sub.index:
            r = sub.loc[tp]; y = ypos[tp]+off
            sig = r["fdr"] < 0.05
            ax.plot([r["d_CI95_low"], r["d_CI95_high"]], [y, y], color=col, lw=2.4, zorder=2)
            ax.scatter([r["cohen_d"]], [y], color=col, s=90 if sig else 60,
                       edgecolor="black" if sig else "none", linewidth=1.2, zorder=3)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_yticks(range(len(TPS))); ax.set_yticklabels(TPS[::-1])
ax.set_xlabel("Cohen's d (MM − SC), 95% CI"); ax.set_ylabel("Timepoint")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([0],[0],color=C_PERS,lw=3,label="persistent-UP"),
                   Line2D([0],[0],color=C_DEL,lw=3,label="delayed-UP"),
                   Line2D([0],[0],marker="o",color="w",markerfacecolor="grey",
                          markeredgecolor="black",markersize=9,label="FDR < 0.05")],
          loc="lower right", frameon=False)
panel_letter(ax, "a")

# (b) acute repeated-measures SC-MM delta
ax = axes[1]
ap = ["T0","T1","T2","T3"]; x = np.arange(len(ap))
for score, col, off in [("score_persistent_up", C_PERS, -0.13), ("score_delayed_up", C_DEL, +0.13)]:
    sub = me[(me.score==score)].set_index("timepoint")
    ys = [sub.loc[tp,"SC_minus_MM"] for tp in ap]
    lo = [sub.loc[tp,"ci_low"] for tp in ap]; hi = [sub.loc[tp,"ci_high"] for tp in ap]
    ax.errorbar(x+off, ys, yerr=[np.array(ys)-np.array(lo), np.array(hi)-np.array(ys)],
                fmt="o", color=col, ms=9, capsize=4, lw=2.2,
                label="persistent-UP" if "pers" in score else "delayed-UP")
    for i,tp in enumerate(ap):
        if sub.loc[tp,"pval"] < 0.05:
            ax.text(x[i]+off, hi[i]+0.06, "*", ha="center", va="bottom", fontsize=20, color=col)
ax.axhline(0, color="black", lw=1, ls="--")
ax.set_xticks(x); ax.set_xticklabels(ap)
ax.set_xlabel("Acute timepoint"); ax.set_ylabel("Severe − mild delta (cluster-robust)")
ax.legend(loc="upper left", frameon=False)
panel_letter(ax, "b")

# (c) TOST at RC
ax = axes[2]
scores = [("score_persistent_up","persistent-UP",1.0),("score_delayed_up","delayed-UP",0.0)]
for score, lab, y in scores:
    r05 = tost[(tost.score==score)&(tost.margin_d==0.5)].iloc[0]
    r08 = tost[(tost.score==score)&(tost.margin_d==0.8)].iloc[0]
    col = C_PERS if "pers" in score else C_DEL
    # margins as shaded spans (in raw units), 0.8 lighter, 0.5 darker
    ax.axvspan(-r08["margin_raw"], r08["margin_raw"], ymin=(y+0.12)/2, ymax=(y+0.88)/2,
               color=col, alpha=0.08)
    ax.plot([-r05["margin_raw"], r05["margin_raw"]], [y+0.30, y+0.30], color=col, lw=0, )
    for m,ls in [(r05["margin_raw"],":"),(r08["margin_raw"],"--")]:
        ax.plot([-m,-m],[y+0.16,y+0.44], color=col, ls=ls, lw=1.6)
        ax.plot([ m, m],[y+0.16,y+0.44], color=col, ls=ls, lw=1.6)
    # diff + 90% CI
    ax.plot([r05["ci90_low"], r05["ci90_high"]], [y+0.30, y+0.30], color=col, lw=3, zorder=3)
    ax.scatter([r05["diff"]], [y+0.30], color=col, s=110, edgecolor="black", zorder=4)
    ax.text(r08["margin_raw"]+0.02, y+0.30,
            f"eq. ±0.8 SD p={r08['p_tost']:.3f}\neq. ±0.5 SD p={r05['p_tost']:.3f}",
            va="center", fontsize=12)
ax.axvline(0, color="black", lw=1, ls="--")
ax.set_yticks([0.30, 1.30]); ax.set_yticklabels(["delayed-UP","persistent-UP"])
ax.set_ylim(-0.05, 1.75); ax.set_xlim(-1.0, 1.15)
ax.set_xlabel("MM − SC at RC (score units); 90% CI, TOST margins")
panel_letter(ax, "c")

fig.tight_layout(w_pad=2.2)
save(fig, "figS12_severity_addons")
plt.close(fig)

# ============================================================ Fig S13: methyl batch
g = pd.read_csv(f"{MDM}/gate2_rc_vs_hc_clean.tsv", sep="\t", usecols=["delta_beta","adj_pvalue"]).dropna()
summ = pd.read_csv(f"{D}/methyl_batch_tierA_summary.tsv", sep="\t").set_index("metric")["value"]
cen = pd.read_csv(f"{D}/methyl_batch_tierB_centering.tsv", sep="\t")
alld = g["delta_beta"].values
global_mean = float(summ["global_mean_delta_beta"])
inv_mean = float(summ["invariant_ctrl_mean_delta_beta"])
d303 = float(summ["delayed303_mean_delta_beta"]); ph155 = float(summ["promhyp155_mean_delta_beta"])

fig, axes = plt.subplots(1, 3, figsize=(18, 5.6))

# (a) genome-wide delta-beta distribution with markers
ax = axes[0]
samp = RNG.choice(alld, size=min(300000, len(alld)), replace=False)
ax.hist(samp, bins=200, range=(-0.15,0.15), color=C_GREY, alpha=0.8)
ax.axvline(global_mean, color="black", lw=2, label=f"genome-wide mean {global_mean:+.3f}")
ax.axvline(ph155, color=C_DEL, lw=2.4, label=f"delayed prom-hyper {ph155:+.3f}")
ax.set_xlabel("Δβ (RC − HC) per CpG"); ax.set_ylabel("CpG count")
ax.legend(loc="upper left", frameon=False, fontsize=12)
panel_letter(ax, "a")

# (b) size-matched null vs observed
ax = axes[1]
null155 = RNG.choice(alld, size=(10000,155), replace=True).mean(1)
ax.hist(null155, bins=60, color=C_GREY, alpha=0.85,
        label="random 155-CpG sets")
ax.axvline(ph155, color=C_DEL, lw=2.6, label=f"observed {ph155:+.3f}")
ax.set_xlabel("Mean Δβ of size-matched CpG set"); ax.set_ylabel("Random draws")
z = (ph155 - null155.mean())/null155.std()
ax.text(0.97, 0.75, f"z ≈ {z:+.0f}\np < 10⁻⁴", transform=ax.transAxes, ha="right", fontsize=13)
ax.legend(loc="upper left", frameon=False, fontsize=12)
panel_letter(ax, "b")

# (c) Tier B before/after centering: mean delta-beta + survival
ax = axes[2]
sets = ["promhyp_155","delayed_303"]
labels = ["promoter-hyper\n(155)","delayed\n(303)"]
x = np.arange(len(sets)); w = 0.34
pre = cen[cen.state=="uncentered"].set_index("cpg_set")
post = cen[cen.state=="centered"].set_index("cpg_set")
ax.bar(x-w/2, [pre.loc[s,"mean_delta_beta"] for s in sets], w, color=C_DEL, alpha=0.9, label="uncentered")
ax.bar(x+w/2, [post.loc[s,"mean_delta_beta"] for s in sets], w, color=C_PERS, alpha=0.9, label="global-shift removed")
for i,s in enumerate(sets):
    ax.text(x[i]-w/2, pre.loc[s,"mean_delta_beta"]+0.002, f"{int(pre.loc[s,'n_survive'])}/{int(pre.loc[s,'n'])}",
            ha="center", va="bottom", fontsize=12)
    ax.text(x[i]+w/2, post.loc[s,"mean_delta_beta"]+0.002, f"{int(post.loc[s,'n_survive'])}/{int(post.loc[s,'n'])}",
            ha="center", va="bottom", fontsize=12)
ax.axhline(0.05, color="black", ls=":", lw=1.2)
ax.text(1.45, 0.052, "|Δβ| = 0.05", fontsize=11, va="bottom", ha="right")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Mean Δβ (RC − HC)"); ax.set_ylim(0, 0.095)
ax.legend(loc="upper right", frameon=False, fontsize=12)
panel_letter(ax, "c")

fig.tight_layout(w_pad=2.2)
save(fig, "figS13_methyl_batch_sensitivity")
plt.close(fig)
print("done")
