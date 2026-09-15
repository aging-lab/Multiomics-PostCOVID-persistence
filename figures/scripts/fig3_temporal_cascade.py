"""
Fig 3 — Temporal dissociation (HEADLINE)
Panels:
  A: RNA z-score trajectory of 263 persistent genes (UP n=182 / DOWN n=81)
     across HC/T0/T1/T2/T3/RC — mean ± SEM, HC = 0 reference
  B: Methylation Δβ trajectory of delayed promoter-hyper CpGs across
     HC/T0/T2/T3/RC (no methylation data at T1).
     Rationale for showing hyper only: within the delayed set (n=396),
     hypermethylation strongly dominates (hyper 354 vs hypo 42, ~8.4:1);
     delayed promoter-hypo CpGs number only 9 — too few for a stable
     trajectory. So delayed promoter-hyper (n=214) is the interpretable,
     representative class (promoter hyper = gene silencing).
  C: Peak-normalized overlay (money panel, full width) — RNA (red) and
     Methylation (orange) each scaled to their own peak (= 1) on a single
     axis, so the eye compares peak *timing* not magnitude: RNA activates
     acutely (T0), methylome peaks at convalescence (RC) — a ~60-day lag.
     Headline temporal dissociation; A/B are its two single-omics components.
     (A single shared, peak-normalized axis is used deliberately instead of a
     dual axis, whose crossover point would be an arbitrary scaling artifact.)

Note: the genome-wide dissociation (RNA DEG 1369->345 vs methyl DMP
152->399, acute->convalescent) is reported in the Results text only, not
as a panel here.

Terminology note: "Acute / Convalescent contrast" defined the persistent set.
No figure-level suptitle (title/caption added in manuscript).
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import blended_transform_factory
import warnings
warnings.filterwarnings("ignore")

# ── paths ──────────────────────────────────────────────────────────────────
BASE = "PROJECT_ROOT/research/LongCOVID/Results"
NEW  = BASE + "_updated_260714"
OUT  = f"{NEW}/manuscript/figures"

ARC_FILE  = f"{NEW}/4.Trajectory/3.Archetypes/data/gene_archetypes_filtered.tsv"  # canonical, IG/pseudo removed (2026-07-30)
METH_TRAJ = f"{NEW}/4.Trajectory/4.Methyl_trajectory/data/methyl_cpg_trajectory_clean.tsv"  # already promoter-hyper subset

# ── style ──────────────────────────────────────────────────────────────────
RNA_UP   = "#E63946"    # red — upregulated persistent
RNA_DN   = "#457B9D"    # blue — downregulated persistent
METH_COL = "#E9A23B"    # amber — methylation

# ── font sizes (enlarged for readability) ──────────────────────────────────
TITLE_FS  = 12      # panel titles
LABEL_FS  = 11      # axis labels
TICK_FS   = 9.5     # tick labels
LEGEND_FS = 9.5     # legends
ANNOT_FS  = 9       # in-panel annotations

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 11,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.9,
    "ytick.major.width": 0.9,
    "figure.dpi": 150,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Plot x-positions. Acute window (HC..T3) is drawn on the real-day scale;
# RC (~D60) is pulled in from day 65 -> 33 so the long convalescent gap does
# not dominate the panel. An axis break (//) between T3 and RC marks the jump.
XPOS = {"HC": -10, "T0": 0, "T1": 7, "T2": 14, "T3": 21, "RC": 33}
XBREAK = 27.0            # break marker sits between T3 (21) and RC (33)
RNA_DAYS  = ["HC", "T0", "T1", "T2", "T3", "RC"]
METH_DAYS = ["HC", "T0", "T2", "T3", "RC"]
TP_LABELS = {"HC": "HC", "T0": "T0\n(D0)", "T1": "T1\n(D7)",
             "T2": "T2\n(D14)", "T3": "T3\n(D21)", "RC": "RC"}


def draw_xbreak(ax, xbreak=XBREAK, rise_in=0.11, angle_deg=60, gap_in=0.05):
    """Two parallel slashes straddling the bottom spine to mark a time jump.

    Sized in physical inches so the slash angle is identical across panels
    regardless of their aspect ratio (a wide/short panel would otherwise make
    the marks look flat).
    """
    fig = ax.figure
    pos = ax.get_position()
    fw, fh = fig.get_size_inches()
    axw_in, axh_in = pos.width * fw, pos.height * fh
    x0, x1 = ax.get_xlim()

    run_in   = rise_in / np.tan(np.radians(angle_deg))   # horizontal run of a slash
    dy       = rise_in / axh_in                          # axes-fraction (full rise)
    run_data = run_in / axw_in * (x1 - x0)               # data-units run
    gap_data = gap_in / axw_in * (x1 - x0)               # data-units centre offset

    tr = blended_transform_factory(ax.transData, ax.transAxes)
    for c in (xbreak - gap_data, xbreak + gap_data):
        ax.plot([c - run_data / 2, c + run_data / 2], [-dy / 2, dy / 2],
                transform=tr, color="k", lw=1.1, clip_on=False, zorder=20,
                solid_capstyle="round")


# ── helpers ────────────────────────────────────────────────────────────────
def parse_gene_name(gid):
    parts = str(gid).split("_", 1)
    return parts[1] if len(parts) > 1 else gid


def compute_rna_trajectory(arc_df):
    """Mean ± SEM of HC-referenced z-score per timepoint, for UP and DOWN."""
    arc_df = arc_df.copy()
    mu_cols = {"HC": "mu_HC", "T0": "mu_T0", "T1": "mu_T1",
               "T2": "mu_T2", "T3": "mu_T3", "RC": "mu_RC"}
    for tp, col in mu_cols.items():
        arc_df[f"z_{tp}"] = arc_df[col] - arc_df["mu_HC"]

    result = {}
    for group, direction in [("up", "up"), ("dn", "down")]:
        df_g = arc_df[arc_df["direction"] == direction]
        means = [df_g[f"z_{tp}"].dropna().mean() for tp in RNA_DAYS]
        sems  = [df_g[f"z_{tp}"].dropna().sem()  for tp in RNA_DAYS]
        result[group] = {"mean": means, "sem": sems, "n": len(df_g)}
    return result


def compute_meth_trajectory(meth_df):
    """Mean ± SEM of Δβ (relative to HC) per timepoint.
    Clean-cohort file is ALREADY the delayed promoter-hyper subset (~155 CpGs),
    so no re-filtering here."""
    delayed_hyper = meth_df.copy()

    means, sems = [], []
    for tp in METH_DAYS:
        if tp == "HC":
            vals = pd.Series(np.zeros(len(delayed_hyper)))
        else:
            vals = delayed_hyper[tp] - delayed_hyper["HC"]
        means.append(vals.dropna().mean())
        sems.append(vals.dropna().sem())
    return {"mean": means, "sem": sems, "n": len(delayed_hyper)}


# ── panels ──────────────────────────────────────────────────────────────────
def panel_A_rna(ax, rna_traj):
    x_days = [XPOS[tp] for tp in RNA_DAYS]

    for group, col, label in [
        ("up", RNA_UP, f"Persistent UP (n={rna_traj['up']['n']})"),
        ("dn", RNA_DN, f"Persistent DOWN (n={rna_traj['dn']['n']})"),
    ]:
        m = np.array(rna_traj[group]["mean"])
        s = np.array(rna_traj[group]["sem"])
        ax.fill_between(x_days, m - s, m + s, alpha=0.18, color=col, lw=0)
        ax.plot(x_days, m, "-o", color=col, ms=5, lw=2, label=label, zorder=3)

    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_ylim(-1.65, 1.15)
    ax.set_xticks(x_days)
    ax.set_xticklabels([TP_LABELS[tp] for tp in RNA_DAYS], fontsize=TICK_FS)
    ax.set_ylabel("Mean Δ z-score (vs HC)", fontsize=LABEL_FS)
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")
    # place on the RIGHT, in the empty mid-band between the UP (top) and DOWN
    # (bottom) lines where both are flat (T2-RC) — clear of the steep left drop
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="center right",
              bbox_to_anchor=(0.99, 0.45))
    ax.tick_params(labelsize=TICK_FS)
    draw_xbreak(ax)

    # acute activation annotation (UP peak at T0), placed in clear headroom
    m_up_t0 = rna_traj["up"]["mean"][1]
    ax.annotate("Acute activation\n(peaks at T0)",
                xy=(XPOS["T0"], m_up_t0),
                xytext=(XPOS["T1"] + 2, 0.85),
                fontsize=ANNOT_FS, color=RNA_UP, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=RNA_UP, lw=0.8))


def panel_B_meth(ax, meth_traj):
    x_days = [XPOS[tp] for tp in METH_DAYS]
    m = np.array(meth_traj["mean"])
    s = np.array(meth_traj["sem"])

    # ±SEM band is drawn but is smaller than the markers (very tight CIs,
    # mean/SEM up to ~53 at RC) — noted in the legend below.
    ax.fill_between(x_days, m - s, m + s, alpha=0.30, color=METH_COL, lw=0)
    ax.plot(x_days, m, "-s", color=METH_COL, ms=5, lw=2,
            label=f"Delayed promoter-hyper CpGs (n={meth_traj['n']}, ±SEM < marker)",
            zorder=3)

    ax.axhline(0, color="#888", lw=0.8, ls="--")
    ax.set_ylim(-0.045, 0.115)

    # T1 gap marker (no methylation data) — fixed, unobtrusive
    ax.axvline(XPOS["T1"], color="#CCC", lw=0.8, ls=":", alpha=0.8)
    ax.text(XPOS["T1"], -0.038, "no data\nat T1", fontsize=ANNOT_FS-1.5, ha="center",
            va="bottom", color="#AAA", style="italic")

    # transient hypo dip at T3 (real signal: −0.015 ± 0.002), currently the only
    # non-monotonic feature — labelled so it is not mistaken for noise
    idx_t3 = METH_DAYS.index("T3")
    ax.annotate("transient dip\nat T3",
                xy=(XPOS["T3"], m[idx_t3]),
                xytext=(XPOS["T2"] - 0.5, -0.033),
                fontsize=ANNOT_FS - 1, color="#7a7a7a", ha="center", va="center",
                arrowprops=dict(arrowstyle="-|>", color="#9a9a9a", lw=0.7))

    # RC peak annotation, kept inside the axes
    idx_rc = METH_DAYS.index("RC")
    ax.annotate(f"peak at RC\nΔβ = +{m[idx_rc]:.3f}",
                xy=(XPOS["RC"], m[idx_rc]),
                xytext=(XPOS["T2"] - 1, 0.095),
                fontsize=ANNOT_FS, color="#B9791F", ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=METH_COL, lw=0.8))

    ax.set_xticks(x_days)
    ax.set_xticklabels([TP_LABELS[tp] for tp in METH_DAYS], fontsize=TICK_FS)
    ax.set_ylabel("Mean Δβ (vs HC)", fontsize=LABEL_FS)
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")
    ax.legend(fontsize=LEGEND_FS, frameon=False, loc="upper left")
    ax.tick_params(labelsize=TICK_FS)
    draw_xbreak(ax)


def panel_C_overlay(ax, rna_traj, meth_traj):
    """Peak-normalized overlay (money panel).

    Both trajectories are scaled to their own peak magnitude (= 1) so the eye
    compares *timing of the peak*, not absolute size — the honest claim here is
    a temporal offset, and a dual axis would let the arbitrary crossover point
    be over-read. RNA peaks acutely (T0), methylation peaks at convalescence
    (RC): a ~60-day lag.
    """
    rna_x = [XPOS[tp] for tp in RNA_DAYS]
    rna_m = np.array(rna_traj["up"]["mean"])
    meth_x = [XPOS[tp] for tp in METH_DAYS]
    meth_m = np.array(meth_traj["mean"])

    # normalize each curve to its own peak (max absolute value) -> peak = 1
    rna_n  = rna_m  / np.nanmax(np.abs(rna_m))
    meth_n = meth_m / np.nanmax(np.abs(meth_m))

    # shaded "events" first (behind lines)
    ax.axvspan(XPOS["T0"] - 2.5, XPOS["T0"] + 2.5, alpha=0.10, color=RNA_UP, zorder=0)
    ax.axvspan(XPOS["RC"] - 3.5, XPOS["RC"] + 3.5, alpha=0.10, color=METH_COL, zorder=0)

    l1, = ax.plot(rna_x, rna_n, "-o", color=RNA_UP, ms=5, lw=2.4, zorder=4)
    l2, = ax.plot(meth_x, meth_n, "-s", color=METH_COL, ms=5, lw=2.4, zorder=3)

    ax.axhline(0, color="#CCC", lw=0.6, ls="--", zorder=1)
    ax.set_ylim(-0.34, 1.20)       # headroom for lag arrow; room below for captions

    ax.set_ylabel("Relative response\n(peak = 1)", fontsize=LABEL_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_yticks([0, 0.5, 1.0])

    ax.set_xticks([XPOS[tp] for tp in RNA_DAYS])
    ax.set_xticklabels([TP_LABELS[tp] for tp in RNA_DAYS], fontsize=TICK_FS)

    # ~60-day lag arrow between the two peaks (RNA @ T0 -> methylation @ RC)
    ax.annotate("", xy=(XPOS["RC"], 1.10), xytext=(XPOS["T0"], 1.10),
                arrowprops=dict(arrowstyle="<|-|>", color="#555", lw=1.1))
    ax.text((XPOS["T0"] + XPOS["RC"]) / 2, 1.14,
            "~60-day lag  (RNA peak → methylation peak)",
            fontsize=ANNOT_FS, color="#555", ha="center", va="bottom")

    # event captions at the BOTTOM of each shaded band (empty space; no overlap)
    ax.text(XPOS["T0"], -0.31, "RNA\nactivates first", fontsize=ANNOT_FS, color=RNA_UP,
            ha="center", va="bottom", fontweight="bold")
    ax.text(XPOS["RC"], -0.31, "Methylation\npeaks later", fontsize=ANNOT_FS,
            color="#B9791F", ha="center", va="bottom", fontweight="bold")

    # no legend: the two colored bottom captions (red "RNA…" / orange
    # "Methylation…") already identify each series by colour, and a legend box
    # would collide with the lag arrow. Add a small colour cue at each line's
    # right end instead.
    ax.text(XPOS["RC"] + 0.6, rna_n[-1], "RNA", fontsize=ANNOT_FS,
            color=RNA_UP, ha="left", va="center", fontweight="bold")
    ax.text(XPOS["RC"] + 0.6, meth_n[-1], "Methyl.", fontsize=ANNOT_FS,
            color="#B9791F", ha="left", va="center", fontweight="bold")
    ax.set_title("",
                 fontsize=TITLE_FS, fontweight="bold", pad=4, loc="left")
    ax.tick_params(labelsize=TICK_FS)
    draw_xbreak(ax)


# ── main ───────────────────────────────────────────────────────────────────
def main():
    arc_df  = pd.read_csv(ARC_FILE, sep="\t")
    meth_df = pd.read_csv(METH_TRAJ, sep="\t")
    arc_df  = arc_df[arc_df["direction"].isin(["up", "down"])].copy()

    rna_traj  = compute_rna_trajectory(arc_df)
    meth_traj = compute_meth_trajectory(meth_df)

    fig = plt.figure(figsize=(13, 8))
    # top row: A (RNA) | B (methylation); bottom row: C (dual-axis money panel)
    # full width but shorter — a wide time-series banner (fills the row cleanly)
    gs = GridSpec(2, 2, figure=fig,
                  left=0.09, right=0.93, top=0.95, bottom=0.10,
                  height_ratios=[1, 0.68], hspace=0.40, wspace=0.34)

    ax_A = fig.add_subplot(gs[0, 0])
    ax_B = fig.add_subplot(gs[0, 1])
    ax_C = fig.add_subplot(gs[1, :])       # money panel: full width, short

    panel_A_rna(ax_A, rna_traj)
    panel_B_meth(ax_B, meth_traj)
    panel_C_overlay(ax_C, rna_traj, meth_traj)

    # standalone panel labels, pulled out to the upper-left of each axes
    for ax, lab, dx in [(ax_A, "A", -0.15), (ax_B, "B", -0.15), (ax_C, "C", -0.07)]:
        ax.text(dx, 1.12, lab, transform=ax.transAxes,
                fontsize=16, fontweight="bold", va="top", ha="left")

    out_png = f"{OUT}/fig3_temporal_cascade.png"
    out_pdf = f"{OUT}/fig3_temporal_cascade.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved: {out_png}")
    print(f"  RNA UP n={rna_traj['up']['n']}, DOWN n={rna_traj['dn']['n']}, "
          f"Methyl CpG n={meth_traj['n']}")
    plt.close()


if __name__ == "__main__":
    main()
