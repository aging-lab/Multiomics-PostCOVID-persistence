# Distinct temporal phases of transcriptomic persistence and delayed methylomic remodeling in whole blood after COVID-19

Analysis and figure-generation code for the study profiling paired whole-blood
transcriptomes and DNA methylomes across a tri-cohort of acute COVID-19 patients
(sampled longitudinally), cross-sectional convalescent survivors, and healthy
controls.

This repository contains **scripts only** (no data or intermediate results). It
is provided for transparency and reproducibility; the scripts read from a local
copy of the project data (see below).

## Data availability

- Whole-blood RNA sequencing: Gene Expression Omnibus (GEO) **GSE300696**
- Targeted bisulfite sequencing (DNA methylation): GEO **GSE301082**
- Healthy-control whole-blood data (Korean Genome Project): available from the
  corresponding author upon reasonable request.

Persistent/delayed marker lists and cis-eQTM pairs are provided as Supplementary
Tables with the paper.

## Repository layout

Scripts follow the analysis pipeline order:

- `2.Data_preparation/` — PCA, batch correction (ComBat-seq), RNA normalization,
  methylation QC, CpG matrix assembly, and cell-type deconvolution.
- `3.Discovery/` — differential expression (RNA) and differential methylation
  against the healthy-control baseline; persistent/delayed marker definition and
  canonical-symbol collapse.
- `4.Trajectory/` — healthy-control-anchored gene-set scores, per-gene Recovery
  Index and recovery archetypes, and the delayed-methylation trajectory.
- `5.Integration/` — cis-eQTM, mediation, and acute-severity (mild-moderate vs
  severe-critical) analyses.
- `6.Robustness_analyses/` — severity effect-size confidence intervals,
  by-subject cluster-robust repeated-measures models, TOST equivalence tests, and
  methylation batch/technical-shift sensitivity analyses.
- `figures/` — main (Fig. 1-5) and supplementary figure generation.

## Usage

The scripts reference the project data via a `PROJECT_ROOT` placeholder, e.g.
`PROJECT_ROOT/research/LongCOVID/Results/...`. Before running, replace
`PROJECT_ROOT` with the path to your local copy of the data (or edit the path
variables at the top of each script). The public deposits above provide the raw
and processed sequencing data these scripts operate on.

Scripts are numbered to indicate execution order within each stage.

## Environment

- **Python 3**: numpy, scipy, pandas, scikit-learn, statsmodels, patsy, gseapy,
  matplotlib; differential expression via DESeq2 (inmoose implementation).
- **R**: lcmm (latent-class modelling), plus standard RNA-seq utilities.
- External tools: CIBERSORTx (RNA deconvolution, LM22 signature) and EpiDISH
  (methylation deconvolution).

## Citation

If you use this code, please cite the associated paper:

> [Authors]. Distinct temporal phases of transcriptomic persistence and delayed
> methylomic remodeling in whole blood after COVID-19. [Journal], [year].
> [DOI to be added upon publication.]

## Code use

This code is released for transparency and reproducibility. It is provided
without a formal open-source license (all rights reserved). For reuse,
redistribution, or adaptation, please contact the corresponding author.
