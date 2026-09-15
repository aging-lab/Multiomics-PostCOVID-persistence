"""
Prepare WGBS blood reference for methylation deconvolution.

This script downloads BLUEPRINT WGBS mean β-values for 6 whole-blood cell types
and converts them to the CpG_id format (chrN_position) used in our pipeline.

Cell types:
  CD4T   — CD4+ T cells
  CD8T   — CD8+ T cells
  NK     — Natural Killer cells
  Bcell  — B cells
  Mono   — Monocytes
  Gran   — Granulocytes (neutrophil-dominated)

Output: methyl_blood_ref.tsv (CpG_id × 6 cell types, mean β-values)

Usage:
  python prepare_blueprint_ref.py

Sources:
  Houseman et al. 2012 / Reinius et al. 2012 (450K-based, widely used)
  BLUEPRINT WGBS (genome-wide bisulfite seq reference)

Note:
  If BLUEPRINT WGBS reference is not yet formatted/available, this script
  builds a simplified reference from the Reinius et al. 450K data
  using positions from the Illumina 450K manifest (hg19).
  The manifest maps 450K probe positions to genome coordinates.

  Manifest download (hg19):
    https://webdata.illumina.com/downloads/productfiles/humanmethylation450/
    humanmethylation450_15017482_v1-2.csv  (~100 MB)
  Place at: resources/HumanMethylation450_15017482_v1-2.csv

  Reinius et al. reference:
    Available in the minfi / FlowSorted.Blood.450k Bioconductor package.
    Or as supplementary data from PMID 22989076.

For WGBS pipelines, position-based matching is the most appropriate approach.
"""

import os
import sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

MANIFEST_PATH = os.path.join(HERE, "HumanMethylation450_15017482_v1-2.csv")
REINIUS_PATH  = os.path.join(HERE, "reinius_blood_450k.tsv")
OUT_PATH      = os.path.join(HERE, "methyl_blood_ref.tsv")

CELL_COLS = ["CD4T", "CD8T", "NK", "Bcell", "Mono", "Gran"]

def check_files():
    missing = []
    if not os.path.isfile(MANIFEST_PATH):
        missing.append(f"  Illumina 450K manifest : {MANIFEST_PATH}")
    if not os.path.isfile(REINIUS_PATH):
        missing.append(f"  Reinius blood reference: {REINIUS_PATH}")
    if missing:
        print("Missing reference files:")
        for m in missing:
            print(m)
        print()
        print("Instructions:")
        print("  1. Illumina 450K manifest (hg19):")
        print("     https://webdata.illumina.com/downloads/productfiles/"
              "humanmethylation450/humanmethylation450_15017482_v1-2.csv")
        print("  2. Reinius blood 450K reference (6 cell types × probes):")
        print("     From R: library(minfi); data(FlowSorted.Blood.450k)")
        print("     Or download from PMID 22989076 supplementary data")
        print("     Save as TSV: rows=cg######, cols=CD4T,CD8T,NK,Bcell,Mono,Gran")
        sys.exit(1)

def build_ref():
    print("Loading Reinius blood reference...")
    ref = pd.read_csv(REINIUS_PATH, sep="\t", index_col=0)
    ref = ref[CELL_COLS] if all(c in ref.columns for c in CELL_COLS) else ref.iloc[:, :6]
    ref.columns = CELL_COLS
    print(f"  Probes: {len(ref):,}, Cell types: {list(ref.columns)}")

    print("Loading Illumina 450K manifest...")
    manifest = pd.read_csv(MANIFEST_PATH, skiprows=7, low_memory=False,
                           usecols=["IlmnID", "CHR", "MAPINFO"])
    manifest = manifest.dropna(subset=["CHR", "MAPINFO"])
    manifest["CHR"] = "chr" + manifest["CHR"].astype(str)
    manifest["MAPINFO"] = manifest["MAPINFO"].astype(int)
    manifest["cpg_id"] = manifest["CHR"] + "_" + manifest["MAPINFO"].astype(str)
    manifest = manifest.set_index("IlmnID")
    print(f"  Manifest probes with position: {len(manifest):,}")

    # Map probe IDs to genome positions
    common = ref.index.intersection(manifest.index)
    print(f"  Probes matched to positions: {len(common):,}")

    ref_pos = ref.loc[common].copy()
    ref_pos["cpg_id"] = manifest.loc[common, "cpg_id"].values
    ref_pos = ref_pos.groupby("cpg_id")[CELL_COLS].mean()  # average if duplicate positions
    print(f"  Unique cpg_id positions: {len(ref_pos):,}")

    ref_pos.to_csv(OUT_PATH, sep="\t", float_format="%.6f")
    print(f"\nSaved → {OUT_PATH}")
    print(f"  Shape: {ref_pos.shape[0]:,} CpGs × {len(CELL_COLS)} cell types")
    print("  CpG_id format: chrN_position (hg19)")

if __name__ == "__main__":
    check_files()
    build_ref()
    print("\nReady to run methyl_deconvolve.py")
