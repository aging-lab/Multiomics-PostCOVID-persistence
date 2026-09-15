"""
Build methyl_blood_ref.tsv by mapping EpiDISH 450K probe IDs to hg38 genome positions.

Input:
  epidish_centDHS_probes.tsv           — EpiDISH centDHSbloodDMC reference (333 probes x 7 cell types)
  HumanMethylation450_15017482_v1-2.csv — Illumina 450K manifest (hg19 coordinates)
  hg19ToHg38.over.chain.gz             — UCSC liftOver chain (hg19 -> hg38)

Output:
  methyl_blood_ref.tsv  — cpg_id (chrN_pos, hg38) x 7 cell types, mean beta-values
"""

import os
import pandas as pd
from pyliftover import LiftOver

HERE = os.path.dirname(os.path.abspath(__file__))

PROBES_PATH   = os.path.join(HERE, "epidish_centDHS_probes.tsv")
MANIFEST_PATH = os.path.join(HERE, "HumanMethylation450_15017482_v1-2.csv")
CHAIN_PATH    = os.path.join(HERE, "hg19ToHg38.over.chain.gz")
OUT_PATH      = os.path.join(HERE, "methyl_blood_ref.tsv")

print("Loading EpiDISH probe reference...")
probes = pd.read_csv(PROBES_PATH, sep="\t", index_col="probe_id")
print(f"  {len(probes)} probes, cell types: {list(probes.columns)}")

print("Loading Illumina 450K manifest (hg19)...")
manifest = pd.read_csv(MANIFEST_PATH, skiprows=7, low_memory=False,
                       usecols=["IlmnID", "CHR", "MAPINFO"])
manifest = manifest.dropna(subset=["CHR", "MAPINFO"])
manifest = manifest[~manifest["CHR"].astype(str).isin(["X", "Y", "M"])]
manifest["CHR"] = "chr" + manifest["CHR"].astype(str)
manifest["MAPINFO"] = manifest["MAPINFO"].astype(int)
manifest = manifest.set_index("IlmnID")
print(f"  {len(manifest):,} autosomal probes with hg19 positions")

print("Matching EpiDISH probes to manifest...")
common = probes.index.intersection(manifest.index)
print(f"  Matched: {len(common)} / {len(probes)}")
ref_hg19 = manifest.loc[common][["CHR", "MAPINFO"]].copy()

print("Lifting over hg19 -> hg38 positions...")
lo = LiftOver(CHAIN_PATH)
hg38_rows = []
for probe_id, row in ref_hg19.iterrows():
    # pyliftover uses 0-based coordinates
    result = lo.convert_coordinate(row["CHR"], row["MAPINFO"] - 1)
    if result and len(result) > 0:
        chrom38 = result[0][0]
        pos38   = result[0][1] + 1  # convert back to 1-based
        hg38_rows.append({"probe_id": probe_id, "cpg_id": f"{chrom38}_{pos38}"})

hg38_df = pd.DataFrame(hg38_rows).set_index("probe_id")
print(f"  Successfully lifted over: {len(hg38_df)} / {len(common)}")

print("Building reference matrix (hg38)...")
ref = probes.loc[hg38_df.index].copy()
ref.index = hg38_df["cpg_id"].values
ref.index.name = "cpg_id"
ref = ref.groupby("cpg_id").mean()
print(f"  Unique hg38 positions: {len(ref)}")

ref.to_csv(OUT_PATH, sep="\t", float_format="%.6f")
print(f"\nSaved -> {OUT_PATH}")
print(f"  Shape: {ref.shape[0]} CpGs x {ref.shape[1]} cell types")
print(f"  CpG_id example: {ref.index[0]} (hg38)")
