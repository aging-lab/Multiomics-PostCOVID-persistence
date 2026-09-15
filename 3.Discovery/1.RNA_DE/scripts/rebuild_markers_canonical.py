#!/usr/bin/env python3
"""Rebuild RNA marker table with canonical-symbol collapse + IG/pseudogene removal.

Policy (decided 2026-07-30):
  - collapse duplicate Ensembl IDs sharing a gene symbol to ONE canonical row
    (canonical = lowest ENSG accession number, a proxy for the primary gene;
     GENCODE biotype unavailable offline).
  - DROP immunoglobulin (IG*/TR* V/D/J/C segments) and pseudogenes
    (pseudogene = symbol '<Parent>P<digits>' whose <Parent> is itself a real
     gene symbol in the tested universe).
  - KEEP unnamed features (symbol == bare ENSG id) — they are simply un-named,
    not artefacts.

Outputs (NEW filtered files; originals untouched):
  3.Discovery/1.RNA_DE/data/rna_markers_canonical_filtered.tsv
  4.Trajectory/3.Archetypes/data/gene_archetypes_filtered.tsv
and prints every recomputed constant the figure scripts need.
"""
import re, csv
from collections import defaultdict, Counter

NEW = "PROJECT_ROOT/research/LongCOVID/Results_updated_260714"
G2      = f"{NEW}/3.Discovery/1.RNA_DE/data/gate2_rc_vs_hc_clean.tsv"
MARK    = f"{NEW}/3.Discovery/1.RNA_DE/data/rna_markers_cleancohort.tsv"
ARC     = f"{NEW}/4.Trajectory/3.Archetypes/data/gene_archetypes_clean.tsv"
OUT_M   = f"{NEW}/3.Discovery/1.RNA_DE/data/rna_markers_canonical_filtered.tsv"
OUT_A   = f"{NEW}/4.Trajectory/3.Archetypes/data/gene_archetypes_filtered.tsv"

sym = lambda g: re.sub(r"^ENSG[0-9.]+_", "", g)
def ensg_num(g):
    m = re.match(r"^ENSG(\d+)", g)
    return int(m.group(1)) if m else 10**12

# ---- universe of real symbols (for pseudogene parent test) ----
universe = set()
with open(G2) as f:
    next(f)
    for line in f:
        universe.add(sym(line.split("\t")[0]))
parents = {s for s in universe if not re.search(r"P[0-9]+$", s)}

def is_pseudo(s):
    m = re.match(r"^(.*?)P[0-9]+$", s)
    return bool(m) and m.group(1) in parents
def is_ig(s):
    return bool(re.match(r"^(IGH[VDJ]|IGK[VJ]|IGL[VJ]|IGHG[0-9]|IGHA[0-9]|IGHM$|IGHD$|IGHE$|IGKC$|IGLC[0-9]|TR[ABGD][VDJ])", s))
def is_unnamed(s):
    return bool(re.match(r"^ENSG[0-9]+$", s))

# ---- read marker table, group rows by symbol ----
with open(MARK) as f:
    rd = csv.reader(f, delimiter="\t")
    header = next(rd)
    rows = [r for r in rd]
idx = {c: i for i, c in enumerate(header)}
by_sym = defaultdict(list)
for r in rows:
    by_sym[sym(r[idx["gene_id"]])].append(r)

# ---- canonical collapse: 1 row per symbol (lowest ENSG accession) ----
canonical = {}
for s, rs in by_sym.items():
    canonical[s] = sorted(rs, key=lambda r: ensg_num(r[idx["gene_id"]]))[0]

# ---- classify + filter ----
kept, dropped = {}, {"IG": [], "pseudo": []}
for s, r in canonical.items():
    if is_ig(s):
        dropped["IG"].append(s); continue
    if is_pseudo(s):
        dropped["pseudo"].append(s); continue
    kept[s] = r

# ---- write filtered marker table ----
with open(OUT_M, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(["symbol"] + header)
    for s in sorted(kept):
        w.writerow([s] + kept[s])

# ---- recompute counts ----
cat_i = idx["category"]; g1_i = idx["in_gate1"]
fc2_i = idx["log2FC_gate2_clean"]; fc1_i = idx["log2FC_gate1"]
truthy = lambda v: str(v).strip().lower() in ("true", "1", "yes")

cats = Counter(kept[s][cat_i] for s in kept)
# acute contrast = significant in gate1 (persistent + acute_only)
acute = [s for s in kept if truthy(kept[s][g1_i])]
acute_up = sum(1 for s in acute if float(kept[s][fc1_i]) > 0)
acute_dn = sum(1 for s in acute if float(kept[s][fc1_i]) < 0)
# persistent up/down (by convalescent log2FC sign)
pers = [s for s in kept if kept[s][cat_i] == "persistent"]
pers_up = sum(1 for s in pers if float(kept[s][fc2_i]) > 0)
pers_dn = sum(1 for s in pers if float(kept[s][fc2_i]) < 0)

print("=== DROPPED ===")
print(f"  IG      : {len(dropped['IG'])}")
print(f"  pseudo  : {len(dropped['pseudo'])}")
print("=== FILTERED MARKER COUNTS (canonical, 1/symbol) ===")
print(f"  persistent   : {cats['persistent']}")
print(f"  delayed      : {cats['delayed']}")
print(f"  acute_only   : {cats['acute_only']}")
print(f"  acute total  : {len(acute)}  (up {acute_up} / down {acute_dn})")
print(f"  persistent up/down : {pers_up} / {pers_dn}")
print(f"  -> wrote {OUT_M}")

# ---- filter archetype file to kept persistent symbols (dedup) ----
# NOTE: keep the original archetype label 'Plateau' in the file — the figure
# scripts key on 'Plateau' internally and render it as 'Sustained'. Renaming
# here would break fig3/fig4 lookups. Display map applied only in printout below.
DISPLAY = {"Plateau": "Sustained"}
pers_set = set(pers)
arc_rows = {}
with open(ARC) as f:
    rd = csv.reader(f, delimiter="\t"); ah = next(rd)
    ai = {c: i for i, c in enumerate(ah)}
    for r in rd:
        g = r[ai["gene_id"]]
        if g in pers_set:
            arc_rows[g] = r          # dedup identical rows; label unchanged
with open(OUT_A, "w", newline="") as f:
    w = csv.writer(f, delimiter="\t")
    w.writerow(ah)
    for g in sorted(arc_rows):
        w.writerow(arc_rows[g])
acnt = Counter(DISPLAY.get(arc_rows[g][ai["archetype"]], arc_rows[g][ai["archetype"]]) for g in arc_rows)
tot = sum(acnt.values())
print("=== FILTERED ARCHETYPES (persistent) ===")
order = ["Escalated", "Sustained", "Partial", "Recovered", "Overshoot"]
for a in order:
    print(f"  {a:10} {acnt.get(a,0):4}  ({100*acnt.get(a,0)/tot:.1f}%)")
nr = acnt.get("Escalated", 0) + acnt.get("Sustained", 0)
print(f"  non-recovering (Esc+Sus): {nr} ({100*nr/tot:.1f}%)   total={tot}")
print(f"  coverage: {len(arc_rows)}/{len(pers_set)} persistent symbols found in archetype file")
print(f"  -> wrote {OUT_A}")
