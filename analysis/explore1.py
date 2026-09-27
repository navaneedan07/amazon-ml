"""Exploration pass 1: country consistency, match samples, France patterns."""
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path("student_resource/dataset")

# --- country consistency of GT matches -------------------------------------
print("loading id->country maps ...", flush=True)
cols = ["entity_id", "country"]
s1 = pd.read_csv(DATA / "train/train_source1.tsv", sep="\t", dtype=str, usecols=cols)
s2 = pd.read_csv(DATA / "train/train_source2.tsv", sep="\t", dtype=str, usecols=cols)
s3 = pd.read_csv(DATA / "train/train_source3.tsv", sep="\t", dtype=str, usecols=cols)
gt = pd.read_csv(DATA / "train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")

ctry = {}
for df, pref in ((s1, "S1"), (s2, "S2"), (s3, "S3")):
    m = dict(zip(df.entity_id, df.country))
    ctry.update(m)

# explode GT
expl = gt[gt.matched_entity_ids != ""].copy()
expl["mids"] = expl.matched_entity_ids.str.split(",")
expl = expl.explode("mids")
expl = expl[expl.mids != ""]
print("exploded GT pairs:", len(expl), flush=True)

expl["c1"] = expl.source1_entity_id.map(ctry)
expl["c2"] = expl.mids.map(ctry)
same = expl.c1 == expl.c2
print("country same ratio:", same.mean())
print("mismatched country pairs:", (~same).sum())
if (~same).sum():
    print(expl[~same].head(20).to_string())

# per-country match counts
print("\nmatches per S1 by country:")
expl["n"] = 1
cnt = expl.groupby("source1_entity_id").size()
s1c = s1.set_index("entity_id").country
per = pd.DataFrame({"n": cnt}).join(s1c, how="left")
print(per.groupby("country")["n"].describe())

# --- sample matched pairs ---------------------------------------------------
print("\n--- sample matched pairs ---")
full = lambda p: pd.read_csv(DATA / p, sep="\t", dtype=str).fillna("")
s1f = full("train/train_source1.tsv").set_index("entity_id")
s2f = full("train/train_source2.tsv").set_index("entity_id")
s3f = full("train/train_source3.tsv").set_index("entity_id")
s2i, s3i, s1i = s2f, s3f, s1f
np.random.seed(0)
sample = expl.sample(15, random_state=1)
for _, r in sample.iterrows():
    a = s1i.loc[r.source1_entity_id]
    b = (s2i if r.mids.startswith("S2") else s3i).loc[r.mids]
    print(f"[{a.country}] S1: {a.business_name} | {a.business_address}")
    print(f"        OTH: {b.business_name} | {b.business_address}")

# --- France test records ----------------------------------------------------
print("\n--- France test samples ---")
t1 = pd.read_csv(DATA / "test/test_source1.tsv", sep="\t", dtype=str)
fr = t1[t1.country == "France"].sample(10, random_state=2)
for _, r in fr.iterrows():
    print(f"{r.business_name} | {r.business_address}")

# --- duplicate analysis within sources -------------------------------------
print("\n--- exact duplicate names within S2 ---")
for name, df in (("S2", s2f), ("S3", s3f)):
    dup = df.business_name.str.lower().str.strip().duplicated().mean()
    print(name, "dup name ratio:", round(dup, 4))

# entity id prefix check
print("\nID prefixes:", s1.entity_id.str[:3].unique(), s2.entity_id.str[:3].unique(), s3.entity_id.str[:3].unique())
print("GT ids all in s1:", gt.source1_entity_id.isin(set(s1.entity_id)).all())
allmatch = set(expl.mids.unique())
print("GT match ids subset of s2|s3:", allmatch <= (set(s2.entity_id) | set(s3.entity_id)))
