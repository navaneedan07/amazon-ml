"""Exploration pass 2: streaming analysis of GT pairs & blocking-key coverage.

Memory-safe: no posting lists; per-pair key comparison only. Answers:
  * country consistency of GT pairs
  * GT degree distribution (matches per S1 entity)
  * which blocking keys cover GT pairs (name_exact / name_sig / name_tok /
    addr_tok / house / postal)
  * example hard pairs (not covered by any simple key)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code/business_entity_resolution/src"))
from textnorm import norm_name, norm_addr, name_sig, house_num, postal_code  # noqa: E402

DATA = Path("student_resource/dataset")
CACHE = Path("analysis/cache_explore2")
CHUNK = 300_000


def main():
    # ---- load normalized fields for all sources -----------------------------
    CACHE.mkdir(exist_ok=True)
    norm = {}
    for src, fname in (("s1", "train/train_source1.tsv"),
                       ("s2", "train/train_source2.tsv"),
                       ("s3", "train/train_source3.tsv")):
        f = CACHE / f"{src}.npz"
        if f.exists():
            z = np.load(f, allow_pickle=False)
            norm[src] = {k: z[k] for k in z.files}
            print(f"  {src}: loaded {len(norm[src]['ids'])} rows from cache", flush=True)
            continue
        ids, names, addrs, ctry = [], [], [], []
        k = 0
        for ch in pd.read_csv(DATA / fname, sep="\t", dtype=str,
                              usecols=["entity_id", "business_name", "business_address", "country"],
                              chunksize=CHUNK):
            ids.extend(ch.entity_id)
            names.extend(ch.business_name.map(norm_name))
            addrs.extend(ch.business_address.map(norm_addr))
            ctry.extend(ch.country)
            k += len(ch)
            if k % 1_000_000 == 0:
                print(f"  {src}: {k}", flush=True)
        np.savez(f, ids=np.asarray(ids), names=np.asarray(names, dtype=object),
                 addrs=np.asarray(addrs, dtype=object), ctry=np.asarray(ctry))
        norm[src] = dict(ids=np.asarray(ids), names=np.asarray(names, dtype=object),
                         addrs=np.asarray(addrs, dtype=object), ctry=np.asarray(ctry))
        print(f"  {src}: {len(ids)} rows", flush=True)

    id2row = {src: {eid: i for i, eid in enumerate(norm[src]["ids"])}
              for src in ("s1", "s2", "s3")}

    # ---- explode GT ---------------------------------------------------------
    print("exploding GT ...", flush=True)
    gt = pd.read_csv(DATA / "train/train_ground_truth.tsv", sep="\t", dtype=str).fillna("")
    split = gt.matched_entity_ids.str.split(",")
    lens = split.map(lambda lst: sum(1 for m in lst if m)).to_numpy()
    mids = np.fromiter((m for lst in split for m in lst if m),
                       dtype=object, count=int(sum(lens)))
    s1_exp = np.repeat(gt.source1_entity_id.to_numpy(), lens)

    r1 = np.asarray([id2row["s1"].get(x, -1) for x in s1_exp], dtype=np.int64)
    src2 = np.where(pd.Series(mids).str.startswith("S2").to_numpy(), "s2", "s3")
    r2 = np.asarray([id2row[s].get(m, -1) for s, m in zip(src2, mids)], dtype=np.int64)
    ok = (r1 >= 0) & (r2 >= 0)
    print(f"GT pairs: {int(ok.sum())} (unmapped {int((~ok).sum())})", flush=True)
    r1, r2, src2 = r1[ok], r2[ok], src2[ok]

    c1 = norm["s1"]["ctry"][r1]
    c2 = np.asarray([norm[s]["ctry"][r] for s, r in zip(src2, r2)], dtype=object)
    print("country same ratio:", float((c1 == c2).mean()), flush=True)

    uniq1, inv = np.unique(r1, return_inverse=True)
    deg = pd.Series(np.bincount(inv))
    print("GT degree describe:\n", deg.describe().to_string(), flush=True)
    n_s1 = len(norm["s1"]["ids"])
    print("S1 with >=1 match:", len(uniq1), "/", n_s1, flush=True)

    # ---- per-pair key coverage ----------------------------------------------
    print("key coverage ...", flush=True)
    n = len(r1)
    cov = {k: np.zeros(n, dtype=bool) for k in
           ("name_exact", "name_sig", "name_tok", "addr_tok", "house", "postal")}
    for k in range(n):
        tr = r2[k]
        s = src2[k]
        n1 = norm["s1"]["names"][r1[k]]
        n2 = norm[s]["names"][tr]
        a1 = norm["s1"]["addrs"][r1[k]]
        a2 = norm[s]["addrs"][tr]
        cov["name_exact"][k] = bool(n1) and n1 == n2
        cov["name_sig"][k] = bool(n1) and name_sig(n1) == name_sig(n2)
        t1 = set(t for t in n1.split() if len(t) >= 4)
        t2 = set(t for t in n2.split() if len(t) >= 4)
        cov["name_tok"][k] = bool(t1 & t2)
        at1 = set(t for t in a1.split() if len(t) >= 4)
        at2 = set(t for t in a2.split() if len(t) >= 4)
        cov["addr_tok"][k] = bool(at1 & at2)
        h1, h2 = house_num(a1), house_num(a2)
        cov["house"][k] = bool(h1) and h1 == h2
        z1, z2 = postal_code(a1), postal_code(a2)
        cov["postal"][k] = bool(z1) and z1 == z2
        if k % 500_000 == 0 and k:
            print(f"  {k}/{n}", flush=True)

    for kk, v in cov.items():
        print(f"{kk}: {v.mean():.4f}", flush=True)
    anycov = np.zeros(n, dtype=bool)
    for v in cov.values():
        anycov |= v
    print("ANY key:", round(float(anycov.mean()), 4), flush=True)

    hard = np.where(~anycov)[0][:12]
    for k in hard:
        s = src2[k]
        tr = r2[k]
        print("HARD S1:", norm["s1"]["names"][r1[k]], "|", norm["s1"]["addrs"][r1[k]])
        print("     T  :", norm[s]["names"][tr], "|", norm[s]["addrs"][tr])

    for s in ("s2", "s3"):
        names = pd.Series(norm[s]["names"])
        sigs = names.map(name_sig)
        print(s, "dup raw name ratio:", round(float(names.duplicated().mean()), 4),
              "dup sig ratio:", round(float(sigs.duplicated().mean()), 4))


if __name__ == "__main__":
    main()
