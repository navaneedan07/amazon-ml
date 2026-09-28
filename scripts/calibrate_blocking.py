"""Stage A: calibrate the product-cap blocking budget.

Builds exact DF tables (sorted unique hashes + counts) for
  name_uni (len>=3), addr_uni (len>=3), addr_bi
on both the train S1 side and the target side, then:
  1. samples missed GT pairs (from the old cap-60 candidate set)
  2. for each, the minimum s1df*tgtdf product over shared keys
  3. prints coverage-of-misses vs product-budget curves
  4. prints added candidate volume vs product-budget curves
"""
import mmap
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

# ensure src/ is in sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from blocking import split_pairs, _hash_strs

BASE = ROOT / "cache" / "train"
KEEP = []


def blob(name):
    path = str(BASE / name)
    off = np.load(path + ".off.npy", mmap_mode="r")
    f = open(path, "rb")
    KEEP.append(f)
    m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    KEEP.append(m)

    def get(i):
        a, b = int(off[i]), int(off[i + 1])
        return m[a:b].decode("utf-8") if a != b else ""
    return get, len(off) - 1


def hashes_for(rows, getter, mode, min_len=3):
    out = []
    for i in rows:
        s = getter(int(i))
        if not s:
            continue
        tt = s.split()
        if mode == "uni":
            ks = [t for t in tt if len(t) >= min_len]
        else:
            ks = [f"{tt[j]} {tt[j+1]}" for j in range(len(tt) - 1)]
        if ks:
            out.append(_hash_strs(ks))
    return np.concatenate(out) if out else np.empty(0, np.uint64)


def df_table(h, tag):
    t0 = time.time()
    uh, cnt = np.unique(h, return_counts=True)
    np.savez(BASE / f"df_{tag}.npz", h=uh, c=cnt)
    print(f"  {tag}: {len(h):,} postings -> {len(uh):,} keys "
          f"({time.time()-t0:.0f}s)", flush=True)
    return uh, cnt


def main():
    s1_names, n_s1 = blob("s1_names.bin")
    tgt_names, n_t = blob("tgt_names.bin")
    s1_addr, _ = blob("s1_addrs.bin")
    tgt_addr, _ = blob("tgt_addrs.bin")
    print(f"s1 {n_s1:,}  tgt {n_t:,}", flush=True)

    all_rows_s1 = np.arange(n_s1)
    all_rows_t = np.arange(n_t)
    tables = {}
    for tag, rows, getter, mode in (
        ("s1_name", all_rows_s1, s1_names, "uni"),
        ("s1_addr", all_rows_s1, s1_addr, "uni"),
        ("s1_abi", all_rows_s1, s1_addr, "bi"),
        ("t_name", all_rows_t, tgt_names, "uni"),
        ("t_addr", all_rows_t, tgt_addr, "uni"),
        ("t_abi", all_rows_t, tgt_addr, "bi"),
    ):
        h = hashes_for(rows, getter, mode)
        tables[tag] = df_table(h, tag)
        del h

    gt = np.load(BASE / "gt_pairs.npy")
    pairs = np.load(BASE / "pairs.npy", mmap_mode="r")
    idx = np.searchsorted(pairs, gt)
    idx_c = np.minimum(idx, len(pairs) - 1)
    miss = gt[pairs[idx_c] != gt]
    miss_s1, miss_t = split_pairs(miss)
    print(f"GT {len(gt):,}  missed {len(miss):,}", flush=True)

    rng = np.random.default_rng(0)
    S = 60_000
    sel = rng.choice(len(miss), S, replace=False)
    ms1, mt = miss_s1[sel].astype(np.int64), miss_t[sel].astype(np.int64)

    def lk(tag):
        uh, cnt = tables[tag]
        return uh, cnt

    s1n_u, s1n_c = lk("s1_name")
    s1a_u, s1a_c = lk("s1_addr")
    s1b_u, s1b_c = lk("s1_abi")
    tn_u, tn_c = lk("t_name")
    ta_u, ta_c = lk("t_addr")
    tb_u, tb_c = lk("t_abi")

    minprod = np.full(S, np.int64(-1))
    share3 = np.zeros(S, dtype=bool)
    BUDGETS = [3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000]
    cov = {b: 0 for b in BUDGETS}
    t0 = time.time()
    for k in range(S):
        hs = np.concatenate([
            hashes_for([ms1[k]], s1_names, "uni"),
            hashes_for([ms1[k]], s1_addr, "uni"),
            hashes_for([ms1[k]], s1_addr, "bi"),
        ])
        ht = np.concatenate([
            hashes_for([mt[k]], tgt_names, "uni"),
            hashes_for([mt[k]], tgt_addr, "uni"),
            hashes_for([mt[k]], tgt_addr, "bi"),
        ])
        if len(hs) == 0 or len(ht) == 0:
            continue
        hu = np.unique(hs)
        ht_u, ht_idx = np.unique(ht, return_index=True)
        shared, sA, sB = np.intersect1d(hu, ht_u, assume_unique=True, return_indices=True)
        if len(shared) == 0:
            continue
        a = np.searchsorted(tn_u, shared)
        a_c = np.minimum(a, len(tn_u) - 1)
        tdf_n = np.where(tn_u[a_c] == shared, tn_c[a_c], 0)
        a = np.searchsorted(ta_u, shared)
        a_c = np.minimum(a, len(ta_u) - 1)
        tdf_a = np.where(ta_u[a_c] == shared, ta_c[a_c], 0)
        a = np.searchsorted(tb_u, shared)
        a_c = np.minimum(a, len(tb_u) - 1)
        tdf_b = np.where(tb_u[a_c] == shared, tb_c[a_c], 0)
        a = np.searchsorted(s1n_u, shared)
        a_c = np.minimum(a, len(s1n_u) - 1)
        sdf_n = np.where(s1n_u[a_c] == shared, s1n_c[a_c], 0)
        a = np.searchsorted(s1a_u, shared)
        a_c = np.minimum(a, len(s1a_u) - 1)
        sdf_a = np.where(s1a_u[a_c] == shared, s1a_c[a_c], 0)
        a = np.searchsorted(s1b_u, shared)
        a_c = np.minimum(a, len(s1b_u) - 1)
        sdf_b = np.where(s1b_u[a_c] == shared, s1b_c[a_c], 0)
        tdf = np.maximum(np.maximum(tdf_n, tdf_a), tdf_b)
        sdf = np.maximum(np.maximum(sdf_n, sdf_a), sdf_b)
        prod = sdf.astype(np.int64) * tdf.astype(np.int64)
        prod = np.where((tdf > 0) & (sdf > 0), prod, np.iinfo(np.int64).max)
        mp = prod.min()
        minprod[k] = mp
        for b in BUDGETS:
            if mp <= b:
                cov[b] += 1
        uni_tdf = np.where(tdf_n > 0, tdf_n, tdf_a)
        if ((uni_tdf > 0) & (uni_tdf <= 60)).any():
            share3[k] = True
        if k % 10000 == 0:
            print(f"  ...{k} ({time.time()-t0:.0f}s)", flush=True)

    print("\n== coverage of missed pairs (share key with product <= B) ==", flush=True)
    print(f"  free (len3 tok, tgtdf<=60): {share3.mean():.3f}", flush=True)
    for b in BUDGETS:
        print(f"  B={b:>9,}: {cov[b]/S:.3f}", flush=True)

    print("\n== added volume (pairs, full scale) per budget ==", flush=True)
    for b in BUDGETS:
        tot = 0
        for s_tag, t_tag, base_keep in (("s1_name", "t_name", True),
                                        ("s1_addr", "t_addr", True),
                                        ("s1_abi", "t_abi", False)):
            su, sc = tables[s_tag]
            tu, tc = tables[t_tag]
            a = np.searchsorted(tu, su)
            a_c = np.minimum(a, len(tu) - 1)
            m = tu[a_c] == su
            sh, ss, tt = su[m], sc[m], tc[a_c[m]]
            prod = ss.astype(np.int64) * tt.astype(np.int64)
            if base_keep:
                m2 = tt > 60
                sh, prod = sh[m2], prod[m2]
            tot += int(np.minimum(prod, b).sum())
        print(f"  B={b:>9,}: +{tot/1e6:,.0f}M pairs over the 111.5M baseline", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
