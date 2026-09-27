"""Sweep blocking DF-caps + per-S1 top-M on the first N train S1 rows.

For each combo:
  - filter the loose index per key type (with_cap)
  - probe all key types per chunk, tagging each pair with the key-type weight
  - keep max weight per pair, then per-S1 top-M by weight (deterministic)
  - report pairs/row, blocking recall on the row range, and weight histogram
    of missed GT pairs (shows which key type loses each miss)

Usage: python analysis/tune_blocking.py [N_ROWS]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import dataio
from blocking import split_pairs, combine_pairs, query_keys

cache = ROOT / "cache"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
CHUNK = 100_000

# key-type weights: how strongly a shared key of this type votes for a pair
WEIGHTS = {
    "name_exact": 10.0, "name_sig": 8.0, "name_nospace": 8.0,
    "name_bi": 4.0, "name_tok": 2.5, "addr_tok": 1.5, "addr_bi": 1.0,
    "house": 2.0, "house_norm": 2.0, "postal": 0.5,
}
KEYS = list(WEIGHTS)
W = np.array([WEIGHTS[k] for k in KEYS], dtype=np.float32)

s1 = dataio.load_role(str(cache), "train", "s1", with_text=True)
tgt = dataio.load_role(str(cache), "train", "tgt", with_text=True)
with open(cache / "train" / "index_loose.pkl", "rb") as fh:
    loose = pickle.load(fh)
print("loose sizes:", {ix.name: len(ix) for ix in loose}, flush=True)
korder = {ix.name: i for i, ix in enumerate(loose)}

gt = np.load(cache / "train" / "gt_pairs.npy")
lo = int(np.searchsorted(gt, np.uint64(0), side="left"))
hi = int(np.searchsorted(gt, (np.uint64(N) << np.uint64(32)) - np.uint64(1),
                         side="right"))
seg = gt[lo:hi]
seg_gt = int(((seg >> np.uint64(32)) < N).sum())
print(f"rows 0-{N:,}: {seg_gt:,} GT pairs", flush=True)

s1c, tgc = s1["country"], tgt["country"]


def block_chunk_weighted(indexes, start, end):
    """Probe every key type; return (pairs, weight_of_pair) deduped by max w."""
    qkeys = query_keys(s1, start, end, indexes)
    all_p, all_w, all_k = [], [], []
    rows = np.arange(start, end, dtype=np.int64)
    for ix, qh, qr in qkeys:
        if len(qh) == 0:
            continue
        pairs = ix.probe(qh, qr)
        if len(pairs) == 0:
            continue
        all_p.append(pairs)
        all_w.append(np.full(len(pairs), WEIGHTS[ix.name], dtype=np.float32))
    if not all_p:
        return np.empty(0, np.uint64), np.empty(0, np.float32)
    pairs = np.concatenate(all_p)
    w = np.concatenate(all_w)
    del all_p, all_w
    # dedupe keeping max weight: lexsort by (pair, w) then take last per pair
    order = np.lexsort((w, pairs))
    pairs, w = pairs[order], w[order]
    keep = np.empty(len(pairs), dtype=bool)
    keep[-1] = True
    keep[:-1] = pairs[1:] != pairs[:-1]
    pairs, w = pairs[keep], w[keep]
    # same-country filter
    s1r, tr = split_pairs(pairs)
    same = s1c[s1r.astype(np.int64)] == tgc[tr.astype(np.int64)]
    return pairs[same], w[same]


def topm_per_s1(pairs, w, m):
    """Keep per-S1 top-m by (w desc, pair asc)."""
    if len(pairs) == 0 or m <= 0:
        return pairs
    s1r = (pairs >> np.uint64(32)).astype(np.int64)
    # sort by s1 asc, w desc, pair asc
    order = np.lexsort((pairs, -w, s1r))
    s1s = s1r[order]
    b = np.searchsorted(s1s, np.arange(int(s1r.max()) + 2), side="left")
    sizes = b[1:] - b[:-1]
    rank = np.arange(len(order)) - np.repeat(b[:-1], sizes)
    keep_sorted = rank < np.repeat(np.minimum(sizes, m), sizes)
    keep = np.zeros(len(pairs), dtype=bool)
    keep[order] = keep_sorted
    return pairs[keep]


def eval_combo(caps, topm=0, tag=""):
    idx = []
    for ix in loose:
        cap = caps.get(ix.name, "keep")
        idx.append(ix if cap in ("keep", None) else ix.with_cap(cap))
    parts, t0 = [], time.time()
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        p, w = block_chunk_weighted(idx, start, end)
        if topm:
            p = topm_per_s1(p, w, topm)
        parts.append(p)
    allp = np.unique(np.concatenate(parts))
    del parts
    s1r = (allp >> np.uint64(32)).astype(np.int64)
    allp = allp[s1r < N]
    hh = np.searchsorted(seg, allp)
    hh_c = np.minimum(hh, len(seg) - 1)
    covered = int((seg[hh_c] == allp).sum())
    rec = covered / max(seg_gt, 1)
    print(f"  [{tag}] topm={topm}: {len(allp):,} pairs ({len(allp)/N:.1f}/row) "
          f"recall {rec:.4f} missed {seg_gt-covered:,} ({time.time()-t0:.0f}s)",
          flush=True)
    if covered < seg_gt:  # which key types cover the missed pairs?
        pos = np.searchsorted(allp, seg)
        pos_c = np.minimum(pos, len(allp) - 1)
        miss = seg[allp[pos_c] != seg]
        bins = [0.5, 1.2, 1.75, 2.25, 3.0, 5.0, 8.5, 11.0]
        hist = np.zeros(len(bins) - 1, dtype=np.int64)
        for start in range(0, N, CHUNK):
            end = min(start + CHUNK, N)
            p, w = block_chunk_weighted(idx, start, end)
            if len(p) == 0:
                continue
            j = np.searchsorted(miss, p)
            j_c = np.minimum(j, len(miss) - 1)
            hitm = miss[j_c] == p
            if hitm.any():
                hist += np.histogram(w[hitm], bins=bins)[0]
            del p, w, j, j_c, hitm
        print("    missed-by-weight [postal,addr_bi,addr_tok,house*,name_tok,"
              "name_bi,sig/exact] =", hist.tolist(), flush=True)
    return allp, rec


C300 = {"name_tok": 300, "addr_tok": 300, "addr_bi": 300,
        "name_bi": 300, "house": 300, "house_norm": 300, "postal": None}
C1000 = {"name_tok": 1000, "addr_tok": 1000, "addr_bi": 1000,
         "name_bi": 1000, "house": 300, "house_norm": 300, "postal": None}
COMBOS = [
    ("tok300", C300, 0),
    ("tok300+topm48", C300, 48),
    ("tok300+topm32", C300, 32),
    ("tok1000", C1000, 0),
    ("tok1000+topm48", C1000, 48),
]

results = {}
for name, caps, topm in COMBOS:
    print(f"== {name} ==", flush=True)
    results[name] = eval_combo(caps, topm, tag=name)
print("done", flush=True)
