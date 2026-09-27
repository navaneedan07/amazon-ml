"""Debug pair scoring on 100k train rows.

Computes summed-IDF scores for all cap-300 candidates of rows 0-100k, then:
  - recall@M curves by score rank (GT vs random baseline)
  - weight distribution GT-in-candidates vs non-GT
  - top-3 scored pairs for a few entities with names printed

Usage: python analysis/debug_score.py
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import dataio
from blocking import TfidfIndex, _sum_weights, split_pairs

cache = ROOT / "cache"
N = 100_000

s1 = dataio.load_role(str(cache), "train", "s1", with_text=True)
tgt = dataio.load_role(str(cache), "train", "tgt", with_text=True)
with open(cache / "train" / "index_loose.pkl", "rb") as fh:
    loose = pickle.load(fh)
knames = [ix.name for ix in loose]
print("loose:", {ix.name: len(ix) for ix in loose}, flush=True)

CAPS = {"name_tok": 300, "addr_tok": 300, "addr_bi": 300, "name_bi": 300,
        "house": 300, "house_norm": 300, "postal": None}
tfs = {}
for ix in loose:
    cap = CAPS.get(ix.name, "keep")
    tfs[ix.name] = TfidfIndex(ix) if cap in ("keep", None) else TfidfIndex(ix.with_cap(cap))

gt = np.load(cache / "train" / "gt_pairs.npy")
hi = int(np.searchsorted(gt, (np.uint64(N) << np.uint64(32)) - np.uint64(1),
                         side="right"))
seg = gt[:hi]
seg = seg[(seg >> np.uint64(32)) < N]
print(f"GT pairs rows<{N}: {len(seg):,}", flush=True)

s1c, tgc = s1["country"], tgt["country"]

# per-key-type probes for rows 0..N
from blocking import query_keys
all_p, all_w = [], []
t0 = time.time()
for start in range(0, N, 50_000):
    end = min(start + N, start + 50_000)
    qkeys = query_keys(s1, start, end, [type("K", (), {"name": n})() for n in knames])
    for ix, qh, qr in qkeys:
        if len(qh) == 0:
            continue
        p, w = tfs[ix.name].probe_scored(qh, qr)
        if len(p):
            all_p.append(p)
            all_w.append(w)
pairs = np.concatenate(all_p)
w = np.concatenate(all_w)
del all_p, all_w
print(f"raw postings: {len(pairs):,} ({time.time()-t0:.0f}s)", flush=True)

pairs, w = _sum_weights(pairs, w)
print(f"unique pairs: {len(pairs):,}", flush=True)
s1r, tr = split_pairs(pairs)
same = s1c[s1r.astype(np.int64)] == tgc[tr.astype(np.int64)]
pairs, w = pairs[same], w[same]
s1r = s1r[same]
print(f"same-country pairs: {len(pairs):,}", flush=True)

# GT membership
pos = np.isin(pairs, seg)
print(f"GT-in-cands: {pos.sum():,} / {len(seg):,}", flush=True)
print(f"weight stats: GT mean {w[pos].mean():.1f} med {np.median(w[pos]):.1f} | "
      f"nonGT mean {w[~pos].mean():.1f} med {np.median(w[~pos]):.1f}", flush=True)

# recall@M by per-entity score rank
order = np.lexsort((w, s1r))
s1s, ws, ps = s1r[order], w[order], pos[order]
b = np.searchsorted(s1s, np.arange(N + 1), side="left")
sizes = b[1:] - b[:-1]
rank = np.arange(len(order), dtype=np.int64) - np.repeat(b[:-1], sizes)
gt_ranks = rank[ps]
for M in (5, 10, 20, 40, 80):
    rec = (gt_ranks < M).mean()
    base = np.minimum(sizes, M).sum() / max(len(pairs), 1)
    print(f"recall@{M}: {rec:.4f} (random-in-cands baseline {base:.4f})", flush=True)

# top-3 pairs for 3 sample entities
names_t = tgt["names"]
names_s = s1["names"]
addrs_t, addrs_s = tgt["addrs"], s1["addrs"]
rng = np.random.default_rng(0)
sample_entities = rng.choice(np.flatnonzero(sizes > 10), 4, replace=False)
for e in sample_entities:
    a, bb = b[e], b[e + 1]
    idxs = order[a:bb][np.argsort(-ws[a:bb])][:3]
    print(f"\n== S1 row {e}: {names_s.get(e)!r} | {addrs_s.get(e)!r}", flush=True)
    gt_rows = seg[seg >> np.uint64(32) == np.uint64(e)]
    gt_t = (gt_rows & np.uint64(0xFFFFFFFF)).astype(np.int64)
    print(f"   GT targets: {gt_t.tolist()}", flush=True)
    for j in idxs:
        t = int(tr[j])
        mark = " <GT>" if pos[j] else ""
        print(f"   w={w[j]:7.1f} t={t}{mark}: {names_t.get(t)!r} | {addrs_t.get(t)!r}",
              flush=True)
print("done", flush=True)
