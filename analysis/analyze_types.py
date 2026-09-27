"""Per-key-type blocking stats on 500k train rows.

For each key type and cap, measures SOLO volume (pairs/row) and SOLO recall
over rows 0-N. Tells us the marginal value of each key type so we can pick
a union with recall >= 0.97 at minimal volume.

Usage: python analysis/analyze_types.py [N_ROWS]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import dataio
from blocking import query_keys

cache = ROOT / "cache"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000
CHUNK = 100_000

s1 = dataio.load_role(str(cache), "train", "s1", with_text=True)
tgt = dataio.load_role(str(cache), "train", "tgt", with_text=True)
with open(cache / "train" / "index_loose.pkl", "rb") as fh:
    loose = pickle.load(fh)

gt = np.load(cache / "train" / "gt_pairs.npy")
lo = int(np.searchsorted(gt, np.uint64(0), side="left"))
hi = int(np.searchsorted(gt, (np.uint64(N) << np.uint64(32)) - np.uint64(1),
                         side="right"))
seg = gt[lo:hi]
seg_gt = int(((seg >> np.uint64(32)) < N).sum())
print(f"rows 0-{N:,}: {seg_gt:,} GT pairs", flush=True)

s1c, tgc = s1["country"], tgt["country"]
qcache = {}
for start in range(0, N, CHUNK):
    end = min(start + CHUNK, N)
    qcache[start] = query_keys(s1, start, end, loose)


def solo(ix, cap):
    if cap is not None:
        ix = ix.with_cap(cap)
    tot, t0 = 0, time.time()
    cov = np.zeros(len(seg), dtype=bool)
    for start in range(0, N, CHUNK):
        end = min(start + CHUNK, N)
        for jx, qh, qr in qcache[start]:
            if jx.name != ix.name or len(qh) == 0:
                continue
            pairs = ix.probe(qh, qr)
            tot += len(pairs)
            if len(pairs):
                s1r = (pairs >> np.uint64(32)).astype(np.int64)
                tr = (pairs & np.uint64(0xFFFFFFFF)).astype(np.int64)
                ok = s1c[s1r] == tgc[tr]
                pairs = pairs[ok]
                if len(pairs):
                    j = np.searchsorted(seg, pairs)
                    j_c = np.minimum(j, len(seg) - 1)
                    hit = seg[j_c] == pairs
                    cov[j_c[hit]] = True
    rec = cov.sum() / seg_gt
    print(f"  {ix.name:12s} cap={cap}: {tot/N:8.1f}/row solo-recall {rec:.4f} "
          f"({time.time()-t0:.0f}s)", flush=True)
    return rec


for ix in loose:
    if ix.name in ("name_exact", "name_sig", "name_nospace"):
        continue
    for cap in (50, 150, 300):
        solo(ix, cap)
print("done", flush=True)
