"""Validate upgraded blocking on the first 500k train S1 rows."""
import numpy as np
import sys
import time

sys.path.insert(0, "code/business_entity_resolution/src")
import dataio
from blocking import build_target_index, block_all

s1 = dataio.load_role("cache", "train", "s1", with_text=True)
tgt = dataio.load_role("cache", "train", "tgt", with_text=True)
t0 = time.time()
idx = build_target_index(tgt)
print(f"index built in {time.time()-t0:.0f}s", flush=True)

gt = np.load("cache/train/gt_pairs.npy")
t0 = time.time()
parts = []
N = 500_000
gt_lo = np.searchsorted(gt, (np.arange(N, dtype=np.uint64) << np.uint64(32)))
gt_hi = np.searchsorted(
    gt, ((np.arange(N, dtype=np.uint64) + 1) << np.uint64(32)) - np.uint64(1),
    side="right")
for start, end, pairs in block_all(s1, tgt, idx, s1["country"], tgt["country"],
                                   chunk=100_000):
    if start >= N:
        break
    parts.append(pairs)
    print(f"  chunk {start}: +{len(pairs):,} ({time.time()-t0:.0f}s)", flush=True)
allp = np.unique(np.concatenate(parts))
lo, hi = gt_lo[0], gt_hi[N - 1]
seg = gt[lo:hi]
hh = np.searchsorted(seg, allp)
hh_c = np.minimum(hh, len(seg) - 1)
inseg = seg[hh_c] == allp
s1r = (allp >> np.uint64(32)).astype(np.int64)
allp = allp[inseg & (s1r < N)]
hh = np.searchsorted(seg, allp)
hh_c = np.minimum(hh, len(seg) - 1)
covered = int((seg[hh_c] == allp).sum())
seg_gt = int(((seg >> np.uint64(32)) < N).sum())
print(f"rows 0-{N}: {len(allp):,} pairs in {time.time()-t0:.0f}s", flush=True)
print(f"GT in range: {seg_gt:,}  covered: {covered:,}  "
      f"local recall: {covered/max(seg_gt,1):.4f}", flush=True)
print(f"pairs per s1 row: {len(allp)/N:.1f}", flush=True)
