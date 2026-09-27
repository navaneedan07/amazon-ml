"""Tune TF-IDF-weighted blocking + per-S1 top-M on 500k train rows.

Pipeline per combo:
  loose index -> cap-filter per key type -> TfidfIndex
  probe all key types, dedupe by max summed-IDF, same-country,
  per-S1 top-M -> volume + recall report.

Usage: python analysis/tune_tfidf.py [N_ROWS]
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import dataio
from blocking import TfidfIndex, block_all_scored

cache = ROOT / "cache"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 500_000

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


def build_tf(caps):
    tf = []
    for ix in loose:
        cap = caps.get(ix.name, "keep")
        tf.append(TfidfIndex(ix if cap in ("keep", None) else ix.with_cap(cap)))
    return tf


def eval_combo(caps, topm, tag):
    tf = build_tf(caps)
    t0 = time.time()
    parts = []
    for start, end, pairs in block_all_scored(s1, tgt, tf, s1["country"],
                                              tgt["country"], chunk=100_000,
                                              topm=topm, n_s1=N):
        parts.append(pairs)
    allp = np.unique(np.concatenate(parts))
    del parts
    s1r = (allp >> np.uint64(32)).astype(np.int64)
    allp = allp[s1r < N]
    j = np.searchsorted(seg, allp)
    j_c = np.minimum(j, len(seg) - 1)
    covered = int((seg[j_c] == allp).sum())
    rec = covered / seg_gt
    print(f"  [{tag}] topm={topm}: {len(allp):,} pairs ({len(allp)/N:.1f}/row) "
          f"recall {rec:.4f} missed {seg_gt - covered:,} ({time.time()-t0:.0f}s)",
          flush=True)
    return rec


BASE = {"name_tok": 300, "addr_tok": 300, "addr_bi": 300, "name_bi": 300,
        "house": 300, "house_norm": 300, "postal": None}
B1000 = dict(BASE, name_tok=1000, addr_tok=1000, addr_bi=1000, name_bi=1000)

for tag, caps, tm in (
    ("c300", BASE, 12),
    ("c300", BASE, 24),
    ("c300", BASE, 40),
    ("c300", BASE, 64),
    ("c1000", B1000, 40),
):
    eval_combo(caps, tm, tag)
print("done", flush=True)
