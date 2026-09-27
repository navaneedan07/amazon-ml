"""Build and cache the LOOSE target-side blocking index for a split.

Loose = token/bigram/house/postal keys keep ALL postings (cap=None); DF caps
are applied afterwards via KeyIndex.with_cap() so caps can be swept cheaply.

Usage: python analysis/build_loose_index.py train|test
Writes cache/<split>/index_loose.pkl
"""
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code" / "business_entity_resolution" / "src"))

import dataio
from blocking import build_target_index

split = sys.argv[1] if len(sys.argv) > 1 else "train"
cache = ROOT / "cache"
out = cache / split / "index_loose.pkl"
if out.exists():
    print(f"exists: {out}", flush=True)
    sys.exit(0)

tgt = dataio.load_role(str(cache), split, "tgt", with_text=True)
t0 = time.time()
idx = build_target_index(tgt)
print(f"index built in {time.time() - t0:.0f}s", flush=True)
with open(out, "wb") as fh:
    pickle.dump(idx, fh, protocol=4)
print(f"cached -> {out}", flush=True)
