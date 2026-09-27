"""Pairwise features + parallel feature/predict execution.

Features are computed per candidate pair from the normalized fields only.
Workers memory-map the prepared blobs, so pages are shared across processes.

Two modes:
  * features mode  -> returns the float32 feature matrix (training)
  * predict mode   -> returns float32 match probabilities (workers load the
                      saved LightGBM booster once at init)
"""
import numpy as np
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler as _JaroWinkler

from blocking import split_pairs
from textnorm import name_sig

FEATURE_NAMES = [
    "name_ratio", "name_tsr", "name_tsort", "name_partial", "name_jacc",
    "name_exact", "name_len_ratio", "name_jw", "name_sig_jacc", "name_sig_exact",
    "addr_ratio", "addr_tsr", "addr_tsort", "addr_jacc", "addr_exact",
    "addr_len_ratio",
    "house_match", "postal_match", "script_mismatch", "same_source",
]
N_FEATURES = len(FEATURE_NAMES)

_W = {}


def _sim(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.ratio(a, b) * 0.01


def _tsr(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.token_set_ratio(a, b) * 0.01


def _tsort(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.token_sort_ratio(a, b) * 0.01


def _partial(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return fuzz.partial_ratio(a, b) * 0.01


def _jw(a, b):
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return _JaroWinkler.similarity(a, b) * 0.01


def _jacc(a, b):
    ta, tb = set(a.split()), set(b.split())
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _len_ratio(a, b):
    la, lb = len(a), len(b)
    if la == 0 and lb == 0:
        return 1.0
    if la == 0 or lb == 0:
        return 0.0
    return min(la, lb) / max(la, lb)


def pair_features(n1, n2, a1, a2, h1, h2, z1, z2, nn1, nn2, an1, an2, ss):
    """Feature vector for one pair (all inputs normalized strings / flags)."""
    sig1, sig2 = name_sig(n1), name_sig(n2)
    return (
        # name block
        _sim(n1, n2),
        _tsr(n1, n2),
        _tsort(n1, n2),
        _partial(n1, n2),
        _jacc(n1, n2),
        float(n1 == n2 and n1 != ""),
        _len_ratio(n1, n2),
        _jw(n1, n2),
        _jacc(sig1, sig2),
        float(sig1 == sig2 and sig1 != ""),
        # address block
        _sim(a1, a2),
        _tsr(a1, a2),
        _tsort(a1, a2),
        _jacc(a1, a2),
        float(a1 == a2 and a1 != ""),
        _len_ratio(a1, a2),
        # structured components
        float(bool(h1) and h1 == h2),
        float(bool(z1) and z1 == z2),
        float(nn1 != nn2 or an1 != an2),
        float(ss),
    )


def worker_init(cache, split, model_path=None):
    from pathlib import Path

    from dataio import load_role
    _W["s1"] = load_role(cache, split, "s1")
    _W["tgt"] = load_role(cache, split, "tgt")
    n_s2 = None
    p = Path(cache) / split / "tgt_n2.txt"
    if p.exists():
        n_s2 = int(p.read_text().strip())
    _W["n_s2"] = n_s2
    _W["model"] = None
    if model_path:
        import lightgbm as lgb
        _W["model"] = lgb.Booster(model_file=model_path)


def worker_run(pairs):
    """pairs: uint64 array. Returns features (float32) or probabilities."""
    s1, tgt = _W["s1"], _W["tgt"]
    s1r, tr = split_pairs(pairs)
    n = len(pairs)
    X = np.empty((n, N_FEATURES), dtype=np.float32)
    names1, addrs1 = s1["names"], s1["addrs"]
    names2, addrs2 = tgt["names"], tgt["addrs"]
    house1, house2 = s1["house"], tgt["house"]
    zip1, zip2 = s1["zip"], tgt["zip"]
    nn1a, nn2a = s1["nn"], tgt["nn"]
    an1a, an2a = s1["an"], tgt["an"]
    n_s2 = _W["n_s2"]
    for i in range(n):
        i1 = int(s1r[i])
        i2 = int(tr[i])
        ss = (i2 >= n_s2) if n_s2 is not None else False
        X[i] = pair_features(
            names1.get(i1), names2.get(i2),
            addrs1.get(i1), addrs2.get(i2),
            bytes(house1[i1]), bytes(house2[i2]),
            bytes(zip1[i1]), bytes(zip2[i2]),
            bool(nn1a[i1]), bool(nn2a[i2]),
            bool(an1a[i1]), bool(an2a[i2]),
            ss,
        )
    if _W["model"] is not None:
        return _W["model"].predict(X)
    return X


def run_pairs(pairs, cache, split, model_path=None, jobs=11, chunk=100_000,
              out_path=None):
    """Compute features (model_path=None) or probabilities for uint64 pairs.

    Feature mode writes the matrix to `out_path` as a disk memmap (too big for
    RAM at full scale); predict mode keeps probabilities in RAM.
    """
    import multiprocessing as mp
    from numpy.lib.format import open_memmap

    n = len(pairs)
    if n == 0:
        return (np.empty((0, N_FEATURES), dtype=np.float32) if model_path is None
                else np.empty(0, dtype=np.float32))
    if model_path is None and out_path:
        out = open_memmap(out_path, mode="w+", dtype=np.float32,
                          shape=(n, N_FEATURES))
    else:
        out = (np.empty((n, N_FEATURES), dtype=np.float32) if model_path is None
               else np.empty(n, dtype=np.float32))
    ctx = mp.get_context("spawn")
    chunks = [(i, min(i + chunk, n)) for i in range(0, n, chunk)]
    with ctx.Pool(jobs, initializer=worker_init,
                  initargs=(cache, split, model_path)) as pool:
        done = 0
        for (a, b), res in zip(chunks, pool.imap(worker_run, [pairs[a:b] for a, b in chunks])):
            out[a:b] = res
            done += 1
            if done % 10 == 0 or done == len(chunks):
                print(f"    {done}/{len(chunks)} chunks ({b}/{n} pairs)", flush=True)
    if isinstance(out, np.memmap):
        out.flush()
    return out


def predict_chunked(booster, X, chunk=2_000_000):
    """Predict probabilities over a (possibly memmapped) matrix in chunks."""
    n = len(X)
    out = np.empty(n, dtype=np.float32)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        out[a:b] = booster.predict(np.ascontiguousarray(X[a:b]))
        if (a // chunk) % 20 == 0:
            print(f"    predict {b}/{n}", flush=True)
    return out
