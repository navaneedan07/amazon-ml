"""Model training, macro-F0.5 evaluation and threshold tuning.

Full-scale memory strategy: the feature matrix lives on disk as a float32
memmap. LightGBM reads it directly (``two_round``); CV folds are implemented
with zero weights on non-fold rows instead of materializing row subsets.
"""
import gc

import numpy as np
import pandas as pd

from blocking import split_pairs


def _id_index(arr):
    """pd.Index of str ids from a fixed-width bytes array (or str array)."""
    a = np.asarray(arr)
    if a.dtype.kind == "S":
        a = np.char.decode(a, "utf-8")
    return pd.Index(a)


# ---------------------------------------------------------------- ground truth

def load_gt(gt_path, s1_ids, tgt_ids):
    """Explode the ground-truth file into combined uint64 pairs.

    Returns (gt_pairs sorted uint64, gt_count per s1 row).
    Unmapped ids (shouldn't happen) are dropped.
    """
    gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    s1_idx = _id_index(s1_ids)
    tgt_idx = _id_index(tgt_ids)
    s1_row = s1_idx.get_indexer(gt["source1_entity_id"])

    split = gt["matched_entity_ids"].map(lambda s: [x for x in s.split(",") if x])
    lens = split.str.len().to_numpy()
    mids = np.fromiter(
        (m for lst in split for m in lst), dtype=object, count=int(lens.sum())
    )
    s1_exp = np.repeat(s1_row, lens)
    tgt_row = tgt_idx.get_indexer(mids)

    ok = (s1_exp >= 0) & (tgt_row >= 0)
    s1_exp, tgt_row = s1_exp[ok], tgt_row[ok]
    pairs = np.unique((s1_exp.astype(np.uint64) << np.uint64(32))
                      | tgt_row.astype(np.uint64))
    counts = np.bincount(s1_exp.astype(np.int64), minlength=len(s1_ids)).astype(np.int64)
    print(f"  GT: {len(pairs)} pairs, {len(s1_ids)} S1 entities, "
          f"{int((counts > 0).sum())} with >=1 match", flush=True)
    return pairs, counts


# ------------------------------------------------------------------- training

def train_lgb(X, y, params=None, rounds=600, seed=42, weight=None):
    """Train a LightGBM binary model (MIT-licensed).

    `X` may be a disk memmap; two_round keeps peak memory low.
    """
    import lightgbm as lgb

    p = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 100,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "two_round": True,
        "verbose": -1,
        "num_threads": 11,
        "seed": seed,
    }
    if params:
        p.update(params)
    dtrain = lgb.Dataset(X, label=y, weight=weight)
    booster = lgb.train(p, dtrain, num_boost_round=rounds,
                        callbacks=[lgb.log_evaluation(0)])
    return booster


def fold_of(s1r, n_splits):
    """Deterministic per-S1-entity fold ids in [0, n_splits)."""
    h = s1r.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15)
    return ((h >> np.uint64(33)) % np.uint64(n_splits)).astype(np.int8)


def subsample_rows(y, neg_stride=3):
    """Deterministic training subsample: all positives + every neg_stride-th
    negative (row-index modulo). Returns int32 row indices, chunk-computed."""
    n = len(y)
    keep = []
    BLOCK = 10_000_000
    for a in range(0, n, BLOCK):
        b = min(a + BLOCK, n)
        rows = np.arange(a, b, dtype=np.int64)
        m = (y[a:b] == 1) | (rows % neg_stride == 0)
        keep.append(rows[m].astype(np.int64))
    idx = np.concatenate(keep) if keep else np.empty(0, np.int64)
    del keep
    print(f"  subsample: {len(idx):,} rows "
          f"({int((y[idx] == 1).sum()):,} pos)", flush=True)
    return idx


def gather_rows(X_mmap, idx, block=2_000_000):
    """Materialize X_mmap[idx] into a fresh RAM matrix, chunk-wise."""
    X = np.empty((len(idx), X_mmap.shape[1]), dtype=np.float32)
    for a in range(0, len(idx), block):
        b = min(a + block, len(idx))
        X[a:b] = X_mmap[idx[a:b]]
    return X


def predict_all_folds(X_sub, y_sub, fold_sub, X_full, fold_full, n_splits,
                      rounds=400, params=None, chunk=2_000_000):
    """Grouped CV with honest OOF probs over ALL full-set rows.

    Each fold trains on the in-RAM subsample (zero-weight rows for its own
    fold) and predicts over the full feature memmap; only the fold's rows are
    kept in the OOF vector. Returns oof_full.
    """
    n_full = len(X_full)
    oof = np.zeros(n_full, dtype=np.float32)
    for f in range(n_splits):
        weight = np.where(fold_sub == f, 0.0, 1.0).astype(np.float32)
        n_tr = int((weight > 0).sum())
        print(f"  fold {f + 1}/{n_splits}: train={n_tr:,} "
              f"val(full)={int((fold_full == f).sum()):,}", flush=True)
        booster = train_lgb(X_sub, y_sub, params=params, rounds=rounds,
                            seed=42 + f, weight=weight)
        preds = np.empty(n_full, dtype=np.float32)
        for a in range(0, n_full, chunk):
            b = min(a + chunk, n_full)
            preds[a:b] = booster.predict(np.ascontiguousarray(X_full[a:b]))
        va = fold_full == f
        oof[va] = preds[va]
        del preds, booster, weight
        gc.collect()
    return oof


# --------------------------------------------------------------- evaluation

def per_entity_f05(pred_tp, pred_fp, gt_count):
    """Macro F0.5 over entities given per-entity counts (arrays).

    F0.5 = 1.25*P*R / (0.25*P + R); entities with no prediction and no truth
    score 1.0, entities with no correct prediction score 0.0.
    """
    tp = np.asarray(pred_tp, dtype=np.float64)
    fp = np.asarray(pred_fp, dtype=np.float64)
    fn = np.maximum(np.asarray(gt_count, dtype=np.float64) - tp, 0.0)
    n_p = tp + fp
    n_t = tp + fn
    prec = np.divide(tp, n_p, out=np.zeros_like(tp), where=n_p > 0)
    rec = np.divide(tp, n_t, out=np.zeros_like(tp), where=n_t > 0)
    den = 0.25 * prec + rec
    f = np.divide(1.25 * prec * rec, den, out=np.zeros_like(tp), where=den > 0)
    f = np.where((n_p == 0) & (n_t == 0), 1.0, f)
    return float(f.mean())


def eval_pairs(probs, pairs, gt_pairs, n_s1, threshold, rows=None):
    """Macro F0.5 for one threshold; optionally restricted to `rows` entities."""
    s1r = split_pairs(pairs)[0].astype(np.int64)
    pos = np.isin(pairs, gt_pairs)
    mask = probs >= threshold
    tp = np.bincount(s1r[mask & pos], minlength=n_s1)
    fp = np.bincount(s1r[mask & ~pos], minlength=n_s1)
    gt_count = np.bincount(split_pairs(gt_pairs)[0].astype(np.int64), minlength=n_s1)
    if rows is not None:
        return per_entity_f05(tp[rows], fp[rows], gt_count[rows])
    return per_entity_f05(tp, fp, gt_count)


def tune_threshold(probs, pairs, gt_pairs, n_s1, rows=None, grid=None):
    """Sweep thresholds, return (best_threshold, best_score, curve)."""
    if grid is None:
        grid = np.arange(0.05, 0.951, 0.025)
    gt_count = np.bincount(split_pairs(gt_pairs)[0].astype(np.int64), minlength=n_s1)
    s1r = split_pairs(pairs)[0].astype(np.int64)
    pos = np.isin(pairs, gt_pairs)
    if rows is not None:
        in_rows = np.zeros(n_s1, dtype=bool)
        in_rows[rows] = True
        pair_ok = in_rows[s1r]
        s1r, pos, probs = s1r[pair_ok], pos[pair_ok], probs[pair_ok]
    best = (-1.0, 0.5)
    curve = []
    for t in grid:
        mask = probs >= t
        tp = np.bincount(s1r[mask & pos], minlength=n_s1)
        fp = np.bincount(s1r[mask & ~pos], minlength=n_s1)
        if rows is not None:
            score = per_entity_f05(tp[rows], fp[rows], gt_count[rows])
        else:
            score = per_entity_f05(tp, fp, gt_count)
        curve.append((float(t), score))
        if score > best[1]:
            best = (float(t), score)
    return best[0], best[1], curve


def blocking_recall(pairs, gt_pairs):
    """Fraction of ground-truth pairs present in the candidate set."""
    if len(gt_pairs) == 0:
        return 1.0
    return float(np.isin(gt_pairs, pairs).mean())
