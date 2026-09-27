"""End-to-end orchestrator: prep -> blocking -> features -> LightGBM -> outputs.

Stages (all cached / restartable):
  1. prep      normalize train+test into compact memory-mappable caches
  2. block     build target-side key indexes (cached), emit candidate pairs
  3. features  pairwise fuzzy/structured features (parallel workers, disk memmap)
  4. cv        grouped-by-S1 5-fold CV -> honest OOF threshold
  5. fit       final LightGBM on all training candidates
  6. infer     score test candidates, apply threshold, write both TSVs

Only the supplied data is used; no external lookups of any kind.
"""
import argparse
import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np

import dataio
from blocking import (build_target_index, block_all_scored, split_pairs,
                      TfidfIndex)
from features import run_pairs, predict_chunked
from model import (load_gt, train_lgb, tune_threshold, blocking_recall,
                   fold_of, subsample_rows, gather_rows, predict_all_folds)

# DF caps per key type (None = keep all postings for that type)
CAPS = {"name_tok": 300, "addr_tok": 300, "addr_bi": 300, "name_bi": 300,
        "house": 300, "house_norm": 300, "postal": None}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------- blocking

def get_tf_index(cache, split, tgt, caps):
    """Loose target index (cached pkl) wrapped with caps + IDF weights."""
    f = Path(cache) / split / "index_loose.pkl"
    if f.exists():
        import pickle
        log(f"  loading loose index {f}")
        with open(f, "rb") as fh:
            loose = pickle.load(fh)
    else:
        log(f"  building loose target index for {split} ...")
        loose = build_target_index(tgt)
        import pickle
        f.parent.mkdir(parents=True, exist_ok=True)
        with open(f, "wb") as fh:
            pickle.dump(loose, fh, protocol=4)
        log(f"  loose index cached -> {f}")
    tf = []
    for ix in loose:
        cap = caps.get(ix.name, "keep")
        tf.append(TfidfIndex(ix) if cap in ("keep", None)
                  else TfidfIndex(ix.with_cap(cap)))
    return tf


def run_block(cache, split, force=False, chunk=100_000, topm=40):
    """Emit candidate pairs (uint64 npy) for a split; cached on disk."""
    out = Path(cache) / split / "pairs.npy"
    if out.exists() and not force:
        log(f"  pairs cached: {out} ({out.stat().st_size // 8:,} pairs)")
        return np.load(out, mmap_mode="r")
    s1 = dataio.load_role(cache, split, "s1", with_text=True)
    tgt = dataio.load_role(cache, split, "tgt", with_text=True)
    tf = get_tf_index(cache, split, tgt, CAPS)
    parts = []
    t0 = time.time()
    for start, end, pairs in block_all_scored(s1, tgt, tf, s1["country"],
                                              tgt["country"], chunk=chunk,
                                              topm=topm, n_s1=s1["n"]):
        parts.append(np.asarray(pairs))
        if sum(len(p) for p in parts) > 300_000_000 and len(parts) > 1:
            merged = np.unique(np.concatenate(parts))
            parts = [merged]
            gc.collect()
        if (start // chunk) % 5 == 0:
            sofar = sum(len(p) for p in parts)
            log(f"  {split}: rows {start:,}-{end:,} -> {sofar:,} pairs "
                f"({time.time() - t0:.0f}s)")
    allp = np.unique(np.concatenate(parts)) if parts else np.empty(0, np.uint64)
    del parts
    gc.collect()
    np.save(out, allp)
    log(f"  {split}: {len(allp):,} candidate pairs -> {out} ({time.time() - t0:.0f}s)")
    return np.load(out, mmap_mode="r")


# ------------------------------------------------------------------ top-k cap

def apply_topk(probs, pairs, n_s1, k):
    """Keep only the k highest-prob pairs per S1 entity (others set to -1)."""
    if len(pairs) == 0 or k <= 0:
        return probs
    s1r = split_pairs(pairs)[0].astype(np.int64)
    order = np.lexsort((probs, s1r))  # by s1, then prob ascending
    s1s = s1r[order]
    boundaries = np.searchsorted(s1s, np.arange(n_s1 + 1), side="left")
    sizes = boundaries[1:] - boundaries[:-1]
    rank = np.arange(len(order)) - np.repeat(boundaries[:-1], sizes)
    keep_sorted = rank >= np.repeat(sizes - k, sizes)
    keep = np.zeros(len(pairs), dtype=bool)
    keep[order] = keep_sorted
    out = probs.copy()
    out[~keep] = -1.0
    return out


def labels_for(pairs, gt_pairs):
    """y=1 if pair in gt (gt must be sorted unique); memory-light searchsorted."""
    if len(gt_pairs) == 0:
        return np.zeros(len(pairs), dtype=np.int8)
    idx = np.searchsorted(gt_pairs, pairs)
    idx_c = np.minimum(idx, len(gt_pairs) - 1)
    return (gt_pairs[idx_c] == pairs).astype(np.int8)


# ------------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    default_data = "dataset" if Path("dataset").exists() else "student_resource/dataset"
    ap.add_argument("--data", default=default_data)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="output")
    ap.add_argument("--jobs", type=int, default=11)
    ap.add_argument("--limit", type=int, default=None,
                    help="smoke-test: cap rows read per source file")
    ap.add_argument("--force-prep", action="store_true")
    ap.add_argument("--force-block", action="store_true")
    ap.add_argument("--force-features", action="store_true")
    ap.add_argument("--rounds", type=int, default=400)
    ap.add_argument("--lr", type=float, default=0.08)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--topk", type=int, default=10,
                    help="keep at most k highest-scored candidates per S1 entity")
    ap.add_argument("--skip-cv", action="store_true",
                    help="reuse cached threshold from a previous run")
    args = ap.parse_args()

    data = Path(args.data)
    cache = Path(args.cache)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tag = f"_lim{args.limit}" if args.limit else ""
    th_file = cache / f"threshold{tag}.json"
    params = {"learning_rate": args.lr}

    # ---------------- 1. prep ----------------
    log("== prep: train ==")
    dataio.prep(data, cache, "train", limit=args.limit, force=args.force_prep)
    log("== prep: test ==")
    dataio.prep(data, cache, "test", limit=args.limit, force=args.force_prep)

    # ---------------- 2. blocking ----------------
    log("== blocking: train ==")
    tr_pairs = run_block(cache, "train", force=args.force_block)
    log("== blocking: test ==")
    te_pairs = run_block(cache, "test", force=args.force_block)

    # ---------------- 3. training features + GT ----------------
    x_file = Path(cache) / "train" / "X.npy"
    need_x = (args.force_features or not x_file.exists()
              or x_file.stat().st_mtime < (Path(cache) / "train" / "pairs.npy").stat().st_mtime)
    if need_x:
        log("== features: train (-> memmap) ==")
        X_tr = run_pairs(tr_pairs, cache, "train", model_path=None, jobs=args.jobs,
                         out_path=str(x_file))
    else:
        log(f"== features: train cached ({x_file}) ==")
        X_tr = np.load(x_file, mmap_mode="r")
    s1_ids_tr = dataio.load_role(cache, "train", "s1", with_text=False)["ids"]
    n_s1_tr = len(s1_ids_tr)
    tgt_ids_tr = dataio.load_role(cache, "train", "tgt", with_text=False)["ids"]
    gt_pairs, gt_counts = load_gt(data / "train/train_ground_truth.tsv",
                                  s1_ids_tr, tgt_ids_tr)
    y_tr = labels_for(tr_pairs, gt_pairs)
    pos_rate = y_tr.mean() if len(y_tr) else 0.0
    rec = blocking_recall(tr_pairs, gt_pairs)
    log(f"train pairs {len(tr_pairs):,} pos {int(y_tr.sum()):,} ({pos_rate:.1%}) "
        f"blocking recall {rec:.4f}")

    # ---------------- 4. holdout threshold (fast, honest) ----------------
    # 5% of S1 entities (hash bucket 0 of 20) are held out: the model is
    # trained with their rows zero-weighted, then the decision threshold is
    # tuned on exact per-entity F0.5 over those entities' FULL candidate set.
    s1r_tr = split_pairs(tr_pairs)[0].astype(np.int64)
    ho_entity = fold_of(np.arange(n_s1_tr, dtype=np.uint64), 20) == 0
    ho_pair = ho_entity[s1r_tr]
    sub_idx = subsample_rows(y_tr, neg_stride=3)
    X_sub = gather_rows(X_tr, sub_idx)
    y_sub = y_tr[sub_idx]
    w_sub = np.where(ho_pair[sub_idx], 0.0, 1.0).astype(np.float32)
    log(f"holdout entities: {int(ho_entity.sum()):,} "
        f"({int(ho_pair.sum()):,} candidate rows excluded from training)")

    if args.skip_cv and th_file.exists():
        best_t = json.loads(th_file.read_text())["threshold"]
        ho_score = json.loads(th_file.read_text()).get("holdout_f05", -1.0)
        log(f"== threshold {best_t:.3f} reused from {th_file}")
    else:
        log("== holdout: train (holdout zero-weighted) ==")
        booster_ho = train_lgb(X_sub, y_sub, params=params, rounds=args.rounds,
                               weight=w_sub)
        del w_sub
        gc.collect()
        log("== holdout: predict holdout rows ==")
        ho_idx = np.flatnonzero(ho_pair)
        X_ho = gather_rows(X_tr, ho_idx)
        ho_probs = predict_chunked(booster_ho, X_ho, chunk=2_000_000)
        del X_ho, booster_ho
        gc.collect()
        best_t, ho_score, curve = tune_threshold(
            ho_probs, tr_pairs[ho_idx], gt_pairs, n_s1_tr, rows=ho_entity)
        log(f"holdout best F0.5={ho_score:.4f} @ t={best_t:.3f}")
        th_file.write_text(json.dumps({"threshold": best_t,
                                       "holdout_f05": ho_score}))
        top = sorted(curve, key=lambda x: -x[1])[:5]
        log("top thresholds: " + ", ".join(f"{t:.3f}:{s:.4f}" for t, s in top))
        del ho_probs, ho_idx
        gc.collect()

    # ---------------- 5. final fit on the subsample ----------------
    log("== fit: final model on train subsample ==")
    booster = train_lgb(X_sub, y_sub, params=params, rounds=args.rounds)
    mpath = cache / f"model{tag}.txt"
    booster.save_model(str(mpath))
    log(f"model saved -> {mpath}")
    del X_sub, y_sub, X_tr, y_tr, tr_pairs
    gc.collect()

    # ---------------- 6. inference ----------------
    log("== infer: test ==")
    te_probs = run_pairs(te_pairs, cache, "test", model_path=str(mpath), jobs=args.jobs)
    s1_te = dataio.load_role(cache, "test", "s1", with_text=False)
    tgt_te = dataio.load_role(cache, "test", "tgt", with_text=False)
    n_s1_te = s1_te["n"]
    te_probs = apply_topk(te_probs, te_pairs, n_s1_te, args.topk)
    keep = te_probs >= best_t

    s1_ids = [s.decode() for s in np.asarray(s1_te["ids"])]
    id_arr = np.asarray([s.decode() for s in np.asarray(tgt_te["ids"])], dtype=object)

    s1r, tr_ = split_pairs(te_pairs)
    s1r = s1r.astype(np.int64)
    order = np.argsort(s1r, kind="stable")
    s1_sorted = s1r[order]
    bounds = np.searchsorted(s1_sorted, np.arange(n_s1_te + 1))
    del s1_sorted
    gc.collect()

    m_lines = ["source1_entity_id\tmatched_entity_ids"]
    c_lines = ["source1_entity_id\tcandidate_entity_ids"]
    for i in range(n_s1_te):
        a, b = bounds[i], bounds[i + 1]
        idxs = order[a:b]
        mids = id_arr[tr_[idxs[keep[idxs]]]]
        cids = id_arr[tr_[idxs]]
        m_lines.append(s1_ids[i] + "\t" + ",".join(mids.tolist()))
        c_lines.append(s1_ids[i] + "\t" + ",".join(cids.tolist()))
        if i % 200_000 == 0 and i:
            log(f"  wrote {i:,}/{n_s1_te:,} rows")
    (out / "matching_results.tsv").write_text("\n".join(m_lines) + "\n", encoding="utf-8")
    (out / "candidate_pairs.tsv").write_text("\n".join(c_lines) + "\n", encoding="utf-8")
    Path("matching_results.tsv").write_text("\n".join(m_lines) + "\n", encoding="utf-8")
    Path("candidate_pairs.tsv").write_text("\n".join(c_lines) + "\n", encoding="utf-8")
    log(f"wrote {out / 'matching_results.tsv'} ({len(m_lines) - 1:,} rows) and "
        f"{out / 'candidate_pairs.tsv'}")
    log("copied submission files to root directory (matching_results.tsv and candidate_pairs.tsv)")
    log(f"final threshold: {best_t:.3f} (from {th_file})")


if __name__ == "__main__":
    main()
