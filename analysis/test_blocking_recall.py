import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, "code/business_entity_resolution/src")
import dataio
import model
from blocking import build_target_index, TfidfIndex, query_keys, _sum_weights, topm_pairs, split_pairs

def eval_multistage(n_eval=200_000):
    cache = "cache"
    s1 = dataio.load_role(cache, "train", "s1", with_text=True)
    tgt = dataio.load_role(cache, "train", "tgt", with_text=True)
    
    with open("cache/train/index_loose.pkl", "rb") as fh:
        import pickle
        loose = pickle.load(fh)
    
    CAPS = {"name_tok": 300, "addr_tok": 300, "addr_bi": 300, "name_bi": 300,
            "house": 300, "house_norm": 300, "postal": None}
    
    tf_exact, tf_tokens = [], []
    for ix in loose:
        cap = CAPS.get(ix.name, "keep")
        tfi = TfidfIndex(ix) if cap in ("keep", None) else TfidfIndex(ix.with_cap(cap))
        if ix.name in ("name_exact", "name_sig", "name_nospace"):
            tf_exact.append(tfi)
        else:
            tf_tokens.append(tfi)
            
    gt_pairs, _ = model.load_gt("student_resource/dataset/train/train_ground_truth.tsv",
                                s1["ids"], tgt["ids"])
    
    # Filter GT for evaluated rows
    gt_s1, _ = split_pairs(gt_pairs)
    eval_gt_mask = (gt_s1 < n_eval)
    eval_gt = gt_pairs[eval_gt_mask]
    print(f"GT pairs in first {n_eval:,} rows: {len(eval_gt):,}")

    exact_names = {ix.name for ix in tf_exact}
    token_names = {ix.name for ix in tf_tokens}
    all_by_name = {ix.name: ix for ix in tf_exact + tf_tokens}
    q_all = query_keys(s1, 0, n_eval, [type("K", (), {"name": n})() for n in all_by_name])
    
    all_parts = []
    p_exact_list = []
    p_tok_list, w_tok_list = [], []
    
    for ix, qh, qr in q_all:
        if len(qh) == 0:
            continue
        if ix.name in exact_names:
            p, _ = all_by_name[ix.name].probe_scored(qh, qr)
            if len(p):
                p_exact_list.append(p)
        else:
            p, w = all_by_name[ix.name].probe_scored(qh, qr)
            if len(p):
                p_tok_list.append(p)
                w_tok_list.append(w)
                
    if p_exact_list:
        p_exact = np.unique(np.concatenate(p_exact_list))
        s1r, tr = split_pairs(p_exact)
        same = s1["country"][s1r.astype(np.int64)] == tgt["country"][tr.astype(np.int64)]
        p_exact = p_exact[same]
        all_parts.append(p_exact)
        print(f"Exact keys pairs: {len(p_exact):,}, recall: {model.blocking_recall(p_exact, eval_gt):.4f}")
    
    if p_tok_list:
        p_tok = np.concatenate(p_tok_list)
        w_tok = np.concatenate(w_tok_list)
        p_tok, w_tok = _sum_weights(p_tok, w_tok)
        s1r, tr = split_pairs(p_tok)
        same = s1["country"][s1r.astype(np.int64)] == tgt["country"][tr.astype(np.int64)]
        p_tok, w_tok = p_tok[same], w_tok[same]
        p_tok_top = topm_pairs(p_tok, w_tok, n_eval, 50)
        all_parts.append(p_tok_top)
        print(f"Token keys (topm=50) pairs: {len(p_tok_top):,}, recall: {model.blocking_recall(p_tok_top, eval_gt):.4f}")
        
    combined = np.unique(np.concatenate(all_parts))
    rec = model.blocking_recall(combined, eval_gt)
    print(f"== Combined Multistage Blocking Recall: {rec:.4f} ({len(combined):,} pairs, {len(combined)/n_eval:.1f} pairs/row) ==")

if __name__ == "__main__":
    eval_multistage()
