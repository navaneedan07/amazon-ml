"""Blocking / candidate generation — the source of candidate_pairs.tsv.

Key types (candidate lists are later filtered to same-country):

    name_exact  full normalized business name
    name_sig    sorted, legal-suffix-stripped name tokens (word order / suffix safe)
    name_tok    name tokens, length >= 4, target-side document frequency <= cap
    addr_tok    address tokens, length >= 4, target-side document frequency <= cap
    house       extracted house number, DF <= cap (addr-less fallback)
    postal      postal/ZIP code, DF <= cap

Implementation: for each key type we build a sorted (hash, row) index over the
target side, then probe it with hashed query keys via searchsorted. Pairs are
emitted as uint64 (s1_row << 32 | tgt_row), deduplicated across key types with
np.unique. Same-country enforcement happens after merging key types.

All blocking keys are derived only from the supplied data (no external lookups).
"""
import numpy as np
import pandas as pd

from textnorm import name_sig

MASK32 = np.uint64(0xFFFFFFFF)


def combine_pairs(s1_rows, tgt_rows):
    return (s1_rows.astype(np.uint64) << np.uint64(32)) | tgt_rows.astype(np.uint64)


def split_pairs(pairs):
    return pairs >> np.uint64(32), pairs & MASK32


def _hash_strs(vals):
    """Stable 64-bit hashes of strings via pandas' xxhash-based hasher."""
    if len(vals) == 0:
        return np.empty(0, dtype=np.uint64)
    return pd.util.hash_pandas_object(
        pd.Series(vals, dtype=object), index=False
    ).to_numpy(dtype=np.uint64)


def _sorted_index(hashes, rows):
    order = np.argsort(hashes, kind="stable")
    return hashes[order], rows[order]


class KeyIndex:
    """One sorted key index over the target side."""

    def __init__(self, name, hashes, rows, cap=None):
        self.name = name
        self.cap = cap
        self.h, self.r = _sorted_index(hashes, rows)

    def __len__(self):
        return len(self.h)

    def with_cap(self, cap):
        """Return a copy restricted to keys whose target-side DF <= cap."""
        if cap is None or self.cap is not None:
            raise ValueError("with_cap only applies to loose indexes (cap=None)")
        if len(self.h) == 0:
            return KeyIndex(self.name, self.h, self.r, cap)
        b = np.flatnonzero(np.diff(self.h)) + 1
        starts = np.concatenate(([0], b))
        ends = np.concatenate((b, [len(self.h)]))
        df = (ends - starts).astype(np.int64)
        posting_df = np.repeat(df, ends - starts)
        keep = posting_df <= cap
        return KeyIndex(self.name, self.h[keep], self.r[keep], cap)

    def probe(self, qh, s1_rows):
        """Return uint64 pairs for query hashes (already DF-filtered upstream)."""
        if len(qh) == 0:
            return np.empty(0, dtype=np.uint64)
        lo = np.searchsorted(self.h, qh, side="left")
        hi = np.searchsorted(self.h, qh, side="right")
        cnt = (hi - lo).astype(np.int64)
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, dtype=np.uint64)
        if total > 400_000_000:
            raise RuntimeError(
                f"blocking key '{self.name}' expanded to {total} pairs for one "
                f"chunk — lower its DF cap"
            )
        starts = np.cumsum(cnt) - cnt
        within = np.arange(total, dtype=np.int64) - np.repeat(starts, cnt)
        tgt = self.r[np.repeat(lo, cnt) + within]
        rep_s1 = np.repeat(s1_rows.astype(np.int64), cnt)
        return combine_pairs(rep_s1, tgt)


def _emit_name_keys(names):
    """Per-record name keys: exact, signature."""
    exact, sig = [], []
    for s in names:
        if not s:
            exact.append("")
            sig.append("")
            continue
        exact.append(s)
        sig.append(name_sig(s))
    return exact, sig


def build_target_index(tgt, cap_exact=2000, cap_tok=None, cap_bi=None,
                       cap_house=None, cap_postal=None, min_tok_len=4):
    """Build the list of KeyIndexes over the target (S2+S3) side."""
    n = tgt["n"]
    rows = np.arange(n, dtype=np.int64)

    names = [tgt["names"].get(i) for i in range(n)]
    exact, sig = _emit_name_keys(names)
    nospace = ["".join(s.split()) if s else "" for s in exact]

    idx = []
    def str_key(vals, label, cap):
        h = _hash_strs(vals)
        keep = np.array([bool(s) for s in vals], dtype=bool)
        uv, cv = np.unique(h[keep], return_counts=True)
        ok = cv <= cap
        m = keep & np.isin(h, uv[ok])
        idx.append(KeyIndex(label, h[m], rows[m], cap))
        print(f"    {label}: {len(idx[-1])} keys", flush=True)

    str_key(exact, "name_exact", cap_exact)
    str_key(sig, "name_sig", cap_exact)
    str_key(nospace, "name_nospace", cap_exact)
    del names, exact, sig, nospace

    def token_keys(field, label, cap, min_len, mode="uni"):
        blob = tgt[field]
        tok_rows, tok_hash = [], []
        CHUNK = 500_000
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            toks = []
            srows = []
            for i in range(start, end):
                s = blob.get(i)
                if not s:
                    continue
                tt = s.split()
                if mode == "uni":
                    keys = [t for t in tt if len(t) >= min_len]
                else:
                    keys = [f"{tt[j]} {tt[j+1]}" for j in range(len(tt) - 1)]
                toks.extend(keys)
                srows.extend([i] * len(keys))
            if not toks:
                continue
            hh = _hash_strs(toks)
            tok_hash.append(hh)
            tok_rows.append(np.asarray(srows, dtype=np.int64))
        if not tok_hash:
            idx.append(KeyIndex(label, np.empty(0, np.uint64), np.empty(0, np.int64), cap))
            print(f"    {label}: 0 keys", flush=True)
            return
        hh = np.concatenate(tok_hash)
        rr = np.concatenate(tok_rows)
        if cap is not None:
            uh, ce = np.unique(hh, return_counts=True)
            ok = ce <= cap
            good = np.isin(hh, uh[ok])
            hh, rr = hh[good], rr[good]
        idx.append(KeyIndex(label, hh, rr, cap))
        print(f"    {label}: {len(idx[-1])} keys", flush=True)

    token_keys("names", "name_bi", cap_bi, 2, mode="bi")
    token_keys("names", "name_tok", cap_tok, min_tok_len)
    token_keys("addrs", "addr_tok", cap_tok, min_tok_len)
    token_keys("addrs", "addr_bi", cap_bi, 2, mode="bi")

    def comp_key(col, label, cap, norm=False):
        arr = tgt[col]
        vals, rows_c = [], []
        for i in range(n):
            v = bytes(arr[i]).decode("utf-8", "ignore")
            if v:
                if norm:
                    v = v.lstrip("0")
                    if not v:
                        continue
                vals.append(v)
                rows_c.append(i)
        if not vals:
            idx.append(KeyIndex(label, np.empty(0, np.uint64), np.empty(0, np.int64), cap))
            print(f"    {label}: 0 keys", flush=True)
            return
        hh = _hash_strs(vals)
        if cap is not None:
            uh, ce = np.unique(hh, return_counts=True)
            ok = ce <= cap
            good = np.isin(hh, uh[ok])
            hh = hh[good]
            rows_c = np.asarray(rows_c, np.int64)[good]
        idx.append(KeyIndex(label, hh, np.asarray(rows_c, np.int64), cap))
        print(f"    {label}: {len(idx[-1])} keys", flush=True)

    comp_key("house", "house", cap_house)
    comp_key("house", "house_norm", cap_house, norm=True)
    comp_key("zip", "postal", cap_postal)
    return idx


def filtered_index(index, cap):
    return index.with_cap(cap)


class TfidfIndex:
    """Wraps a loose KeyIndex with per-key IDF weights."""

    def __init__(self, ix):
        self.name = ix.name
        self.h, self.r = ix.h, ix.r
        b = np.flatnonzero(np.diff(self.h)) + 1
        starts = np.concatenate(([0], b))
        ends = np.concatenate((b, [len(self.h)]))
        df = (ends - starts).astype(np.float64)
        self.N = float(self.r[-1] + 1) if len(self.r) else 1.0
        self.idf = np.log1p(self.N / df).astype(np.float32)
        self.starts, self.ends = starts, ends

    def probe_scored(self, qh, s1_rows, idf_from="tgt"):
        if len(qh) == 0:
            return np.empty(0, np.uint64), np.empty(0, np.float32)
        lo = np.searchsorted(self.h, qh, side="left")
        hi = np.searchsorted(self.h, qh, side="right")
        cnt = (hi - lo).astype(np.int64)
        total = int(cnt.sum())
        if total == 0:
            return np.empty(0, np.uint64), np.empty(0, np.float32)
        if total > 400_000_000:
            raise RuntimeError(
                f"blocking key '{self.name}' expanded to {total} pairs for one "
                f"chunk — lower its DF cap")
        run = np.searchsorted(self.starts, lo, side="right") - 1
        run_c = np.maximum(run, 0)
        qw = np.where(cnt > 0, self.idf[run_c], np.float32(0.0)).astype(np.float32)
        starts = np.cumsum(cnt) - cnt
        within = np.arange(total, dtype=np.int64) - np.repeat(starts, cnt)
        tgt = self.r[np.repeat(lo, cnt) + within]
        rep_s1 = np.repeat(s1_rows.astype(np.int64), cnt)
        pairs = combine_pairs(rep_s1, tgt)
        w = np.repeat(qw, cnt)
        return pairs, w


def _sum_weights(pairs, w):
    order = np.argsort(pairs, kind="stable")
    p = pairs[order]
    ws = w[order]
    del order
    is_new = np.empty(len(p), dtype=bool)
    is_new[0] = True
    np.not_equal(p[1:], p[:-1], out=is_new[1:])
    starts = np.flatnonzero(is_new)
    del is_new
    sums = np.add.reduceat(ws, starts).astype(np.float32)
    return p[starts], sums


def topm_pairs(pairs, w, n_s1, m):
    if len(pairs) == 0 or m <= 0:
        return pairs
    s1r = pairs >> np.uint64(32)
    tr = pairs & np.uint64(0xFFFFFFFF)
    wf = w.astype(np.float64)
    q = np.rint((wf / (wf + 8.0)) * 65535.0).astype(np.uint64)
    v = (s1r << np.uint64(40)) | ((np.uint64(65535) - q) << np.uint64(24)) | tr
    v.sort()
    g = v >> np.uint64(40)
    b = np.searchsorted(g, np.arange(n_s1 + 1, dtype=np.uint64), side="left")
    sizes = (b[1:] - b[:-1]).astype(np.int64)
    rank = np.arange(len(v), dtype=np.int64) - np.repeat(b[:-1], sizes)
    keep = rank < np.repeat(np.minimum(sizes, m), sizes)
    v = v[keep]
    return ((v >> np.uint64(40)) << np.uint64(32)) | (v & np.uint64(0xFFFFFF))


def block_all_scored(s1, tgt, tf_indexes, s1_country, tgt_country, chunk=100_000,
                     topm=0, n_s1=None):
    exact_names = {"name_exact", "name_sig", "name_nospace"}
    by_name = {ix.name: ix for ix in tf_indexes}
    for start in range(0, s1["n"], chunk):
        end = min(start + chunk, s1["n"])
        qkeys = query_keys(s1, start, end,
                           [type("K", (), {"name": n})() for n in by_name])
        exact_p = []
        tok_p, tok_w = [], []
        for ix, qh, qr in qkeys:
            if len(qh) == 0:
                continue
            p, w = by_name[ix.name].probe_scored(qh, qr)
            if len(p):
                if ix.name in exact_names:
                    exact_p.append(p)
                else:
                    tok_p.append(p)
                    tok_w.append(w)
        
        parts = []
        if exact_p:
            ep = np.concatenate(exact_p)
            s1r, tr = split_pairs(ep)
            same = s1_country[s1r.astype(np.int64)] == tgt_country[tr.astype(np.int64)]
            if not same.all():
                ep = ep[same]
            parts.append(np.unique(ep))
            del exact_p, ep, s1r, tr, same

        if tok_p:
            tp = np.concatenate(tok_p)
            tw = np.concatenate(tok_w)
            tp, tw = _sum_weights(tp, tw)
            s1r, tr = split_pairs(tp)
            same = s1_country[s1r.astype(np.int64)] == tgt_country[tr.astype(np.int64)]
            if not same.all():
                tp, tw = tp[same], tw[same]
            if topm and n_s1:
                tp = topm_pairs(tp, tw, min(end, s1["n"]), topm)
            parts.append(tp)
            del tok_p, tok_w, tp, tw, s1r, tr, same

        if not parts:
            yield start, end, np.empty(0, np.uint64)
            continue
        pairs = np.unique(np.concatenate(parts))
        yield start, end, pairs


def query_keys(s1, start, end, indexes, min_tok_len=4):
    names = [s1["names"].get(i) for i in range(start, end)]
    exact, sig = _emit_name_keys(names)
    nospace = ["".join(s.split()) if s else "" for s in exact]
    rows = np.arange(start, end, dtype=np.int64)
    out = []

    by_name = {ix.name: ix for ix in indexes}
    h = _hash_strs(exact)
    m = np.array([bool(s) for s in exact], dtype=bool)
    out.append((by_name["name_exact"], h[m], rows[m]))
    h = _hash_strs(sig)
    m = np.array([bool(s) for s in sig], dtype=bool)
    out.append((by_name["name_sig"], h[m], rows[m]))
    h = _hash_strs(nospace)
    m = np.array([bool(s) for s in nospace], dtype=bool)
    out.append((by_name["name_nospace"], h[m], rows[m]))

    for field, label, mode in (("names", "name_bi", "bi"),
                               ("names", "name_tok", "uni"),
                               ("addrs", "addr_tok", "uni"),
                               ("addrs", "addr_bi", "bi")):
        blob = s1[field]
        toks, srows = [], []
        for j, i in enumerate(range(start, end)):
            s = blob.get(i)
            if not s:
                continue
            tt = s.split()
            if mode == "uni":
                keys = [t for t in tt if len(t) >= min_tok_len]
            else:
                keys = [f"{tt[j]} {tt[j+1]}" for j in range(len(tt) - 1)]
            toks.extend(keys)
            srows.extend([i] * len(keys))
        out.append((by_name[label], _hash_strs(toks), np.asarray(srows, np.int64)))

    for col, labels in (("house", ("house", "house_norm")), ("zip", ("postal",))):
        arr = s1[col]
        vals, rws = [], []
        for i in range(start, end):
            v = bytes(arr[i]).decode("utf-8", "ignore")
            if v:
                vals.append(v)
                rws.append(i)
        if "house_norm" in labels:
            vnorm = [v.lstrip("0") for v in vals]
            kv = [(v, r) for v, r in zip(vnorm, rws) if v]
            out.append((by_name["house_norm"],
                        _hash_strs([v for v, _ in kv]),
                        np.asarray([r for _, r in kv], np.int64)))
        out.append((by_name[labels[0]], _hash_strs(vals), np.asarray(rws, np.int64)))
    return out


def block_chunk(s1, tgt, indexes, start, end, s1_country, tgt_country, min_tok_len=4):
    qkeys = query_keys(s1, start, end, indexes, min_tok_len)
    parts = []
    for ix, qh, qr in qkeys:
        if len(qh):
            parts.append(ix.probe(qh, qr))
    if not parts:
        return np.empty(0, dtype=np.uint64)
    pairs = np.unique(np.concatenate(parts))
    s1r, tr = split_pairs(pairs)
    same = s1_country[s1r] == tgt_country[tr]
    if not same.all():
        pairs = pairs[same]
    return pairs


def block_all(s1, tgt, indexes, s1_country, tgt_country, chunk=200_000, min_tok_len=4):
    for start in range(0, s1["n"], chunk):
        end = min(start + chunk, s1["n"])
        yield start, end, block_chunk(
            s1, tgt, indexes, start, end, s1_country, tgt_country, min_tok_len
        )
