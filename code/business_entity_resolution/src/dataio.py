"""Data loading and compact on-disk cache.

Prep writes, per split ('train'/'test') and role ('s1'/'tgt'):

    {base}_ids.npy       fixed-width bytes entity ids (S dtype)
    {base}_names.bin     concatenated utf-8 normalized names
    {base}_names.bin.off.npy  int64 offsets (len n+1); slice [off[i]:off[i+1]]
    {base}_addrs.bin/off same for normalized addresses
    {base}_house.npy     house number  (S8, b'' if absent)
    {base}_zip.npy       postal code   (S12, b'' if absent)
    {base}_nn.npy        name has non-ascii chars (bool)
    {base}_an.npy        addr has non-ascii chars (bool)
    {base}_country_str.npy  country labels (U dtype)
    {base}_country.npy   country code (uint8)
    countries.json       {country str -> code} for the split
    _DONE                marker: cache is complete (safe to reuse)

Blobs are memory-mapped by feature workers, so pages are shared across
processes instead of being copied into each worker.
"""
import json
import mmap
import re
from pathlib import Path

import numpy as np
import pandas as pd

from textnorm import _LATIN1_FOLD, _STATES_RE, _states_sub, _STREET

COLS = ["entity_id", "business_name", "business_address", "country"]

_NOT_WORD_RE = re.compile(r"[^\w\s]+")
_WS_RE = re.compile(r"\s+")
# 5-digit (US ZIP / common EU) or 6-digit (India PIN) codes; also French
# CEDEX-style 5+2 is handled by taking the leading 5.
_ZIP_RE = r"(?<!\d)(\d{6}|\d{5})(?!\d)"
_HOUSE_RE = r"\b(\d{1,7}[a-z]?)\b"
_NONASCII_RE = r"[^\x00-\x7F]"


def _norm_basic(s: pd.Series) -> pd.Series:
    """Vectorized norm_text: NFKC, casefold, latin fold, &->and, strip punct, ws."""
    s = s.fillna("").astype(str)
    s = s.str.normalize("NFKC").str.casefold().str.translate(_LATIN1_FOLD)
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(_NOT_WORD_RE, " ", regex=True)
    s = s.str.replace("_", " ", regex=False)
    s = s.str.replace(_WS_RE, " ", regex=True).str.strip()
    return s


def _norm_names(s: pd.Series) -> pd.Series:
    return _norm_basic(s)


_STREET_TAB = np.array(list(_STREET.items()), dtype=object)


def _norm_addrs(s: pd.Series) -> pd.Series:
    """Vectorized norm_addr: basic norm + 'null' drop + US state fold +
    street-type map applied to every token except the first (keeps
    'St' == saint at the head)."""
    a = _norm_basic(s)
    a = a.str.replace(r"\bnull\b", " ", regex=True)
    a = a.str.replace(_STATES_RE, _states_sub, regex=True)
    a = a.str.replace(_WS_RE, " ", regex=True).str.strip()
    smap = dict(_STREET_TAB.tolist())
    out = []
    for v in a.to_numpy():
        if not v or " " not in v:
            out.append(v)
            continue
        head, rest = v.split(" ", 1)
        out.append(head + " " + " ".join(smap.get(t, t) for t in rest.split()))
    return pd.Series(out, index=a.index, dtype=object)


def _read(path, limit=None):
    df = pd.read_csv(path, sep="\t", dtype=str, nrows=limit)
    for c in COLS:
        if c not in df:
            df[c] = ""
    return df[COLS].fillna("")


def _write_str_blob(path, values):
    """Write concatenated utf-8 strings + .off.npy offsets (len n+1)."""
    offsets = np.empty(len(values) + 1, dtype=np.int64)
    offsets[0] = 0
    with open(path, "wb") as f:
        pos = 0
        for i, s in enumerate(values, 1):
            b = s.encode("utf-8")
            f.write(b)
            pos += len(b)
            offsets[i] = pos
    np.save(path + ".off.npy", offsets)


def prep(data_dir, cache_dir, split, limit=None, force=False):
    """Normalize all source files for a split and write the cache."""
    data_dir, cache_dir = Path(data_dir), Path(cache_dir)
    out = cache_dir / split
    done = out / "_DONE"
    if done.exists() and not force and limit is None:
        print(f"  [{split}] cache exists, skipping prep ({done})", flush=True)
        return json.loads((out / "countries.json").read_text(encoding="utf-8"))
    out.mkdir(parents=True, exist_ok=True)

    jobs = {
        "s1": [data_dir / split / f"{split}_source1.tsv"],
        "tgt": [data_dir / split / f"{split}_source2.tsv",
                data_dir / split / f"{split}_source3.tsv"],
    }

    n_rows = {}
    for role, paths in jobs.items():
        id_l, name_l, addr_l = [], [], []
        house_l, zip_l, nn_l, an_l, ctry_l = [], [], [], [], []
        first_tgt_rows = None
        for p in paths:
            df = _read(p, limit)
            names = _norm_names(df.business_name)
            addrs = _norm_addrs(df.business_address)
            id_l.extend(df.entity_id)
            name_l.extend(names)
            addr_l.extend(addrs)
            addr_s = pd.Series(addrs, dtype=object)
            house_l.extend(addr_s.str.extract(_HOUSE_RE, expand=False).fillna(""))
            zip_l.extend(addr_s.str.extract(_ZIP_RE, expand=False).fillna(""))
            nn_l.extend(names.str.contains(_NONASCII_RE, regex=True, na=False))
            an_l.extend(addr_s.str.contains(_NONASCII_RE, regex=True, na=False))
            ctry_l.extend(df.country)
            if role == "tgt" and first_tgt_rows is None:
                first_tgt_rows = len(df)  # rows coming from source2 (for same_source)
            print(f"  [{split}/{role}] {p.name}: {len(df)} rows", flush=True)
            del df, names, addrs, addr_s
        base = str(out / role)
        maxlen = max((len(s.encode("utf-8")) for s in id_l), default=1)
        np.save(base + "_ids.npy",
                np.asarray([s.encode("utf-8") for s in id_l], dtype=f"S{maxlen}"))
        _write_str_blob(base + "_names.bin", name_l)
        _write_str_blob(base + "_addrs.bin", addr_l)
        np.save(base + "_house.npy",
                np.asarray([h.encode()[:8] for h in house_l], dtype="S8"))
        np.save(base + "_zip.npy",
                np.asarray([z.encode()[:12] for z in zip_l], dtype="S12"))
        np.save(base + "_nn.npy", np.asarray(nn_l, dtype=bool))
        np.save(base + "_an.npy", np.asarray(an_l, dtype=bool))
        np.save(str(out / role) + "_country_str.npy",
                np.asarray(ctry_l, dtype=f"U{max((len(c) for c in ctry_l), default=1)}"))
        n_rows[role] = len(id_l)
        if role == "tgt":
            (out / "tgt_n2.txt").write_text(str(first_tgt_rows or 0), encoding="utf-8")
        del id_l, name_l, addr_l, house_l, zip_l, nn_l, an_l, ctry_l

    countries = set()
    for role in jobs:
        countries.update(np.load(str(out / role) + "_country_str.npy",
                                 allow_pickle=False).tolist())
    cmap = {c: i for i, c in enumerate(sorted(countries))}
    for role in jobs:
        cs = np.load(str(out / role) + "_country_str.npy")
        np.save(str(out / role) + "_country.npy",
                np.asarray([cmap[c] for c in cs.tolist()], dtype=np.uint8))
    (out / "countries.json").write_text(json.dumps(cmap), encoding="utf-8")
    if limit is None:
        done.write_text("ok\n", encoding="utf-8")
    print(f"  [{split}] countries: {cmap} rows: {n_rows}", flush=True)
    return cmap


class StrBlob:
    """Random access to the concatenated utf-8 string blob."""

    def __init__(self, bin_path):
        self.path = str(bin_path)
        self.off = np.load(self.path + ".off.npy", mmap_mode="r")
        f = open(self.path, "rb")
        self._f = f
        if f.seek(0, 2) == 0:
            self.mm = None
        else:
            f.seek(0)
            self.mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.off) - 1

    def get(self, i):
        a, b = int(self.off[i]), int(self.off[i + 1])
        if a == b or self.mm is None:
            return ""
        return self.mm[a:b].decode("utf-8")


def load_role(cache_dir, split, role, with_text=True):
    """Load arrays for a role; returns dict. Text fields are lazy StrBlobs."""
    base = str(Path(cache_dir) / split / role)
    ids = np.load(base + "_ids.npy", mmap_mode="r")
    d = {
        "ids": ids,
        "country": np.load(base + "_country.npy", mmap_mode="r"),
        "house": np.load(base + "_house.npy", mmap_mode="r"),
        "zip": np.load(base + "_zip.npy", mmap_mode="r"),
        "nn": np.load(base + "_nn.npy", mmap_mode="r"),
        "an": np.load(base + "_an.npy", mmap_mode="r"),
        "n": len(ids),
    }
    if with_text:
        d["names"] = StrBlob(base + "_names.bin")
        d["addrs"] = StrBlob(base + "_addrs.bin")
    return d
