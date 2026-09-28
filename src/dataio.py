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
    street-type map applied to every token except the first."""
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


class StrBlob:
    """Memory-mapped accessor for concatenated string blob + offsets."""

    def __init__(self, path):
        self.off = np.load(path + ".off.npy", mmap_mode="r")
        self.f = open(path, "rb")
        self.m = mmap.mmap(self.f.fileno(), 0, access=mmap.ACCESS_READ)

    def __len__(self):
        return len(self.off) - 1

    def get(self, i: int) -> str:
        a, b = int(self.off[i]), int(self.off[i + 1])
        if a == b:
            return ""
        return self.m[a:b].decode("utf-8", "replace")

    def close(self):
        self.m.close()
        self.f.close()


def prepare_split(data_dir, cache_dir, split="train", limit=None, force=False):
    """Normalize and write on-disk blob cache for a split ('train'|'test')."""
    base = Path(cache_dir) / split
    done_file = base / "_DONE"
    if done_file.exists() and not force and limit is None:
        print(f"  [{split}] cache exists, skipping prep ({done_file})", flush=True)
        return
    base.mkdir(parents=True, exist_ok=True)

    s1_path = Path(data_dir) / split / f"{split}_source1.tsv"
    s2_path = Path(data_dir) / split / f"{split}_source2.tsv"
    s3_path = Path(data_dir) / split / f"{split}_source3.tsv"

    print(f"  [{split}/s1] {s1_path.name}: reading...", flush=True)
    df_s1 = _read(s1_path, limit)
    print(f"  [{split}/tgt] {s2_path.name}: reading...", flush=True)
    df_s2 = _read(s2_path, limit)
    print(f"  [{split}/tgt] {s3_path.name}: reading...", flush=True)
    df_s3 = _read(s3_path, limit)

    tgt_df = pd.concat([df_s2, df_s3], ignore_index=True)
    (base / "tgt_n2.txt").write_text(str(len(df_s2)))

    all_countries = pd.concat([df_s1["country"], tgt_df["country"]]).unique()
    c_map = {str(c): i for i, c in enumerate(sorted(all_countries))}
    (base / "countries.json").write_text(json.dumps(c_map))

    def _write_role(role, df):
        prefix = f"{role}_"
        np.save(base / f"{prefix}ids.npy", df["entity_id"].astype(bytes).to_numpy())
        c_codes = df["country"].map(lambda c: c_map.get(str(c), 0)).astype(np.uint8).to_numpy()
        np.save(base / f"{prefix}country.npy", c_codes)
        np.save(base / f"{prefix}country_str.npy", df["country"].astype(str).to_numpy())

        names_norm = _norm_names(df["business_name"])
        _write_str_blob(str(base / f"{prefix}names.bin"), names_norm)

        addrs_norm = _norm_addrs(df["business_address"])
        _write_str_blob(str(base / f"{prefix}addrs.bin"), addrs_norm)

        h = addrs_norm.str.extract(_HOUSE_RE, expand=False).fillna("")
        np.save(base / f"{prefix}house.npy", h.astype("S8").to_numpy())

        z = addrs_norm.str.extract(_ZIP_RE, expand=False).fillna("")
        np.save(base / f"{prefix}zip.npy", z.astype("S12").to_numpy())

        nn = df["business_name"].str.contains(_NONASCII_RE, regex=True).fillna(False)
        an = df["business_address"].str.contains(_NONASCII_RE, regex=True).fillna(False)
        np.save(base / f"{prefix}nn.npy", nn.to_numpy(dtype=bool))
        np.save(base / f"{prefix}an.npy", an.to_numpy(dtype=bool))

    _write_role("s1", df_s1)
    _write_role("tgt", tgt_df)
    done_file.write_text("ok")
    print(f"  [{split}] done -> {base}", flush=True)


def load_role(cache_dir, split, role, with_text=True):
    """Load a role dictionary for a split."""
    base = Path(cache_dir) / split / f"{role}_"
    d = {
        "ids": np.load(str(base) + "ids.npy", mmap_mode="r"),
        "country": np.load(str(base) + "country.npy", mmap_mode="r"),
        "country_str": np.load(str(base) + "country_str.npy", mmap_mode="r"),
        "house": np.load(str(base) + "house.npy", mmap_mode="r"),
        "zip": np.load(str(base) + "zip.npy", mmap_mode="r"),
        "nn": np.load(str(base) + "nn.npy", mmap_mode="r"),
        "an": np.load(str(base) + "an.npy", mmap_mode="r"),
    }
    d["n"] = len(d["ids"])
    if with_text:
        d["names"] = StrBlob(str(base) + "names.bin")
        d["addrs"] = StrBlob(str(base) + "addrs.bin")
    return d
