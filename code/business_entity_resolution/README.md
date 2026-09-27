# Business Entity Resolution

Scalable pipeline for the Business Entity Resolution Challenge (ML Challenge 2026).

Designed for the full dataset size (~2.2M Source-1 training entities, ~10.3M
target records, ~1.7M test entities) on a 16 GB / 12-core machine: normalized
text lives in memory-mapped blobs, blocking uses sorted DF-capped key indexes,
the feature matrix is a disk memmap, and LightGBM trains with `two_round`.

## Pipeline

1. **Normalize** names/addresses (NFKC, casefold, accent fold, `&`→`and`,
   punctuation strip, US-state folding, street-type expansion, `nan` cleanup)
   into compact per-split caches (`cache/<split>/`).
2. **Blocking** — union of same-country candidate keys:
   - exact normalized name, order-insensitive name signature (legal suffixes
     stripped: `ltd/llc/inc/gmbh/sarl/...` incl. transliterated Indian forms)
   - name tokens (len >= 4) and address tokens (len >= 4), each dropped when
     its document frequency in the target side exceeds a cap (60 / 2000)
   - house number, postal code (5- or 6-digit)
   All key indexes are DF-capped sorted arrays probed with `searchsorted`;
   candidates are deduplicated per chunk and filtered to identical country
   labels (100% of training ground-truth pairs are same-country).
3. **Features** (20 per pair): ratio / token-set / token-sort / partial /
   Jaro-Winkler / Jaccard / exact / length-ratio for name and address,
   name-signature Jaccard + exact, house match, postal match, script-mismatch
   flags, same-source flag (S2 vs S3).
4. **Model**: LightGBM binary classifier (MIT license). Trained on a
   deterministic subsample (all positives + every 3rd negative).
5. **Threshold**: grouped-by-S1 5-fold CV; each fold model predicts the *full*
   candidate set so the OOF threshold is chosen on complete per-entity score
   distributions. Decision threshold maximizes macro-F0.5. A top-10 cap per
   Source-1 entity guards against runaway merges (99.8% of true degrees <= 9).
6. **Outputs**: `candidate_pairs.tsv` (blocking output actually fed to the
   model) and `matching_results.tsv` (matches after threshold; a subset of the
   candidates). One row per test Source-1 entity.

## Run

From the challenge `student_resource` directory:

```bash
pip install -r code/business_entity_resolution/requirements.txt

# full run (train + test). Cache is restartable; stages are skipped if present.
python code/business_entity_resolution/src/pipeline.py \
    --data dataset --cache ../cache --out output --jobs 11

# quick smoke test on 20k rows per file
python code/business_entity_resolution/src/pipeline.py \
    --data dataset --cache ../cache_smoke --out ../output_smoke \
    --limit 20000 --rounds 60 --folds 3

# reuse a cached threshold, skip CV
python code/business_entity_resolution/src/pipeline.py \
    --data dataset --cache ../cache --out output --skip-cv
```

Validate before submitting:

```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Smoke-test reference (20k rows/file): blocking recall 0.970, OOF macro-F0.5
0.969 @ threshold 0.95. Full-data smoke of the same code path validated PASS
with the organizer's validator.

The implementation uses only the supplied data — no external business
databases, geocoders, or APIs.
