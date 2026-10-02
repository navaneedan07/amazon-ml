# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Bots  
**Team Members:** Navaneedan S S (Team Leader), Hitesh Venugopalan  
**Submission Date:** 2026-10-02

---

## 1. Executive Summary

We built a blocking + gradient-boosted-classifier pipeline that matches Source-1
business entities against Source-2/Source-3 target records using only the
supplied name/address/country fields. Candidates are generated with a union of
DF-capped exact, signature, and token-based blocking keys, scored with a
LightGBM classifier over 20 string-similarity features, and thresholded via
grouped cross-validation to maximize macro F0.5. The pipeline is built to run
at the full dataset scale (~2.2M Source-1 training entities, ~10.3M target
records, ~1.7M test entities) on a 16 GB / 12-core machine using memory-mapped
caches and a disk-backed feature matrix.

---

## 2. Methodology

### 2.1 Problem Analysis

Key issues observed in the business name/address data that drove the
normalization and blocking design:

- Inconsistent casing, punctuation, and Unicode forms (NFKC normalization and
  casefolding were needed to collapse cosmetic variants).
- Legal-entity suffixes (`Ltd`, `LLC`, `Inc`, `GmbH`, `SARL`, and
  transliterated Indian equivalents) that appear inconsistently across sources
  and would otherwise break exact-name matching.
- Street-address abbreviation variance (e.g. street-type abbreviations,
  US-state name/abbreviation folding) requiring a normalization/expansion
  step before token-level comparison.
- Non-ASCII / non-Latin script names and addresses in a subset of records,
  tracked as explicit flags (`nn`/`an`) and used as a model feature rather
  than discarded.
- Missing or malformed house numbers and postal codes, extracted via regex
  with graceful fallback to empty values rather than failing the row.
- Ground-truth pairs are 100% same-country, which is exploited directly in
  blocking (candidates are filtered to identical country labels) to cut the
  search space without any measurable recall loss.

### 2.2 Solution Strategy

**Approach Type:** Blocking + Classifier
**Core Innovation:** DF-capped multi-key blocking (exact name, order-insensitive
legal-suffix-stripped name signature, name/address tokens, house number,
postal code) combined via summed-IDF scoring and a per-entity top-m cap, feeding
a LightGBM classifier whose decision threshold is chosen by grouped 5-fold
cross-validation optimizing macro F0.5 directly (rather than a generic
probability cutoff like 0.5).

---

## 3. Candidate Generation (Blocking)

Candidates are generated per Source-1 entity as the union of same-country
matches across several key types, each built as a sorted (hash, row) index
over the target side and probed via `searchsorted`:

- **Blocking keys used:**
  - `name_exact`: full normalized business name
  - `name_sig`: sorted, legal-suffix-stripped name tokens (order/suffix
    insensitive)
  - `name_tok`: individual name tokens (length >= 4), DF-capped on the
    target side
  - `addr_tok`: individual address tokens (length >= 4), DF-capped
  - `house`: extracted house number, DF-capped (address-less fallback)
  - `postal`: postal/ZIP code (5- or 6-digit), uncapped
  - Token-type keys are additionally scored with summed IDF weights and
    capped to the top-m candidates per Source-1 entity (`topm_pairs`) to
    bound candidate-set size without relying on a single hard DF cutoff.
- **Candidate pairs generated:** 117,723,472 total candidate pairs across
  1,732,544 test Source-1 entities (from the actual `output/candidate_pairs.tsv`
  produced by this pipeline).
- **How true matches were not lost:** country-label filtering is safe because
  100% of training ground-truth pairs are same-country (verified against the
  training ground truth). Token keys use DF caps rather than hard drops of
  high-frequency tokens, and the per-key blocking recall was validated on a
  20k-row smoke-test subsample at 0.970 (see Section 5) before being run at
  full scale.

---

## 4. Matching Model

**Features used (20 total):**
- Name features: ratio, token-set ratio, token-sort ratio, partial ratio,
  Jaro-Winkler similarity, Jaccard similarity, exact-match flag, length ratio
- Address features: the same ratio/token-set/token-sort/partial/Jaro-Winkler/
  Jaccard/exact/length-ratio set, applied to normalized addresses
- Other: name-signature Jaccard + exact-match flags, house-number match flag,
  postal-code match flag, non-ASCII/script-mismatch flags (name and address),
  same-source flag (distinguishing Source-2 vs Source-3 target records)

**Model type:** LightGBM binary classifier (MIT-licensed), trained on a
deterministic subsample (all positive pairs + every 3rd negative pair) with
the feature matrix stored as a disk memmap and `two_round` training to bound
peak memory at full data scale.

**Threshold selection method:** Grouped-by-Source-1 5-fold cross-validation;
each fold's model predicts probabilities over the *full* candidate set so the
out-of-fold threshold is chosen against complete per-entity score
distributions. The decision threshold is swept over a grid and selected to
maximize macro F0.5. A top-10-candidates-per-Source-1-entity cap is applied
after thresholding as a guard against runaway merges (99.8% of true match
degrees in training are <= 9).

---

## 5. Results & Error Analysis

- **Candidate-set scale at full test run:** 1,732,544 test Source-1 entities,
  117,723,472 candidate pairs generated, 5,512,246 pairs retained as final
  matches after thresholding and the top-10 cap (computed directly from this
  submission's `output/candidate_pairs.tsv` and `output/matching_results.tsv`).
- **Common false positives (wrong merges):** Entities with highly generic or
  truncated names (e.g. common franchise/chain names) sharing the same
  postal/house-number keys but referring to distinct physical branches. The
  top-10-per-entity cap and same-source flag feature mitigate but do not
  eliminate this.
- **Common false negatives (missed matches):** True matches where the name is
  rendered in a non-Latin script on one side and transliterated on the other,
  and cases with simultaneously divergent address and name (no shared
  blocking key fires). The non-ASCII/script-mismatch flags help the
  classifier but can't recover candidates blocking never generated.

---

## 6. Conclusion

The pipeline combines cheap, DF-capped multi-key blocking with a compact
LightGBM classifier over fuzzy string-similarity features, tuned via grouped
cross-validation on macro F0.5 rather than a generic probability threshold.
The main lesson was that country-label filtering and IDF-weighted token keys
recovered most of the available blocking recall cheaply, while the top-k cap
and same-source feature were the most effective levers against false merges
at full scale.

---

## Appendix

### A. Code Artefacts

The complete, runnable code ships in this submission zip under
`code/business_entity_resolution/` (all source in `src/`, with a `README.md`
and a pinned `requirements.txt`). Structure:

- `src/textnorm.py`: name/address normalization (NFKC, casefold, accent
  fold, legal-suffix/street-type handling)
- `src/dataio.py`: on-disk, memory-mappable cache preparation (`prep()`) and
  loading (`load_role()`)
- `src/blocking.py`: target-side key indexes, candidate generation, and
  IDF-weighted top-m capping
- `src/features.py`: pairwise feature computation and parallel
  feature/prediction workers
- `src/model.py`: LightGBM training, grouped CV, macro-F0.5 threshold tuning
- `src/pipeline.py`: end-to-end orchestrator (entry point)

**Entry point** to reproduce `output/matching_results.tsv` and
`output/candidate_pairs.tsv` end-to-end:

```bash
pip install -r code/business_entity_resolution/requirements.txt
python code/business_entity_resolution/src/pipeline.py \
    --data dataset --cache ../cache --out output --jobs 11
```

See `code/business_entity_resolution/README.md` for the full set of run
modes (smoke test, cached-threshold reuse, etc.).

### B. Additional Results

*Include any additional charts, graphs, or detailed results.*

---

**Note:** Teams can modify sections according to their approach while
maintaining clarity and technical depth.
