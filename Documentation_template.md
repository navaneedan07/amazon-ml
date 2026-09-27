# Methodology

## Methodology used
Supervised entity resolution: normalize all records, generate same-country
candidates with a union of DF-capped blocking keys, score every candidate pair
with a LightGBM binary classifier over fuzzy-string and structured-agreement
features, and keep pairs whose probability exceeds a threshold chosen by
grouped cross-validation to maximize macro-F0.5 (the challenge metric). A
per-entity top-k cap bounds false merges. The pipeline is fully restartable
(every stage caches to disk) and uses only the supplied training data — no
external entity databases, APIs, or geocoders.

## Candidate generation / blocking strategy
Candidates are the union of same-country matches from six key families,
computed over the normalized text:

- exact normalized business name
- order-insensitive name signature (legal suffixes stripped: ltd/llc/inc/
  gmbh/sarl/pvt/... including transliterated Indian suffixes)
- individual name tokens (length >= 4) whose document frequency on the target
  side is <= 60
- individual address tokens (length >= 4), DF <= 60
- house number (first numeric component), DF <= 40
- postal code (5- or 6-digit), DF <= 40

DF caps keep frequent tokens (e.g. "company", "road") from exploding the
candidate set. Key indexes are sorted arrays probed with binary search; pairs
are deduplicated per chunk and restricted to identical country labels
(exploration showed 100% of ground-truth pairs are same-country, so this only
removes noise). On the training data this blocking covers 99.9% of
ground-truth pairs (measured per-key: name tokens 83.8%, address tokens 95.4%,
house number 69.0%, any key 99.92%).

## Model architecture and feature engineering
20 features per pair, all computed from normalized fields:

- name: ratio, token-set ratio, token-sort ratio, partial ratio, Jaro-Winkler,
  token Jaccard, exact match, length ratio, signature Jaccard, signature exact
- address: ratio, token-set ratio, token-sort ratio, token Jaccard, exact
  match, length ratio
- structured: house-number match, postal-code match, script-mismatch flags
  (non-ASCII name/address disagreement), same-source flag (S2 vs S3)

Normalization itself removes most noise: NFKC + casefold + Latin-accent
folding (Indic scripts preserved), '&'→'and', punctuation stripping, US state
name→abbreviation folding, street-type expansion (rd→road, ave→avenue, ...),
and 'nan'/null placeholder removal.

The classifier is LightGBM (MIT license) trained on a deterministic subsample
(all positives + every 3rd negative) of the ~10^8 candidate pairs. Grouped
5-fold CV (folds = Source-1 entities) produces out-of-fold probabilities over
the full candidate set; the decision threshold maximizing macro-F0.5 is chosen
on those OOF scores, then the final model is fit on the whole subsample.
A top-10 cap on matches per Source-1 entity guards against runaway merges
(99.8% of true match-list degrees are <= 9; F0.5 is precision-heavy).

## Threshold selection
Macro-F0.5 is computed per Source-1 entity over the OOF probabilities for a
grid of thresholds; the argmax is used for test inference. Singletons are
included in the metric exactly as in the challenge formula (empty truth and
empty prediction = 1.0 for that entity).

## Scale engineering
The full dataset (~2.2M Source-1 + 10.3M target training records; 1.7M + 10M
test) does not fit in RAM, so: normalized text lives in concatenated
memory-mapped blobs; the feature matrix is a float32 disk memmap; LightGBM
trains with two_round on the subsample; workers share cache pages across
processes.

## External data
No external entity databases, APIs, geocoders, government registries, or
internet-based business lookups are used. Every blocking key and feature is
derived from the supplied train/test files alone.
