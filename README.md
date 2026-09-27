# Submission Package

This package contains a runnable Business Entity Resolution pipeline.

Place the official `dataset/` directory from the challenge beside `code/` and run the command in `code/business_entity_resolution/README.md`.

Expected generated files:
- `output/matching_results.tsv`
- `output/candidate_pairs.tsv`

Run the organizer's validator before submission:

```bash
python3 utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

The included pipeline does not fabricate test predictions without the challenge dataset. It generates them when the supplied dataset is placed in the expected location.
