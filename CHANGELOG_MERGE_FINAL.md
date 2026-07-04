# Third round: merging two independent fix lineages into one final version

Two separate rounds of fixes were made to this project after grading feedback,
each catching different real problems. This document merges both into a single
version and records what came from where.

## Lineage A: full-dataset + methodology fixes (kept from this round's base)

- Replaced the 55-row newer-dataset sample with the **full 58,645-row**
  `dataset_small.csv` (Vrbančič, Fister Jr. & Podgorelec, 2020), verified against
  the published paper's own reported class counts (30,647 phishing / 27,998
  legitimate) - not a fabricated or hand-typed sample.
- Fixed a real feature-engineering bug: the 2020 dataset's features are
  continuous counts/ratios/timings, not the 2015 dataset's categorical
  {-1, 0, 1} states, so one-hot-encoding them (the original approach) was
  methodologically wrong. Added `src/models.make_numeric_preprocessor` /
  `make_numeric_model_pipelines` using `StandardScaler` instead.
- Added an RBF-feature-map + `SGDClassifier` fallback for SVM on the newer
  dataset, since exact probability-calibrated `SVC` is quadratic and
  impractical at 58,645 rows.
- Fixed a shared-mutable-state code smell: `make_model_pipelines` previously
  built one `ColumnTransformer` instance and reused it across all four
  pipelines; now each pipeline gets its own preprocessor instance.

## Lineage B: reproducibility + statistical-rigor fixes (merged in this round)

- Replaced the `hash(tuple(row.values))` group key in
  `src/preprocessing.add_group_key` with `_stable_row_hash`, a deterministic
  SHA-256 hash of each row's canonical string representation. Python's
  built-in `hash()` is salted per-process for strings (`PYTHONHASHSEED`) and
  truncates to a machine-width int - both avoidable weaknesses for a key
  whose only job is exact-duplicate identification.
- **This surfaced a real methodological finding, not just a style fix.**
  Re-running the pipeline with the new key changed the *specific* single
  train/test partition produced by `GroupShuffleSplit`, because scikit-learn
  sorts unique groups internally before applying the `random_state`-controlled
  shuffle, and sorting hex-string keys gives a different order than sorting
  int-hash keys. The report now states this explicitly (Section 6.8) instead
  of presenting one grouped split as "the strictest, most trustworthy"
  estimate, and treats **group-aware cross-validation** (`GroupKFold`,
  averaged over folds, insensitive to this kind of single-split ordering
  artifact) as the authoritative "grouped" number: Random Forest mean F1
  0.9720 -> 0.9612.
- The same fix reversed an earlier claim that SVM RBF slightly outperformed
  Random Forest under the grouped split; with the stable key, Random Forest
  wins on every metric. This is now noted directly in the report rather than
  left as a stale, contradicted claim.
- Added `src/evaluation.bootstrap_metric_ci`, a nonparametric bootstrap
  confidence interval, and wired it into `run_newer_dataset_experiment`
  (which now returns a third value, a per-model CI table). On the full
  58,645-row dataset this produces a tight band (Random Forest F1 95% CI
  approximately [0.956, 0.962]) - a concrete, quantified confirmation that
  the full dataset (Lineage A) actually resolved the small-sample
  uncertainty an earlier 55-row draft would have carried, rather than
  merely asserting that it did.
- Added direct unit tests for the new hash function
  (`test_stable_row_hash_is_deterministic_across_calls`,
  `test_stable_row_hash_differs_for_different_rows`,
  `test_stable_row_hash_is_not_python_builtin_hash`) and for the CI table
  shape/bracketing (`test_run_newer_dataset_experiment_runs_end_to_end`).
- **Re-executed the entire notebook.** After merging, the notebook's cached
  cell outputs still reflected pre-fix numbers from one or both lineages.
  Since the sandbox has no `jupyter`/`nbconvert` and no network to install
  them, a small custom in-process notebook executor was used to actually
  re-run all 26 code cells. All executed with zero errors, and every table
  in the notebook now matches the report, README, and committed
  `results/*.csv` files exactly.

## Net effect

`scripts/run_all_experiments.py` runs clean end-to-end on the full 58,645-row
external dataset, the stable-hash grouped split, and every other experiment;
22/22 project-runnable tests pass; the notebook, report, README, and
`results/*.csv` are all mutually consistent; and the report's strongest claims
(grouped-split trustworthiness, external-dataset generalization) are each now
backed by a quantified robustness check (cross-validation instead of a single
split; a bootstrap CI instead of a bare point estimate) rather than resting on
prose alone.
