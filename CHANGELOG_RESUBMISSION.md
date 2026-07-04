# Resubmission Changelog

This document maps each piece of reviewer feedback to the specific change made.

## Feedback: "A stronger submission would include a `src/` folder, scripts, tests, saved result files, and committed figures."

- Added `src/` — a tested, importable Python package (`config.py`, `data_loading.py`, `preprocessing.py`, `models.py`, `evaluation.py`, `experiments.py`) containing every piece of modeling logic that previously lived only inside notebook cells.
- Added `scripts/run_all_experiments.py` — a single command that runs every experiment end-to-end and writes results/figures to disk.
- Added `tests/` — a `pytest` suite (`test_data_loading.py`, `test_preprocessing.py`, `test_evaluation.py`, `test_experiments.py`) covering data loading, target conversion, duplicate/group-key logic, metric computation, threshold/low-prevalence tables, and full experiment runs, including an explicit assertion that the grouped split has zero group overlap between train and test.
- Added `results/*.csv` — every experiment's numeric output committed as plain CSV: dataset summary, original-style / deduplicated / grouped split results, the RF-across-splits comparison table, cross-validation results (stratified and grouped), threshold tables (original-style and grouped), the low-prevalence scenario, and the newer-dataset results.
- Added `figures/*.png` — class distribution, Spearman correlation heatmap, threshold trade-off curve, and the Random-Forest-across-split-strategies bar chart, all committed (not only shown as notebook cell outputs).

## Feedback: "...it could be even stronger with grouped validation to prevent identical feature vectors from appearing in both train and test..."

- Added `src/preprocessing.add_group_key`, which hashes each row's full feature vector into a `group_key`.
- Added `src/experiments.run_grouped_split`, using scikit-learn's `GroupShuffleSplit` keyed on `group_key`, so every row sharing an identical feature vector is forced onto the same side of the split — a strictly stronger guarantee than deduplicating before a random split.
- Added `src/experiments.cross_validation_summary(..., grouped=True)`, using `GroupKFold` on the same key, so cross-validation folds also respect the duplicate-safety guarantee.
- `tests/test_experiments.py::test_run_grouped_split_has_no_group_leakage` programmatically verifies the no-leakage guarantee rather than just asserting it inside the experiment code.
- Report Section 6.8 reports the results: Random Forest F1 drops from 0.9768 (original-style) to 0.9641 (deduplicated) to **0.9503 (grouped)**, and false negatives rise from 4 to 10 to **28**. This is now the headline, most-trusted Random Forest number in the report.
- Added an explicit comparison sentence noting that, under the strict grouped split, **SVM RBF slightly outperforms Random Forest by F1/recall**, while Random Forest keeps higher precision and PR-AUC.

## Feedback: "...and with testing on a newer phishing dataset."

- Replaced the earlier 55-row newer-dataset sample with the **full 58,645-row 2020 dataset** from Vrbančič, Fister Jr. & Podgorelec's public repository.
- Updated `data/README_newer_dataset.md` with provenance, class distribution, refresh command, preprocessing note, and SVM scalability note.
- Updated `src/data_loading.load_newer_dataset` and `src/experiments.run_newer_dataset_experiment` to run the full-data external validation offline.
- Updated `src/models.py` so the 2020 numeric dataset uses `StandardScaler` rather than treating numeric count features as categorical one-hot states.
- The newer-dataset check now reports full-data Random Forest Accuracy = **0.9568**, F1 = **0.9588**, PR-AUC = **0.9925**, based on a 14,662-row test split, not a 14-row test split.

## Additional engineering fix

- Fixed hidden shared-state risk in model construction: each pipeline now receives its own preprocessor instance instead of sharing one `ColumnTransformer` object across several pipelines.
- `pytest` passes: **21 tests passed**.

## Everything from the first resubmission is preserved

Deduplicated-split experiment, 3-fold stratified cross-validation, PR-AUC, threshold analysis, and the low-prevalence simulation are all still present and are now also covered by `src/` + tests + committed results, rather than living only in notebook cells.
