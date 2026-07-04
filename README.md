# Critical Reproduction and Evaluation of Machine Learning for Phishing Website Detection

## Project description

This repository contains a final project for the course **Data Science in Cybersecurity**.
The project critically evaluates a published tutorial about phishing website detection using machine learning, reproduces the author's solution, and checks whether the claims are supported by the data and experiments.

This is the **second resubmission**, updated in response to reviewer feedback. See [`CHANGELOG_RESUBMISSION.md`](CHANGELOG_RESUBMISSION.md) for exactly what changed and why.

## Selected article / tutorial

- **Article:** Phishing Websites Detection — Rishabh Shukla
- **Article link:** https://rishy.github.io/projects/2015/05/08/phishing-websites-detection/

## Original GitHub repository

- **Original repository:** https://github.com/rishy/phishing-websites

## Datasets used

- **Article dataset (2015):** https://raw.githubusercontent.com/rishy/phishing-websites/master/Datasets/phising.csv (local copy: `data/phising.csv`)
- **Official related UCI page:** https://archive.ics.uci.edu/dataset/327/phishing+websites
- **Newer dataset (2020), added for the resubmission's cross-dataset generalization check:** Vrbančič, Fister Jr. & Podgorelec, *Data in Brief* (2020) — full 58,645-row `dataset_small.csv` copy included locally as `data/newer_phishing_dataset_2020.csv`; see `data/README_newer_dataset.md` for provenance and refresh instructions.

## Repository structure

```
├── data/                          # datasets + provenance notes
├── src/                           # tested, importable Python package (the actual pipeline)
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py           # includes duplicate-safe group-key construction
│   ├── models.py
│   ├── evaluation.py
│   └── experiments.py             # original-style / deduplicated / grouped splits, CV, newer-dataset check
├── scripts/
│   └── run_all_experiments.py     # single entry point -> writes results/*.csv and figures/*.png
├── tests/                         # pytest unit tests, including a grouped-split leakage guarantee test
├── results/                       # committed CSV outputs of every experiment
├── figures/                       # committed PNG figures
├── notebooks/
│   └── phishing_detection_reproduction.ipynb   # narrative EDA notebook, extended with the resubmission's new sections
├── report/
│   ├── report_final_with_results.md
│   └── report_final_with_results.pdf
├── requirements.txt
└── README.md
```

## How to run

1. Clone or download this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Reproduce every number and figure in the report with a single command:
   ```bash
   python scripts/run_all_experiments.py
   ```
   This writes fresh CSVs to `results/` and PNGs to `figures/`, overwriting the committed copies with a byte-for-byte reproduction (given the fixed random seed).
4. Run the test suite:
   ```bash
   pytest tests/ -v
   ```
5. (Optional) Open `notebooks/phishing_detection_reproduction.ipynb` in Jupyter/Colab/VS Code for the narrative, cell-by-cell walkthrough — it calls the same `src/` functions as the script, so notebook and script cannot drift apart.

The pipeline first tries to load the local `data/phising.csv` file; if missing, it falls back to the original article's GitHub CSV URL.

## Main original-style results

| Model | Accuracy | Precision | Recall | F1 | F2 | MCC | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Random Forest | 0.9739 | 0.9656 | 0.9883 | 0.9768 | 0.9837 | 0.9474 | 0.9951 | 0.9960 |
| SVM RBF | 0.9674 | 0.9679 | 0.9736 | 0.9708 | 0.9725 | 0.9340 | 0.9935 | 0.9954 |
| Logistic Regression | 0.9479 | 0.9427 | 0.9648 | 0.9536 | 0.9603 | 0.8945 | 0.9931 | 0.9947 |
| Dummy Majority Baseline | 0.5554 | 0.5554 | 1.0000 | 0.7141 | 0.8620 | 0.0000 | 0.5000 | 0.5554 |

Best original-style model: **Random Forest**.

## Random Forest across increasingly strict validation schemes

The dataset contains **740 duplicate rows**, which a random split can place on both sides of train/test. This resubmission adds a **grouped, duplicate-safe split** (`GroupShuffleSplit`/`GroupKFold`, keyed on a stable SHA-256 hash of the full feature row, `src/preprocessing._stable_row_hash`) on top of the earlier deduplicated-split experiment, so no identical feature vector can ever appear in both train and test.

| Random Forest metric | Original-style split | Deduplicated split | Grouped split (single `GroupShuffleSplit`) |
|---|---:|---:|---:|
| Accuracy | 0.9739 | 0.9627 | 0.9787 |
| Recall | 0.9883 | 0.9556 | 0.9765 |
| F1 | 0.9768 | 0.9641 | 0.9809 |
| False negatives | 4 | 10 | 8 |

**A single grouped split is not the authoritative estimate.** While hardening this experiment we found that `GroupShuffleSplit`'s specific train/test assignment is sensitive to how the group key is encoded (it sorts unique groups internally, so a numeric-hash key vs. a hex-string key produces a different split for the same `random_state`) — a fragility worth flagging rather than papering over. The robust, order-independent number is group-aware cross-validation: Random Forest mean F1 falls from **0.9720** (`StratifiedKFold`) to **0.9612** (`GroupKFold`) — a real but modest duplicate-leakage effect. See report Section 6.8 for the full discussion, including how this same fix reversed an earlier (unstable-key) observation that SVM RBF edged out Random Forest under the grouped split — with the stable key, Random Forest wins on every metric.

## Newer (2020) external dataset check

The newer-dataset check now uses the **full 58,645-row 2020 dataset**, not a small sample. Because the 2020 dataset has a different numeric feature schema, the models are retrained from scratch with dataset-appropriate numeric scaling. Random Forest reaches Accuracy = **0.9568**, Precision = **0.9573**, Recall = **0.9602**, F1 = **0.9588**, and PR-AUC = **0.9925** on a 14,662-row test split, clearly ahead of the dummy baseline. Every model's Accuracy and F1 also ships with a bootstrap 95% confidence interval (`src/evaluation.bootstrap_metric_ci`, `results/results_newer_dataset_2020_bootstrap_ci.csv`) — Random Forest's F1 CI is a tight **[0.956, 0.962]**, concretely confirming that the full dataset resolves the uncertainty a small sample would carry. See `data/README_newer_dataset.md` and report Section 6.9 for provenance, feature-schema transfer limitations, and the SVM scalability note.

## Additional critical evaluation (carried over from the first resubmission)

- 3-fold stratified **and** group-aware cross-validation.
- PR-AUC.
- Threshold analysis.
- A simulated low-prevalence phishing scenario.

## Main conclusion

The article's general claim is mostly supported for this dataset: machine learning can classify the selected phishing website dataset with strong performance. However, the grouped, duplicate-safe evaluation shows that the original-style performance was more optimistic than even the first resubmission's deduplication-only check suggested. The project should be treated as a strong educational baseline, not a complete production phishing defense system.
