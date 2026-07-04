# Newer external dataset (full-data cross-dataset generalization check)

**File:** `newer_phishing_dataset_2020.csv`

**Source:** G. Vrbančič, I. Fister Jr., V. Podgorelec (2020), *"Datasets for
Phishing Websites Detection"*, Data in Brief, Vol. 33, 106438.
Repository: https://github.com/GregaVrbancic/Phishing-Dataset
(`dataset_small.csv` variant.)

**Why this dataset:** the reviewer feedback asked for a check against a
newer phishing dataset. This dataset is:
- collected and published in 2020 (5 years after the 2015 article dataset),
- built from a different, larger URL/domain/certificate/DNS-derived feature
  schema (111 features vs. the article's 30 categorical features),
- publicly available and citable.

**Bundled data:** this repository now includes the full `dataset_small.csv`
file locally, not only a sample:

- rows: 58,645
- columns: 112 total = 111 features + `phishing` target
- class distribution: 30,647 phishing and 27,998 legitimate

This means `python scripts/run_all_experiments.py` can run the newer-dataset
check offline and reproduce the committed `results/results_newer_dataset_2020.csv`.

**To refresh the full dataset from the source repository:**
```bash
wget https://raw.githubusercontent.com/GregaVrbancic/Phishing-Dataset/master/dataset_small.csv \
    -O data/newer_phishing_dataset_2020.csv
python scripts/run_all_experiments.py
```

**Why the pretrained 2015 models are not reused directly:** the 2020 dataset's
feature schema does not overlap with the 2015 article dataset's feature schema
(different features entirely, not just different scaling). Applying the original
trained Random Forest directly to this dataset is therefore not meaningful. The
project instead retrains the same model families and evaluates them with the
same metric set on the newer data. This tests whether the general methodology
still works on more recent phishing patterns.

**Preprocessing note:** the 2015 article dataset uses categorical {-1, 0, 1}
feature states, so one-hot encoding is appropriate there. The 2020 dataset uses
numeric counts, ratios, and binary flags, so the external-dataset experiment uses
numeric scaling (`StandardScaler`) instead. This is documented in `src/models.py`
and prevents the newer-dataset check from treating numeric count features as
unordered categories.

**SVM scalability note:** exact probability-calibrated RBF SVC is quadratic in
the number of training samples and is not practical for the full 58,645-row
external check. The newer-dataset SVM row therefore uses an RBF feature-map
approximation plus a linear margin classifier. The main conclusion of the
external check is based on the full-data Random Forest result, which is directly
comparable as a tree-ensemble method.
