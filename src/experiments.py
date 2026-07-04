"""End-to-end experiment runners.

Each function returns plain pandas DataFrames / dicts so that scripts,
notebooks, and tests can all consume the same logic without duplicating it.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, make_scorer, matthews_corrcoef
from sklearn.model_selection import (
    GroupKFold, GroupShuffleSplit, StratifiedKFold, cross_validate,
    train_test_split,
)

from . import config, evaluation, models, preprocessing


def _fit_and_score_all_models(X_train, X_test, y_train, y_test, usable_features,
                               experiment_name, rf_n_estimators=300):
    rows = []
    fitted = {}
    pipelines = models.make_model_pipelines(usable_features, rf_n_estimators=rf_n_estimators)
    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluation.evaluate_pipeline(pipeline, X_test, y_test)
        rows.append({"Experiment": experiment_name, "Model": model_name, **metrics})
        y_score = evaluation.get_positive_class_score(pipeline, X_test)
        fitted[model_name] = {
            "model": pipeline, "X_test": X_test, "y_test": y_test, "y_score": y_score,
        }
    results_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return results_df, fitted


def run_original_style_split(df_raw, test_size=config.TEST_SIZE,
                              random_state=config.RANDOM_STATE):
    """Reproduces the article's methodology: a single stratified random
    train/test split, duplicates included (i.e. NOT deduplicated,
    NOT group-aware). Kept as the baseline "as the article did it" result.
    """
    df, usable = preprocessing.prepare_binary_dataset(df_raw)
    X = df[usable]
    y = df[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return _fit_and_score_all_models(X_train, X_test, y_train, y_test, usable,
                                      "Original-style split (random, with duplicates)")


def run_deduplicated_split(df_raw, test_size=config.TEST_SIZE,
                            random_state=config.RANDOM_STATE):
    """Removes exact duplicate rows *before* splitting, then does a plain
    random split on the deduplicated data. This still does not guarantee
    that near-duplicate feature vectors are separated if duplicates were
    already removed - see run_grouped_split for the stronger guarantee.
    """
    df_dedup_raw = preprocessing.deduplicate(df_raw)
    df, usable = preprocessing.prepare_binary_dataset(df_dedup_raw)
    X = df[usable]
    y = df[config.TARGET_COLUMN]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return _fit_and_score_all_models(X_train, X_test, y_train, y_test, usable,
                                      "Deduplicated split (random)")


def run_grouped_split(df_raw, test_size=config.TEST_SIZE,
                       random_state=config.RANDOM_STATE):
    """Reviewer-requested addition: a *grouped* train/test split on the
    ORIGINAL (non-deduplicated) data, where the group key is the exact
    feature-vector hash. GroupShuffleSplit guarantees that all rows sharing
    an identical feature vector (i.e. duplicates of each other) end up
    entirely in train or entirely in test - so no row's exact twin can leak
    from train into test, without discarding any rows the way plain
    deduplication does.
    """
    df, usable = preprocessing.prepare_binary_dataset(df_raw)
    df = preprocessing.add_group_key(df, usable)
    X = df[usable]
    y = df[config.TARGET_COLUMN]
    groups = df["group_key"]

    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    # Sanity check: no group should appear on both sides.
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups & test_groups
    assert len(overlap) == 0, "Group leakage detected between train and test!"

    return _fit_and_score_all_models(X_train, X_test, y_train, y_test, usable,
                                      "Grouped split (duplicate-safe, GroupShuffleSplit)")


def cross_validation_summary(df_raw, label, n_splits=config.N_CV_SPLITS,
                              random_state=config.RANDOM_STATE, grouped=False):
    """Stratified (or group-aware) K-fold cross-validation summary for
    Logistic Regression and Random Forest.

    grouped=True uses GroupKFold on the feature-vector hash instead of
    StratifiedKFold, so duplicate rows never split across folds either.
    """
    df, usable = preprocessing.prepare_binary_dataset(df_raw)
    X = df[usable]
    y = df[config.TARGET_COLUMN]

    cv_scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "f2": make_scorer(fbeta_score, beta=2),
        "mcc": make_scorer(matthews_corrcoef),
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
    }

    model_dict = models.make_model_pipelines(usable, rf_n_estimators=30)
    selected_models = ["Logistic Regression", "Random Forest"]

    if grouped:
        df = preprocessing.add_group_key(df, usable)
        groups = df["group_key"]
        cv = GroupKFold(n_splits=n_splits)
        cv_args = dict(groups=groups)
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        cv_args = {}

    rows = []
    for model_name in selected_models:
        cv_scores = cross_validate(
            model_dict[model_name], X, y, cv=cv, scoring=cv_scoring, n_jobs=1, **cv_args
        )
        row = {"Dataset": label, "Model": model_name}
        for metric in cv_scoring:
            row[f"{metric}_mean"] = float(cv_scores[f"test_{metric}"].mean())
            row[f"{metric}_std"] = float(cv_scores[f"test_{metric}"].std())
        rows.append(row)
    return pd.DataFrame(rows)


def run_newer_dataset_experiment(df_newer, target_column="phishing",
                                  test_size=config.TEST_SIZE,
                                  random_state=config.RANDOM_STATE):
    """Reviewer-requested addition: re-run the SAME modeling methodology
    (same four pipelines, same split strategy, same metric set) on a newer
    (2020) phishing dataset with a different feature schema, as a
    cross-dataset / temporal-generalization sanity check.

    IMPORTANT: because the feature schemas do not overlap between the 2015
    article dataset and this 2020 dataset, the already-trained models
    CANNOT be transferred directly. This function instead retrains fresh
    instances of the same four model types on the newer data, so the
    comparison is about whether the same *methodology* still works well on
    more recent phishing patterns, not about applying old model weights to
    new data.
    """
    df = df_newer.copy()
    feature_columns = [c for c in df.columns if c != target_column]
    usable = preprocessing.drop_single_value_columns(df, feature_columns)
    X = df[usable]
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    rows = []
    fitted = {}
    pipelines = models.make_numeric_model_pipelines(usable, rf_n_estimators=100)
    for model_name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluation.evaluate_pipeline(pipeline, X_test, y_test)
        rows.append({
            "Experiment": "Newer dataset (2020 full dataset, different numeric schema)",
            "Model": model_name,
            **metrics,
        })
        fitted[model_name] = {
            "model": pipeline,
            "X_test": X_test,
            "y_test": y_test,
            "y_score": evaluation.get_positive_class_score(pipeline, X_test),
        }
    results_df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)

    # Attach a bootstrap CI for Accuracy and F1 for every model. With the
    # full 58,645-row dataset (a ~14,662-row test split) this is expected to
    # be a tight band around the point estimate - which is itself a useful,
    # concrete confirmation that using the full dataset (rather than an
    # earlier small sample) actually resolved small-sample uncertainty,
    # rather than just asserting that it did.
    from sklearn.metrics import accuracy_score, f1_score as _f1_score

    ci_rows = []
    for model_name, info in fitted.items():
        y_pred = info["model"].predict(info["X_test"])
        acc_ci = evaluation.bootstrap_metric_ci(info["y_test"], y_pred, accuracy_score,
                                                 random_state=random_state)
        f1_ci = evaluation.bootstrap_metric_ci(
            info["y_test"], y_pred, lambda yt, yp: _f1_score(yt, yp, zero_division=0),
            random_state=random_state,
        )
        ci_rows.append({
            "Model": model_name,
            "Accuracy": acc_ci["point_estimate"],
            "Accuracy_CI_low": acc_ci["ci_low"],
            "Accuracy_CI_high": acc_ci["ci_high"],
            "F1": f1_ci["point_estimate"],
            "F1_CI_low": f1_ci["ci_low"],
            "F1_CI_high": f1_ci["ci_high"],
            "n_test_rows": acc_ci["n_samples"],
        })
    ci_df = pd.DataFrame(ci_rows)

    return results_df, fitted, ci_df
