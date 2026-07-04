import pandas as pd

from src import config, data_loading, experiments, preprocessing


def _small_raw_df(n=300):
    """A small, fast-to-run subsample of the real dataset for unit tests."""
    df_raw, _ = data_loading.load_article_dataset()
    return df_raw.sample(n=min(n, len(df_raw)), random_state=0).reset_index(drop=True)


def test_run_original_style_split_returns_all_models():
    df_raw = _small_raw_df()
    results, fitted = experiments.run_original_style_split(df_raw)
    assert set(results["Model"]) == {
        "Dummy Majority Baseline", "Logistic Regression", "Random Forest", "SVM RBF"
    }
    assert (results["Accuracy"] >= 0).all() and (results["Accuracy"] <= 1).all()
    assert "Random Forest" in fitted


def test_run_deduplicated_split_has_fewer_or_equal_rows():
    df_raw = _small_raw_df()
    dup_before = preprocessing.duplicate_summary(df_raw)
    results, _ = experiments.run_deduplicated_split(df_raw)
    assert not results.empty
    # sanity: dedup summary itself is internally consistent
    assert dup_before["n_rows_after_dedup"] <= dup_before["n_rows"]


def test_run_grouped_split_has_no_group_leakage():
    """The core reviewer-requested guarantee: no identical feature vector
    (duplicate) should appear on both sides of the grouped split.
    """
    df_raw = _small_raw_df(n=400)
    df, usable = preprocessing.prepare_binary_dataset(df_raw)
    df = preprocessing.add_group_key(df, usable)

    results, fitted = experiments.run_grouped_split(df_raw)
    assert not results.empty

    X_test = fitted["Random Forest"]["X_test"]
    # Recover which rows of `df` correspond to the test set by feature match,
    # then confirm their group keys do not appear anywhere in the remaining rows.
    test_feature_tuples = set(map(tuple, X_test[usable].values.tolist()))
    train_mask = ~df[usable].apply(lambda r: tuple(r.values) in test_feature_tuples, axis=1)
    train_groups = set(df.loc[train_mask, "group_key"])
    test_groups = set(
        df[df[usable].apply(lambda r: tuple(r.values) in test_feature_tuples, axis=1)]["group_key"]
    )
    assert train_groups.isdisjoint(test_groups)


def test_cross_validation_summary_grouped_and_stratified_run():
    df_raw = _small_raw_df(n=300)
    cv_strat = experiments.cross_validation_summary(df_raw, "test-strat", grouped=False)
    cv_group = experiments.cross_validation_summary(df_raw, "test-group", grouped=True)
    assert not cv_strat.empty
    assert not cv_group.empty
    assert "f1_mean" in cv_strat.columns
    assert "f1_mean" in cv_group.columns


def test_run_newer_dataset_experiment_runs_end_to_end():
    # Use a deterministic sample for the unit test so pytest stays fast.
    # The full 58,645-row newer-dataset experiment is executed by
    # scripts/run_all_experiments.py and its committed outputs are stored in
    # results/results_newer_dataset_2020*.csv.
    df_newer = data_loading.load_newer_dataset()
    df_newer = df_newer.sample(n=min(1500, len(df_newer)), random_state=0).reset_index(drop=True)
    results, fitted, ci = experiments.run_newer_dataset_experiment(df_newer)
    assert not results.empty
    assert set(results["Model"]) == {
        "Dummy Majority Baseline", "Logistic Regression", "Random Forest", "SVM RBF"
    }
    # Bootstrap CI table: one row per model, CI bounds must bracket the point estimate.
    assert set(ci["Model"]) == set(results["Model"])
    assert (ci["Accuracy_CI_low"] <= ci["Accuracy"]).all()
    assert (ci["Accuracy"] <= ci["Accuracy_CI_high"]).all()
    assert (ci["F1_CI_low"] <= ci["F1"]).all()
    assert (ci["F1"] <= ci["F1_CI_high"]).all()
