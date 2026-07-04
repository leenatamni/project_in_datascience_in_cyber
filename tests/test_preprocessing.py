import pandas as pd

from src import data_loading, preprocessing


def test_to_binary_target_maps_minus_one_to_phishing():
    df_raw = pd.DataFrame({"a": [1, 2, 3], "target": [-1, 1, -1]})
    df = preprocessing.to_binary_target(df_raw)
    assert list(df["is_phishing"]) == [1, 0, 1]
    assert "target" not in df.columns


def test_get_feature_columns_excludes_target():
    df = pd.DataFrame({"a": [1], "b": [2], "is_phishing": [1]})
    cols = preprocessing.get_feature_columns(df)
    assert set(cols) == {"a", "b"}


def test_drop_single_value_columns():
    df = pd.DataFrame({"a": [1, 1, 1], "b": [1, 2, 3]})
    usable = preprocessing.drop_single_value_columns(df, ["a", "b"])
    assert usable == ["b"]


def test_duplicate_summary_counts_correctly():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    summary = preprocessing.duplicate_summary(df)
    assert summary["n_rows"] == 3
    assert summary["n_duplicate_rows"] == 1
    assert summary["n_rows_after_dedup"] == 2


def test_deduplicate_removes_exact_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2], "b": [1, 1, 2]})
    deduped = preprocessing.deduplicate(df)
    assert len(deduped) == 2


def test_group_key_shared_by_identical_feature_rows():
    df = pd.DataFrame({
        "a": [1, 1, 2, 2],
        "b": [5, 5, 6, 7],
        "is_phishing": [1, 0, 1, 0],  # target differs, features identical for rows 0/1
    })
    df_with_groups = preprocessing.add_group_key(df, ["a", "b"])
    # rows 0 and 1 have identical (a, b) -> same group, regardless of target
    assert df_with_groups.loc[0, "group_key"] == df_with_groups.loc[1, "group_key"]
    # rows 2 and 3 differ in b -> different groups
    assert df_with_groups.loc[2, "group_key"] != df_with_groups.loc[3, "group_key"]


def test_stable_row_hash_is_deterministic_across_calls():
    # Same feature values, called twice (simulating two separate process runs) -> same hash.
    row_a = pd.Series([1, -1, 0.5, "x"])
    row_b = pd.Series([1, -1, 0.5, "x"])
    assert preprocessing._stable_row_hash(row_a) == preprocessing._stable_row_hash(row_b)


def test_stable_row_hash_differs_for_different_rows():
    row_a = pd.Series([1, -1, 0.5])
    row_b = pd.Series([1, -1, 0.6])
    assert preprocessing._stable_row_hash(row_a) != preprocessing._stable_row_hash(row_b)


def test_stable_row_hash_is_not_python_builtin_hash():
    # Guards against regressing back to hash(tuple(...)), which is randomized
    # across processes for strings via PYTHONHASHSEED and truncates to a
    # machine-width int.
    row = pd.Series([1, -1, 0.5])
    digest = preprocessing._stable_row_hash(row)
    assert isinstance(digest, str)
    assert len(digest) == 64  # SHA-256 hex digest length
    assert digest != hash(tuple(row.values))


def test_prepare_binary_dataset_on_real_data_has_no_target_leak():
    df_raw, _ = data_loading.load_article_dataset()
    df, usable = preprocessing.prepare_binary_dataset(df_raw)
    assert "target" not in df.columns
    assert "is_phishing" not in usable
