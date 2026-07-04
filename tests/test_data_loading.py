import os

import pytest

from src import config, data_loading


def test_load_article_dataset_shape_and_columns():
    df, source = data_loading.load_article_dataset()
    assert df.shape[1] == len(config.ARTICLE_COLUMNS)
    assert list(df.columns) == config.ARTICLE_COLUMNS
    assert df.shape[0] > 0
    assert "Local CSV" in source or "GitHub" in source


def test_load_article_dataset_target_values():
    df, _ = data_loading.load_article_dataset()
    unique_targets = set(df["target"].dropna().unique().tolist())
    # article dataset target is coded as -1 (phishing) / 1 (legitimate)
    assert unique_targets.issubset({-1.0, 1.0})


def test_load_article_dataset_raises_on_missing_file(tmp_path):
    bad_path = os.path.join(str(tmp_path), "does_not_exist.csv")
    with pytest.raises(Exception):
        data_loading.load_article_dataset(
            local_path=bad_path, url="https://example.invalid/not-a-real-url.csv"
        )


def test_load_newer_dataset():
    df = data_loading.load_newer_dataset()
    assert df.shape[0] > 0
    assert "phishing" in df.columns


def test_load_newer_dataset_missing_file_raises(tmp_path):
    bad_path = os.path.join(str(tmp_path), "missing.csv")
    with pytest.raises(FileNotFoundError):
        data_loading.load_newer_dataset(path=bad_path)
