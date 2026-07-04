"""Loading routines for the original article dataset and the newer external dataset."""

import os
import pandas as pd

from . import config


def load_article_dataset(local_path=None, url=None):
    """Load the 2015-era phishing dataset used by the reproduced article.

    Tries the local CSV copy first (for offline reproducibility), then falls
    back to the original GitHub URL used by the article.

    Returns
    -------
    df : pandas.DataFrame
        Raw dataframe with columns named per config.ARTICLE_COLUMNS.
    source : str
        Human-readable description of where the data came from.
    """
    local_path = local_path or config.LOCAL_DATA_PATH
    url = url or config.ARTICLE_DATA_URL

    if os.path.exists(local_path):
        source = f"Local CSV copy included in this repository ({local_path})"
        df = pd.read_csv(local_path, header=None)
    else:
        source = f"Original article GitHub CSV ({url})"
        df = pd.read_csv(url, header=None)

    if df.shape[1] != len(config.ARTICLE_COLUMNS):
        raise ValueError(
            f"Expected {len(config.ARTICLE_COLUMNS)} columns, got {df.shape[1]}. "
            "The dataset schema may have changed."
        )
    df.columns = config.ARTICLE_COLUMNS

    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df, source


def load_newer_dataset(path=None):
    """Load the newer (2020) external phishing dataset used for the
    cross-dataset generalization check requested in the reviewer feedback.

    This dataset (Vrbancic, Fister Jr. & Podgorelec, 2020, "Datasets for
    Phishing Websites Detection", Data in Brief) uses a completely different,
    URL/domain/certificate-derived feature schema (111 features) than the
    2015 article dataset (30 categorical features). Because the feature
    schemas do not overlap, the pretrained models cannot be applied directly;
    instead the same modeling *methodology* is re-run on this dataset so the
    two experiments are comparable.

    The full `dataset_small.csv` file is bundled locally so the
    notebook/tests run fully offline. See data/README_newer_dataset.md for
    provenance and refresh instructions.
    """
    path = path or config.NEWER_DATA_PATH
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Newer dataset not found at {path}. See "
            "data/README_newer_dataset.md for how to obtain it."
        )
    df = pd.read_csv(path)
    return df
