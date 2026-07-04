"""Target conversion, duplicate handling, and group-key construction.

The reviewer's key methodological request was to add a *grouped* validation
scheme so that identical feature vectors (duplicate rows) cannot appear in
both the train and the test split. A random split can place duplicate rows
on both sides of the split, letting the model "memorize" a row it has
literally already seen at test time, which inflates every metric.

We address this with a group key equal to a stable content hash of the
full feature row. Rows that are exact duplicates of each other (i.e.
identical feature vectors) receive the same group id, and scikit-learn's
GroupKFold / GroupShuffleSplit machinery is used so that an entire group
always lands entirely in train or entirely in test.

Note on the hashing implementation: Python's built-in `hash()` is not used
here on purpose. For strings, `hash()` is randomized per-process (salted
by `PYTHONHASHSEED`) unless that seed is fixed, so group ids built from it
would not be reproducible across runs/machines even though grouping
*within* a single run is still internally consistent. `hash()` also
truncates to a machine-width integer, which is an avoidable, if small,
extra collision risk for an operation whose entire job is exact-duplicate
detection. We instead hash a canonical, deterministic string
representation of each row with `hashlib.sha256`, which is stable across
processes/machines and does not depend on `PYTHONHASHSEED`.

Also note (see report Section 6.8 for the full discussion): scikit-learn's
`GroupShuffleSplit` sorts unique groups internally before applying its
`random_state`-controlled shuffle, so the *specific* single split it
produces can depend on the sort order of the group-key representation
(e.g. numeric hash vs. hex string), even though the set of groups and the
no-leakage guarantee are unaffected. Because of this, this project treats
group-aware cross-validation (`GroupKFold`, averaged over folds) rather
than a single `GroupShuffleSplit` draw as the authoritative "grouped"
estimate.
"""

import hashlib

import pandas as pd

from . import config


def to_binary_target(df_raw):
    """Convert the raw {-1, 1} target column into a 0/1 `is_phishing` column.

    In the UCI/article schema, -1 conventionally denotes phishing and 1
    denotes legitimate.
    """
    df = df_raw.copy()
    df[config.TARGET_COLUMN] = (df["target"] == -1).astype(int)
    df = df.drop(columns=["target"])
    return df


def get_feature_columns(df, target_column=config.TARGET_COLUMN):
    return [c for c in df.columns if c != target_column]


def drop_single_value_columns(df, feature_columns):
    """Return feature columns with zero variance removed."""
    return [c for c in feature_columns if df[c].nunique(dropna=False) > 1]


def _stable_row_hash(row):
    """Deterministic, collision-resistant hash of a row's feature values.

    Uses a canonical string join (fixed separator, `repr()` of each value
    so floats/ints/strings are unambiguous) fed through SHA-256, rather
    than Python's built-in `hash()`. This is reproducible across processes
    and machines and gives a 256-bit digest instead of a truncated
    machine-width int, which matters here because the whole point of the
    key is exact-duplicate identification - a hash collision would
    silently merge two distinct rows into one group.
    """
    canonical = "\x1f".join(repr(v) for v in row.values)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def add_group_key(df, feature_columns):
    """Add a `group_key` column: rows with identical feature vectors share
    the same group id. Used for duplicate-aware (grouped) splitting.
    """
    df = df.copy()
    df["group_key"] = df[feature_columns].apply(_stable_row_hash, axis=1)
    return df


def deduplicate(df):
    """Drop exact duplicate rows (all columns, including target)."""
    return df.drop_duplicates().reset_index(drop=True)


def prepare_binary_dataset(df_raw):
    """Full prep: raw article dataframe -> binary target + usable feature list."""
    df = to_binary_target(df_raw)
    feature_columns = get_feature_columns(df)
    usable = drop_single_value_columns(df, feature_columns)
    return df, usable


def duplicate_summary(df_raw):
    """Diagnostics used in the EDA / reproducibility sections of the report."""
    n_rows = len(df_raw)
    n_duplicates = int(df_raw.duplicated().sum())
    n_unique = n_rows - n_duplicates
    return {
        "n_rows": n_rows,
        "n_duplicate_rows": n_duplicates,
        "n_rows_after_dedup": n_unique,
    }
