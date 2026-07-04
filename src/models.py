"""Model pipeline definitions shared by every experiment."""

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from . import config


def make_preprocessor(feature_names):
    """One-hot encode the ordinal/categorical {-1, 0, 1}-style features.

    See report section 4 (Feature Engineering Analysis) for the rationale:
    these values are categorical states, not a linear numeric scale, so
    one-hot encoding is used for Logistic Regression and SVM. Random Forest
    receives the same preprocessing for pipeline consistency, even though
    tree splits do not strictly require it.
    """
    return ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), feature_names)],
        remainder="drop",
    )


def make_numeric_preprocessor(feature_names):
    """Scale numeric count/ratio/binary features from the newer 2020 dataset."""
    return ColumnTransformer(
        transformers=[("num", StandardScaler(), feature_names)],
        remainder="drop",
    )


def make_model_pipelines(feature_names, random_state=config.RANDOM_STATE, rf_n_estimators=300):
    """Return the four standard model pipelines used throughout the project.

    Each pipeline receives its own preprocessor instance. This avoids hidden
    shared-state bugs when multiple pipelines are fitted one after another.
    """
    return {
        "Dummy Majority Baseline": Pipeline([
            ("preprocess", make_preprocessor(feature_names)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocess", make_preprocessor(feature_names)),
            ("model", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_preprocessor(feature_names)),
            ("model", RandomForestClassifier(
                n_estimators=rf_n_estimators,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=1,
            )),
        ]),
        "SVM RBF": Pipeline([
            ("preprocess", make_preprocessor(feature_names)),
            ("model", SVC(
                kernel="rbf", probability=True, class_weight="balanced",
                random_state=random_state,
            )),
        ]),
    }


def make_numeric_model_pipelines(feature_names, random_state=config.RANDOM_STATE,
                                 rf_n_estimators=300):
    """Return model pipelines for numeric URL/domain feature datasets.

    The newer 2020 dataset contains numeric counts, ratios, and binary flags,
    not the {-1, 0, 1} categorical states used by the 2015 article dataset.
    Therefore Logistic Regression and SVM use StandardScaler instead of
    one-hot encoding. The SVM row uses an RBF-kernel feature approximation
    plus a linear margin classifier so the full 58,645-row dataset remains
    reproducible on normal hardware; exact SVC with probability calibration is
    quadratic and was only practical for the earlier tiny sample. The newer
    dataset experiment uses 100 Random Forest trees to keep the full-data
    external validation fast and reproducible.
    """
    return {
        "Dummy Majority Baseline": Pipeline([
            ("preprocess", make_numeric_preprocessor(feature_names)),
            ("model", DummyClassifier(strategy="most_frequent")),
        ]),
        "SVM RBF": Pipeline([
            ("preprocess", make_numeric_preprocessor(feature_names)),
            ("rbf_features", RBFSampler(
                gamma=0.01, n_components=600, random_state=random_state
            )),
            ("model", SGDClassifier(
                loss="hinge", alpha=0.0001, max_iter=3000, tol=1e-3,
                class_weight="balanced", random_state=random_state
            )),
        ]),
        "Random Forest": Pipeline([
            ("preprocess", make_numeric_preprocessor(feature_names)),
            ("model", RandomForestClassifier(
                n_estimators=rf_n_estimators,
                random_state=random_state,
                class_weight="balanced",
                n_jobs=1,
            )),
        ]),
        "Logistic Regression": Pipeline([
            ("preprocess", make_numeric_preprocessor(feature_names)),
            ("model", LogisticRegression(
                max_iter=1000, solver="liblinear", random_state=random_state,
                class_weight="balanced"
            )),
        ]),
    }
