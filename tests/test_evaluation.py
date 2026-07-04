import numpy as np

from src import evaluation


def test_score_predictions_perfect_classifier():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 0, 1, 1]
    y_score = [0.1, 0.2, 0.9, 0.8]
    metrics = evaluation.score_predictions(y_true, y_pred, y_score)
    assert metrics["Accuracy"] == 1.0
    assert metrics["Precision"] == 1.0
    assert metrics["Recall"] == 1.0
    assert metrics["FP"] == 0
    assert metrics["FN"] == 0


def test_score_predictions_all_wrong():
    y_true = [0, 0, 1, 1]
    y_pred = [1, 1, 0, 0]
    metrics = evaluation.score_predictions(y_true, y_pred)
    assert metrics["Accuracy"] == 0.0
    assert metrics["FN"] == 2
    assert metrics["FP"] == 2
    assert np.isnan(metrics["ROC_AUC"])  # no y_score provided


def test_threshold_table_lower_threshold_catches_more_positives():
    y_true = [0, 0, 1, 1, 1]
    y_score = [0.2, 0.4, 0.35, 0.6, 0.9]
    table = evaluation.threshold_table(y_true, y_score, thresholds=(0.3, 0.5, 0.8))
    # recall should be monotonically non-increasing as threshold rises
    recalls = table.sort_values("Threshold")["Recall"].tolist()
    assert all(recalls[i] >= recalls[i + 1] for i in range(len(recalls) - 1))


def test_simulate_low_prevalence_matches_requested_positive_count():
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_score = rng.uniform(0, 1, size=100)
    y_score[y_true == 1] += 0.3  # give positives a higher score on average
    table = evaluation.simulate_low_prevalence(y_true, y_score, n_positive=5)
    assert (table["Phishing count"] == 5).all()
    assert (table["Sample size"] == 55).all()  # 50 negatives + 5 positives
