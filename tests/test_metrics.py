import pytest

from traceguard.metrics import binary_metrics


def test_binary_metrics_perfect_predictions():
    metrics = binary_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["accuracy"] == 1.0
    assert metrics["balanced_accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["false_negative_rate"] == 0.0


def test_auc_gives_half_credit_to_ties():
    metrics = binary_metrics([0, 1], [0.5, 0.5])
    assert metrics["roc_auc"] == pytest.approx(0.5)


def test_metrics_reject_mismatched_inputs():
    with pytest.raises(ValueError):
        binary_metrics([0, 1], [0.2])
