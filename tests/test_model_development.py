"""Tests for model development"""

import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score


def test_offline_online_performance_consistency(cv, trained_model, off_data, on_data):
    """
    Ensure offline proxy performance (accuracy & log-loss)
    is close to online performance on a live holdout set.
    """
    # Offline metrics
    x_off = cv.transform(off_data["Review"]).toarray()
    y_off = off_data["Liked"].values
    proba_off = trained_model.predict_proba(x_off)[:, 1]
    offline_acc = accuracy_score(y_off, trained_model.predict(x_off))
    offline_ll = log_loss(y_off, proba_off)

    # Online metrics
    x_on = cv.transform(on_data["Review"]).toarray()
    y_on = on_data["Liked"].values
    proba_on = trained_model.predict_proba(x_on)[:, 1]
    online_acc = accuracy_score(y_on, trained_model.predict(x_on))
    online_ll = log_loss(y_on, proba_on)

    # Accuracy difference within 5%
    assert abs(offline_acc - online_acc) < 0.05, (
        "Offline vs. online accuracy differ too much: "+
        f"{offline_acc:.3f} vs. {online_acc:.3f}"
    )
    # Log-loss difference within 0.1
    assert abs(offline_ll - online_ll) < 0.1, (
        "Offline vs. online log-loss differ too much: "+
        f"{offline_ll:.3f} vs. {online_ll:.3f}"
    )


def test_hyperparameters_tuned(trained_model):
    """
    Ensure hyperparameters were actually tuned.
    For GaussianNB, default var_smoothing is 1e-9; we assert var_smoothing != 1e-9.
    """
    assert hasattr(
        trained_model, "var_smoothing"), "Model missing 'var_smoothing' attribute"
    assert trained_model.var_smoothing != 1e-9, (
        "var_smoothing is still default (1e-9); please run a hyperparameter search."
    )


def test_simple_baseline_benchmark(cv, trained_model, off_data):
    """
    Confirm model beats a trivial baseline (majority‐class predictor).
    """
    x = cv.transform(off_data["Review"]).toarray()
    y = off_data["Liked"].values
    majority_class = pd.Series(y).mode()[0]
    baseline_preds = np.full_like(y, majority_class)
    baseline_acc = accuracy_score(y, baseline_preds)
    model_acc = accuracy_score(y, trained_model.predict(x))

    assert model_acc > baseline_acc, (
        f"Model acc ({model_acc:.3f}) not above baseline ({baseline_acc:.3f})"
    )
