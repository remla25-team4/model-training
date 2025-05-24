import pytest
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import log_loss, accuracy_score


@pytest.fixture(scope="module")
def count_vectorizer():
    return joblib.load("models/count_vectorizer.joblib")


@pytest.fixture(scope="module")
def model():
    return joblib.load("models/naive_bayes.joblib")


@pytest.fixture(scope="module")
def offline_data():
    return pd.read_csv("datasets/training_dataset.tsv", delimiter='\t')


@pytest.fixture(scope="module")
def online_data():
    return pd.read_csv("datasets/online_dataset.tsv", delimiter='\t')


def test_offline_online_performance_consistency(count_vectorizer, model, offline_data, online_data):
    """
    Ensure offline proxy performance (accuracy & log-loss)
    is close to online performance on a live holdout set.
    """
    # Offline metrics
    X_off = count_vectorizer.transform(offline_data["Review"]).toarray()
    y_off = offline_data["Liked"].values
    proba_off = model.predict_proba(X_off)[:, 1]
    offline_acc = accuracy_score(y_off, model.predict(X_off))
    offline_ll = log_loss(y_off, proba_off)

    # Online metrics
    X_on = count_vectorizer.transform(online_data["Review"]).toarray()
    y_on = online_data["Liked"].values
    proba_on = model.predict_proba(X_on)[:, 1]
    online_acc = accuracy_score(y_on, model.predict(X_on))
    online_ll = log_loss(y_on, proba_on)

    # Accuracy difference within 5%
    assert abs(offline_acc - online_acc) < 0.05, (
        f"Offline vs. online accuracy differ too much: "
        f"{offline_acc:.3f} vs. {online_acc:.3f}"
    )
    # Log-loss difference within 0.1
    assert abs(offline_ll - online_ll) < 0.1, (
        f"Offline vs. online log-loss differ too much: "
        f"{offline_ll:.3f} vs. {online_ll:.3f}"
    )


def test_hyperparameters_tuned(model):
    """
    Ensure hyperparameters were actually tuned.
    For GaussianNB, default var_smoothing is 1e-9; we assert var_smoothing != 1e-9.
    """
    assert hasattr(
        model, "var_smoothing"), "Model missing 'var_smoothing' attribute"
    assert model.var_smoothing != 1e-9, (
        f"var_smoothing is still default (1e-9); please run a hyperparameter search."
    )


def test_simple_baseline_benchmark(count_vectorizer, model, offline_data):
    """
    Confirm model beats a trivial baseline (majority‐class predictor).
    """
    X = count_vectorizer.transform(offline_data["Review"]).toarray()
    y = offline_data["Liked"].values
    majority_class = pd.Series(y).mode()[0]
    baseline_preds = np.full_like(y, majority_class)
    baseline_acc = accuracy_score(y, baseline_preds)
    model_acc = accuracy_score(y, model.predict(X))

    assert model_acc > baseline_acc, (
        f"Model acc ({model_acc:.3f}) not above baseline ({baseline_acc:.3f})"
    )
