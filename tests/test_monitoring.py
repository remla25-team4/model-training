import pytest
import joblib
import pandas as pd
import os
import time
from datetime import datetime, timedelta


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


def test_serving_feature_distribution_stability(offline_data, online_data, max_allowed_drift=0.25):
    """
    Monitor 2: Data invariants hold for inputs.
    Checks for significant drift in a basic feature characteristic (e.g. review length)
    between training and serving data.
    """
    offline_review_lengths = offline_data["Review"].astype(str).apply(len)
    online_review_lengths = online_data["Review"].astype(str).apply(len)

    mean_offline_length = offline_review_lengths.mean()
    mean_online_length = online_review_lengths.mean()

    print("Mean of offline review lengths:", mean_offline_length)
    print("Mean of online review lengths:", mean_online_length)

    # currently allow for up to a 25% change in mean review length
    assert abs(mean_online_length - mean_offline_length) / mean_offline_length < max_allowed_drift, \
        (f"Significant change in mean review length detected. "
         f"Training data mean: {mean_offline_length:.2f}, Serving data mean: {mean_online_length:.2f}")


def test_model_staleness(model_path="models/naive_bayes.joblib", max_days_old=30):
    """
    Monitor 4: Models are not too stale.
    Checks if the model file is not older than 30 days.
    """
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    model_mod_time = os.path.getmtime(model_path)
    model_age = datetime.now() - datetime.fromtimestamp(model_mod_time)

    print(f"Age: {model_age.days} day(s)")

    assert model_age <= timedelta(days=max_days_old), \
        (f"Model file at {model_path} is older than {max_days_old} days. "
         f"Age: {model_age.days} days. Consider retraining or redeploying.")

def test_prediction_latency(count_vectorizer, model, online_data, max_latency_per_100_samples=10):
    """
    Monitor 6: Computing performance of the model has not regressed over time
    Checks if prediction latency for a batch of serving data is within an acceptable threshold (10ms).
    """
    if online_data.empty:
        pytest.skip("Online data is empty, skipping latency test.")

    sample_data = online_data.sample(min(1000, len(online_data)), random_state=42)
    X_sample = count_vectorizer.transform(sample_data["Review"]).toarray()

    start_time = time.time()
    model.predict(X_sample)
    end_time = time.time()

    duration = end_time - start_time
    num_samples = len(X_sample)
    avg_latency_per_100 = (duration * 1000 / num_samples) * 100 if num_samples > 0 else 0

    print(f'Average latency per 100 predictions: {avg_latency_per_100:.4f} ms')

    assert avg_latency_per_100 < max_latency_per_100_samples, \
        (f"Prediction latency ({avg_latency_per_100:.4f}ms per 100 samples) "
         f"exceeds threshold ({max_latency_per_100_samples}ms per 100 samples).")


