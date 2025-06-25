"""Monitors for model and data quality in the ML pipeline."""

import os
import time
from datetime import datetime, timedelta
import pytest

@pytest.mark.monitoring_test
@pytest.mark.monitor_2
def test_serving_feature_distribution_stability(
    off_data, on_data, max_allowed_drift=0.25
):
    """
    Monitor 2: Data invariants hold for inputs.
    Checks for significant drift in a basic feature characteristic (e.g. review length)
    between training and serving data.
    """
    offline_review_lengths = off_data["Review"].astype(str).apply(len)
    online_review_lengths = on_data["Review"].astype(str).apply(len)

    mean_offline_length = offline_review_lengths.mean()
    mean_online_length = online_review_lengths.mean()

    print("Mean of offline review lengths:", mean_offline_length)
    print("Mean of online review lengths:", mean_online_length)

    drift = abs(mean_online_length - mean_offline_length) / mean_offline_length
    assert drift < max_allowed_drift, (
        f"Significant change in mean review length detected. "
        f"Training: {mean_offline_length:.2f}, Serving: {mean_online_length:.2f}, "
        f"Drift: {drift:.2%}"
    )


@pytest.mark.monitoring_test
@pytest.mark.monitor_4
def test_model_staleness(model_path="models/naive_bayes.joblib", max_days_old=30):
    """
    Monitor 4: Models are not too stale.
    Checks if the model file is not older than 30 days.
    """
    assert os.path.exists(model_path), f"Model file not found at {model_path}"

    model_mod_time = os.path.getmtime(model_path)
    model_age = datetime.now() - datetime.fromtimestamp(model_mod_time)

    print(f"Age: {model_age.days} day(s)")

    assert model_age <= timedelta(days=max_days_old), (
        f"Model file at {model_path} is older than {max_days_old} days. "
        f"Age: {model_age.days} days. Consider retraining or redeploying."
    )

@pytest.mark.monitoring_test
@pytest.mark.monitor_6
def test_prediction_latency(
    cv, trained_model, on_data, max_latency_per_100_samples=10
):
    """
    Monitor 6: Computing performance
    of the model has not regressed over time
    Checks if prediction latency for a batch
     of serving data is within an acceptable threshold (10ms).
    """
    if on_data.empty:
        pytest.skip("Online data is empty, skipping latency test.")

    sample_data = on_data.sample(min(1000, len(on_data)), random_state=42)
    x_sample = cv.transform(sample_data["Review"]).toarray()

    start_time = time.time()
    trained_model.predict(x_sample)
    end_time = time.time()

    duration = end_time - start_time
    num_samples = len(x_sample)
    avg_latency_per_100 = (duration * 1000 / num_samples) * 100 if num_samples > 0 else 0

    print(f'Average latency per 100 predictions: {avg_latency_per_100:.4f} ms')

    assert avg_latency_per_100 < max_latency_per_100_samples, (
        f"Prediction latency ({avg_latency_per_100:.4f}ms per 100 samples) "
        f"exceeds threshold ({max_latency_per_100_samples}ms per 100 samples)."
    )
