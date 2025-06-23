"""Performance test for model inference time"""

import time
import pytest

@pytest.mark.monitoring_test
@pytest.mark.monitor_6
def test_model_inference_time(trained_model, sample_input):
    """Check that model.predict() runs under acceptable time threshold"""
    model = trained_model
    input_sample = sample_input

    start_time = time.time()
    model.predict(input_sample)
    end_time = time.time()

    inference_time = end_time - start_time

    assert inference_time < 4, f"Inference took too long: {inference_time:.2f} seconds"
