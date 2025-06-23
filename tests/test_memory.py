"""Test for model memory usage during predictions"""

import os
import psutil
import pytest

def get_memory_usage():
    """Return current process memory usage in MB"""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024

@pytest.mark.monitoring_test
@pytest.mark.monitor_6
def test_model_memory_usage(trained_model, sample_input):
    """Check that model.predict() does not exceed memory threshold. Monitor 6"""
    model = trained_model
    sample_data = sample_input  # Define a fixture from x_train (check conftest.py)

    mem_before = get_memory_usage()
    model.predict(sample_data)
    mem_after = get_memory_usage()

    mem_used = mem_after - mem_before
    assert mem_used < 500, f"Memory used too high: {mem_used:.2f} MB"
