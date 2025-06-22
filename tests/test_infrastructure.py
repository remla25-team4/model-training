"""Tests for ML infrastructure via reproducibility"""

import numpy as np
from pipeline.training import train_model, load_processed_training_data

def test_training_reproducibility():
    """
    Test that training the model twice on the same data yields identical predictions.
    This checks for reproducibility (Infra1 from ML Test Score Guide).
    """
    x_data, y_data = load_processed_training_data()

    model_1 = train_model(x_data, y_data)
    model_2 = train_model(x_data, y_data)

    preds_1 = model_1.predict(x_data)
    preds_2 = model_2.predict(x_data)

    assert np.array_equal(preds_1, preds_2), (
        "Model predictions differ across runs, indicating training is not reproducible"
    )
