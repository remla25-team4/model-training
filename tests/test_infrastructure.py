"""Tests for ML infrastructure via reproducibility"""

import numpy as np
from pipeline.training import train_model, load_processed_training_data

def test_training_reproducibility():
    """
    Tests that training the model twice on the same data with fixed random seed. Folling Infra1 from ML Test Score Guide
    """
    
    X, y = load_processed_training_data()

    # train twice
    model_1 = train_model(X, y)
    model_2 = train_model(X, y)

    # compare predictions (should be enough for GaussianNB)
    preds_1 = model_1.predict(X)
    preds_2 = model_2.predict(X)

    assert np.array_equal(preds_1, preds_2), (
        "Model predictions differ across runs, indicating training is not reproducible"
    )
