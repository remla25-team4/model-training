"""Metamorphic test to check model prediction consistency with punctuation variations."""

import json
from pathlib import Path

import pytest
import pandas as pd
from joblib import load


def load_model_and_vectorizer():
    """Load the trained model and vectorizer from disk, or skip if not found."""
    model_path = Path("models/naive_bayes.joblib")
    vectorizer_path = Path("models/count_vectorizer.joblib")

    if not model_path.exists() or not vectorizer_path.exists():
        pytest.skip("Model or vectorizer not found. Run `dvc pull` first.")

    return load(model_path), load(vectorizer_path)


def load_sample_data():
    """Load and sample the training dataset for testing."""
    data_path = Path("data/raw/training_dataset.tsv")
    if not data_path.exists():
        pytest.skip("Dataset not found. Run `dvc pull` first.")

    data = pd.read_csv(data_path, sep="\t")
    return data.sample(n=min(50, len(data)), random_state=42)


def mutate_texts(text):
    """Return variations of a text with added punctuation marks."""
    mutations = []
    if not text.endswith('.'):
        mutations.append(text + ".")
    if not text.endswith('!'):
        mutations.append(text + "!")
    return mutations


@pytest.mark.model_test
def test_prediction_consistency_with_punctuation():
    """Test that adding punctuation to reviews doesn't change the model's prediction"""

    model, vectorizer = load_model_and_vectorizer()
    sample_data = load_sample_data()

    failures = []

    for _, row in sample_data.iterrows():
        review = row["Review"]
        original_vec = vectorizer.transform([review]).toarray()
        original_pred = model.predict(original_vec)[0]

        for mutated in mutate_texts(review):
            mutated_vec = vectorizer.transform([mutated]).toarray()
            mutated_pred = model.predict(mutated_vec)[0]
            if mutated_pred != original_pred:
                failures.append({
                    "original": review,
                    "mutated": mutated,
                    "original_prediction": int(original_pred),
                    "mutated_prediction": int(mutated_pred)
                })

    output_path = Path("data/processed/mutamorphic_failures.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    assert not failures, f"{len(failures)} inconsistencies found. See {output_path}"
