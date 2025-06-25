"""Metamorphic test to check model prediction consistency with punctuation and antonym variations"""

import json
from pathlib import Path
import pytest
import pandas as pd
from joblib import load

def load_model_and_vectorizer():
    """loading the model and the count vectorizer"""
    model_path = Path("models/naive_bayes.joblib")
    vectorizer_path = Path("models/count_vectorizer.joblib")
    if not model_path.exists() or not vectorizer_path.exists():
        pytest.skip("Model or vectorizer not found. Run `dvc pull` first.")
    return load(model_path), load(vectorizer_path)

def load_sample_data():
    """loading the sampling data with a min of 50"""
    data_path = Path("data/raw/training_dataset.tsv")
    if not data_path.exists():
        pytest.skip("Dataset not found. Run `dvc pull` first.")
    data = pd.read_csv(data_path, sep="\t")
    return data.sample(n=min(50, len(data)), random_state=42)

def mutate_punctuation(text):
    """ mutate punctuation fr metamorphic testing"""
    mutations = []
    if not text.endswith('.'):
        mutations.append((text + ".", "same"))
    if not text.endswith('!'):
        mutations.append((text + "!", "same"))
    return mutations

def apply_antonyms(text):
    """changing certain words for antonyms for metamorphic testing"""
    antonyms = {
        "good": "bad", "bad": "good",
        "happy": "unhappy", "unhappy": "happy",
        "like": "dislike", "dislike": "like",
        "love": "hate", "hate": "love",
        "great": "terrible", "terrible": "great",
        "awesome": "awful", "awful": "awesome",
        "positive": "negative", "negative": "positive",
        "didn't": "did", "did": "didn't",
        "better": "worse", "worse": "better",
        "amazing": "horrible", "horrible": "amazing",
        "never": "always", "always": "never",
        "wow": "ew", "ew": "wow",
        "terrific": "horrific", "horrific": "terrific"
    }
    tokens = text.split()
    replaced = [antonyms.get(token.lower(), token) for token in tokens]
    return " ".join(replaced)

def expected_flip(label):
    """reverse the label when antonym applied"""
    return 0 if label == 1 else 1

def update_metrics(metrics, result, test_type, context, failures):
    """update metrics for automatic inconsistency repair."""
    metrics["total_tests"] += 1
    metrics[f"{test_type}_total"] += 1
    expected, predicted = result["expected"], result["predicted"]
    if expected == predicted:
        metrics["correct"] += 1
        metrics[f"{test_type}_correct"] += 1
    else:
        failures.append({
            "original": context["original"],
            "mutated": context["mutated"],
            "expected": "same" if test_type == "punc" else "flipped",
            "original_prediction": expected if test_type == "punc" else expected_flip(expected),
            "mutated_prediction": predicted
        })

@pytest.mark.model_test
def test_consistency_with_punctuation_and_antonyms(trained_model,
                                                    cv, output_dir=Path("data/processed")):
    """
    Performs metamorphic testing for punctuation and antonyms.
    This version is refactored to use fewer local variables.
    """
    sample_data = load_sample_data()

    failures = []
    metrics = {
        "total_tests": 0, "correct": 0,
        "punc_total": 0, "punc_correct": 0,
        "antonym_total": 0, "antonym_correct": 0,
    }

    # Process all reviews
    for _, row in sample_data.iterrows():
        original_review = row["Review"]
        original_pred = int(trained_model.predict(cv.transform([original_review]).toarray())[0])

        # Test punctuation mutations (consistency check)
        for mutated_text, _ in mutate_punctuation(original_review):
            mutated_pred = int(trained_model.predict(cv.transform([mutated_text]).toarray())[0])
            update_metrics(
                metrics,
                result={"expected": original_pred, "predicted": mutated_pred},
                test_type="punc",
                context={"original": original_review, "mutated": mutated_text},
                failures=failures
            )

        # Test antonym mutation (flip check)
        antonym_text = apply_antonyms(original_review)
        if antonym_text != original_review:
            antonym_pred = int(trained_model.predict(cv.transform([antonym_text]).toarray())[0])
            update_metrics(
                metrics,
                result={"expected": expected_flip(original_pred), "predicted": antonym_pred},
                test_type="antonym",
                context={"original": original_review, "mutated": antonym_text},
                failures=failures
            )

    # --- Save results and print metrics ---
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metamorphic_failures.json", "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)

    if failures:
        pd.DataFrame(failures).to_csv(
            output_dir / "retrain_candidates.tsv", sep="\t", index=False
        )

    # REFACTORED: Calculate rates and update metrics dictionary directly
    metrics.update({
        "failures": len(failures),
        "consistency_rate": round(
            (metrics["correct"] / metrics["total_tests"])
            if metrics["total_tests"] else 0, 3
        ),
        "punc_consistency_rate": round(
            (metrics["punc_correct"] / metrics["punc_total"])
            if metrics["punc_total"] else 0, 3
        ),
        "antonym_flip_rate": round(
            (metrics["antonym_correct"] / metrics["antonym_total"])
            if metrics["antonym_total"] else 0, 3
        )
    })

    # Print results directly from the metrics dictionary
    print("Punctuation Consistency Rate:")
    print(f" {metrics['punc_correct']}/{metrics['punc_total']} = "
          f"{metrics['punc_consistency_rate']:.3f}")
    print("Antonym Flip Accuracy Rate:")
    print(f" {metrics['antonym_correct']}/{metrics['antonym_total']} = "
          f"{metrics['antonym_flip_rate']:.3f}")
    print(f"[Metamorphic Test] Overall Correct: {metrics['correct']}/{metrics['total_tests']}")
    print(f"Overall consistency rate = {metrics['consistency_rate']:.2f}")

    # Save the final metrics dictionary to a file
    with open(output_dir / "metamorphic_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Failures saved to {output_dir / 'metamorphic_failures.json'}")
    print(f"Metrics saved to {output_dir / 'metamorphic_metrics.json'}")

    # Assert using the value from the metrics dictionary
    assert metrics['consistency_rate'] >= 0.8, \
        f"Overall consistency rate is too low: {metrics['consistency_rate']:.2f}"
