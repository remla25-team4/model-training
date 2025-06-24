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
def test_prediction_consistency_with_punctuation_and_antonyms():
    """applying metamorphic testing"""
    model, vectorizer = load_model_and_vectorizer()
    sample_data = load_sample_data()

    failures = []
    metrics = {
        "total_tests": 0,
        "correct": 0,
        "punc_total": 0,
        "punc_correct": 0,
        "antonym_total": 0,
        "antonym_correct": 0,
    }

    for _, row in sample_data.iterrows():
        review = row["Review"]
        original_vec = vectorizer.transform([review]).toarray()
        original_pred = int(model.predict(original_vec)[0])

        for mutated_text, _ in mutate_punctuation(review):
            mutated_vec = vectorizer.transform([mutated_text]).toarray()
            mutated_pred = int(model.predict(mutated_vec)[0])
            context = {"original": review, "mutated": mutated_text}
            result = {"expected": original_pred, "predicted": mutated_pred}
            update_metrics(metrics,result,"punc",context,failures)

        antonym_text = apply_antonyms(review)
        if antonym_text != review:
            antonym_vec = vectorizer.transform([antonym_text]).toarray()
            antonym_pred = int(model.predict(antonym_vec)[0])
            context = {"original": review, "mutated": antonym_pred}
            result = {"expected": expected_flip(original_pred), "predicted": antonym_pred}
            update_metrics(metrics,result,"antonym", context, failures)


    output_path = Path("data/processed/mutamorphic_failures.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(failures, f, indent=2)


    if failures:
        repaired_data = pd.DataFrame(failures)
        repaired_path = Path("data/processed/retrain_candidates.tsv")
        repaired_data.to_csv(repaired_path, sep="\t", index=False)


    consistency_rate = round(metrics["correct"] / metrics["total_tests"], 3)
    punc_rate=round(metrics["punc_correct"]/metrics["punc_total"],3) if metrics["punc_total"] else 0
    t = metrics["antonym_total"]
    antonym_rate = round(metrics["antonym_correct"] / t, 3) if t else 0

    print("Punctuation Consistency Rate:")
    print(f" {metrics['punc_correct']}/{metrics['punc_total']} = {punc_rate}")
    print("Antonym Flip Accuracy Rate: ")
    print(f" {metrics['antonym_correct']}/{metrics['antonym_total']} = {antonym_rate}")
    print(f"[Mutamorphic Test] {metrics['correct']}/{metrics['total_tests']} correct")
    print(f"consistency rate = {consistency_rate:.2f}")


    metrics["failures"] = len(failures)
    metrics["consistency_rate"] = consistency_rate
    metrics_path = Path("data/processed/mutamorphic_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Failures saved to {output_path}")
    print(f"Metrics saved to {metrics_path}")
    assert consistency_rate >= 0.8, f"Consistency too low: {consistency_rate:.2f}"
