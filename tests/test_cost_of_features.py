"""Evaluate how varying the number of features affects model accuracy (cost of features)."""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from pipeline.training import train_model

def evaluate_feature_limit_accuracy(texts, labels, feature_limits):
    """Determining accuracy vales for each vector"""
    accuracies = []
    for max_feat in feature_limits:
        vectorizer = CountVectorizer(max_features=max_feat)
        x_data = vectorizer.fit_transform(texts).toarray()
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, labels, test_size=0.2, random_state=42
        )
        model = train_model(x_train, y_train)
        y_pred = model.predict(x_test)
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)
    return accuracies


def test_cost_of_features_ratio():
    """Check that accuracy does not decrease significantly as feature count increases."""
    df = pd.read_csv("data/raw/training_dataset.tsv", sep="\t")
    texts = df["Review"]
    labels = df["Liked"]

    full_vectorizer = CountVectorizer()
    full_vectorizer.fit(texts)
    vocab_size = len(full_vectorizer.vocabulary_)

    ratios = [0.5, 0.75, 1.0]
    feature_limits = [int(vocab_size * r) for r in ratios]

    accuracies = evaluate_feature_limit_accuracy(texts, labels, feature_limits)

    print(f"\nFeature limits: {feature_limits}")
    print(f"Accuracies: {accuracies}")

    differences = [
        round(accuracies[i + 1] - accuracies[i], 3)
        for i in range(len(accuracies) - 1)
    ]

    for i, diff in enumerate(differences):
        if diff > 0.05:
            print(
                f"Accuracy improved significantly from {ratios[i]} to {ratios[i+1]} ({diff:+})"
            )
        elif diff < -0.05:
            print(
                f"Accuracy dropped significantly from {ratios[i]} to {ratios[i+1]} ({diff:+})"
            )
        else:
            print(
                f"Insignificant accuracy change between {ratios[i]} and {ratios[i+1]} ({diff:+})"
            )

    assert all(
        diff >= -0.05 for diff in differences
    ), f"Test failed - accuracy drop: {differences}"
