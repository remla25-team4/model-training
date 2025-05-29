"""This module evaluates the trained model and outputs metrics."""
import json
import os

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

PROCESSED_DATA_DIR = 'data/processed'
MODELS_DIR = 'models'
METRICS_OUTPUT_DIR = 'metrics'

def load_processed_evaluation_data(base_dir=PROCESSED_DATA_DIR):
    """Loads the processed evaluation data"""
    print(f"Loading processed evaluation data from {base_dir}...")
    x_eval_path = os.path.join(base_dir, 'X_test.joblib')
    y_eval_path = os.path.join(base_dir, 'y_test.joblib')

    x_eval = joblib.load(x_eval_path)
    y_eval = joblib.load(y_eval_path)

    print("Data loaded successfully.")
    return x_eval, y_eval

def evaluate(x_eval, y_eval):
    """Predicts on test set using model and calculate metrics"""
    # Load the trained model
    model = joblib.load('models/naive_bayes.joblib')

    y_pred = model.predict(x_eval)

    # Calculate metrics
    cm = confusion_matrix(y_eval, y_pred)
    print('Confusion matrix:\n', cm)

    accuracy = accuracy_score(y_eval, y_pred)

    # Setting zero_division=0 to prevent warnings if a class has no predictions/true samples
    report = classification_report(y_eval, y_pred, output_dict=True, zero_division=0)
    return report, accuracy

def output_metrics(computed_metrics):
    """Saves calculated metrics to json file"""
    print("Evaluation Metrics:")
    for metric_name, metric_value in computed_metrics.items():
        print(f'{metric_name.replace("_", " ").capitalize()}: {metric_value:.4f}')
    # Save metrics to a file
    os.makedirs(METRICS_OUTPUT_DIR, exist_ok=True)
    metrics_file_path = os.path.join(METRICS_OUTPUT_DIR, 'evaluation_metrics.json')
    with open(metrics_file_path, "w", encoding='utf-8') as f:
        json.dump(computed_metrics, f, indent=4)
    print(f"Evaluation metrics saved to {metrics_file_path}")


if __name__ == "__main__":
    X_test, y_test = load_processed_evaluation_data()
    cls_report, acc = evaluate(X_test, y_test)

    metrics_data = {
		"accuracy": acc,
		"precision_weighted": cls_report["weighted avg"]["precision"],
		"recall_weighted": cls_report["weighted avg"]["recall"],
		"f1_score_weighted": cls_report["weighted avg"]["f1-score"]
	}

    output_metrics(metrics_data)
