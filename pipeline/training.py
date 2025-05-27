# pipeline/training.py

import joblib
import pandas as pd
import json  # For saving metrics
import os

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score

# No longer importing get_dataset from pipeline.data_processing

PROCESSED_DATA_DIR = 'data/processed'
MODELS_DIR = 'models'
METRICS_DIR = 'metrics'  # Directory to save metrics


def load_processed_data(base_dir=PROCESSED_DATA_DIR):
    print(f"Loading processed data from {base_dir}...")
    X_train_path = os.path.join(base_dir, 'X_train.joblib')
    X_test_path = os.path.join(base_dir, 'X_test.joblib')
    y_train_path = os.path.join(base_dir, 'y_train.joblib')
    y_test_path = os.path.join(base_dir, 'y_test.joblib')

    X_train = joblib.load(X_train_path)
    X_test = joblib.load(X_test_path)

    y_train = joblib.load(y_train_path)
    y_test = joblib.load(y_test_path)

    print("Data loaded successfully.")
    return X_train, X_test, y_train, y_test


def train_model(X_train, X_test, y_train, y_test):
    print("Training GaussianNB model...")
    # Using var_smoothing=1e-3 as a more common default if issues arise with 2e-9, but keeping yours for now
    classifier = GaussianNB(var_smoothing=2e-9)
    classifier.fit(X_train, y_train)

    print("Evaluating model...")
    y_pred = classifier.predict(X_test)

    cm = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    print("Confusion Matrix:")
    print(cm)
    print(f"Accuracy: {acc:.4f}")

    return classifier

def save_model(model, model_name='naive_bayes.joblib', base_dir=MODELS_DIR):
    os.makedirs(base_dir, exist_ok=True)
    model_path = os.path.join(base_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()

    trained_model = train_model(X_train, X_test, y_train, y_test)

    save_model(trained_model)

    print("Training pipeline finished.")