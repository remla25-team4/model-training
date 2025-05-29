"""Training code for a GaussianNB classifier"""
import os
import joblib
from sklearn.naive_bayes import GaussianNB

PROCESSED_DATA_DIR = 'data/processed'
MODELS_DIR = 'models'

def load_processed_training_data(base_dir=PROCESSED_DATA_DIR):
    """Loads the processed training data"""
    print(f"Loading processed training data from {base_dir}...")
    x_train_path = os.path.join(base_dir, 'X_train.joblib')
    y_train_path = os.path.join(base_dir, 'y_train.joblib')

    x_train = joblib.load(x_train_path)
    y_train = joblib.load(y_train_path)

    print("Data loaded successfully.")
    return x_train, y_train


def train_model(x_train, y_train):
    """Trains the model"""
    print("Training GaussianNB model...")
    # Using var_smoothing=1e-3 as a more common default if issues arise with 2e-9,
    # but keeping yours for now
    classifier = GaussianNB(var_smoothing=2e-9)
    classifier.fit(x_train, y_train)

    return classifier

def save_model(model, model_name='naive_bayes.joblib', base_dir=MODELS_DIR):
    """Save the trained model"""
    os.makedirs(base_dir, exist_ok=True)
    model_path = os.path.join(base_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    x_training, y_training = load_processed_training_data()

    trained_model = train_model(x_training, y_training)

    save_model(trained_model)

    print("Training pipeline finished.")
