"""Preprocesses raw data for training"""

import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from lib_ml.preprocessing import preprocess

def preprocess_dataset(dataset):
    """preprocess raw data for training using lib-ml preprocess function"""
    dataset_as_list = dataset['Review'].to_list()
    reviews = preprocess(dataset_as_list)
    return reviews

def to_bag_of_words(corpus, dataset):
    """save count vectorizer model and convert corpus to bag of words"""
    cv = CountVectorizer(max_features=140)
    cv_model = cv.fit(corpus)

    joblib.dump(cv_model, 'models/count_vectorizer.joblib')

    x = cv_model.transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    return x, y

def get_dataset():
    """preprocess the raw dataset and create a train test split"""
    dataset = pd.read_csv("data/raw/training_dataset.tsv", delimiter='\t')
    corpus = preprocess_dataset(dataset)
    x, y = to_bag_of_words(corpus, dataset)
    return train_test_split(x, y, test_size=0.20, random_state=42)

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_dataset()

    print(X_train.shape, y_train.shape)

    PROCESSED_DATA_DIR = 'data/processed'
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

    # --- Save the processed data ---
    joblib.dump(X_train, os.path.join(PROCESSED_DATA_DIR, 'X_train.joblib'))
    joblib.dump(X_test, os.path.join(PROCESSED_DATA_DIR, 'X_test.joblib'))

    joblib.dump(y_train, os.path.join(PROCESSED_DATA_DIR, 'y_train.joblib'))
    joblib.dump(y_test, os.path.join(PROCESSED_DATA_DIR, 'y_test.joblib'))
