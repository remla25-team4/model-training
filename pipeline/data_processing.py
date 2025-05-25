#the data processing file

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from lib_ml.preprocessing import preprocess

def preprocess_dataset(dataset):
    dataset_as_list = dataset['Review'].to_list()
    reviews = preprocess(dataset_as_list)
    return reviews

def to_bag_of_words(corpus, dataset):
    cv = CountVectorizer(max_features=140)
    cv_model = cv.fit(corpus)

    joblib.dump(cv_model, 'models/count_vectorizer.joblib')

    X = cv_model.transform(corpus).toarray()
    y = dataset.iloc[:, -1].values
    return X, y

def get_dataset():
    dataset = pd.read_csv("datasets/training_dataset.tsv", delimiter='\t')
    corpus = preprocess_dataset(dataset)
    X, y = to_bag_of_words(corpus, dataset)
    return train_test_split(X, y, test_size=0.20, random_state=42)

