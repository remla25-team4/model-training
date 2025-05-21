import sklearn.feature_extraction
import numpy as np
import pandas as pd
import nltk 
import re 
import sklearn
import joblib

from sklearn.model_selection import train_test_split
from lib_ml.preprocessing import preprocess

def preprocess_dataset(dataset):
    # Preprocess dataset to a list and then use the lib-ml preprocessing library
    dataset_as_list = dataset['Review'].to_list()
    reviews = preprocess(dataset_as_list)
    
    return reviews


def to_bag_of_words(corpus, dataset):
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=140)
    cv_model = cv.fit(corpus)

    joblib.dump(cv_model, 'models/count_vectorizer.joblib') 

    X = cv_model.transform(corpus).toarray()
    y = dataset.iloc[: , -1].values

    return X, y

def get_dataset():
    # Read the dataset.
    # Dataset obtained from https://github.com/aadimangla/Restaurant-Reviews-Sentiment-Analysis/blob/master/Dataset/Restaurant_Reviews.tsv

    dataset = pd.read_csv("datasets/training_dataset.tsv", delimiter = '\t')
    corpus = preprocess_dataset(dataset)

    X, y = to_bag_of_words(corpus, dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    return X_train, X_test, y_train, y_test 


if __name__ == "__main__":
    get_dataset()
