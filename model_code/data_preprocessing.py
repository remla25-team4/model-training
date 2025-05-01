import sklearn.feature_extraction
import numpy as np
import pandas as pd
import nltk 
import re 
import sklearn

from sklearn.model_selection import train_test_split


def preprocess_dataset(dataset):
    # Get English stopwords.

    nltk.download('stopwords')
    words_to_remove = nltk.corpus.stopwords.words('english')
    words_to_remove.remove('not')

    reviews = []

    for review in dataset['Review']:
        # Preprocess review string into a list of words
        preprocessed_review = re.sub('[^a-zA-Z]', ' ', review)
        preprocessed_review = preprocessed_review.lower()
        preprocessed_review = preprocessed_review.split()

        # Remove the stopwords
        preprocessed_review = [word for word in preprocessed_review if word not in words_to_remove]

        reviews.append(' '.join(preprocessed_review))

    return reviews

def to_bag_of_words(corpus, dataset):
    cv = sklearn.feature_extraction.text.CountVectorizer(max_features=140)
    X = cv.fit_transform(corpus).toarray()
    y = dataset.iloc[: , -1].values
    return X, y

def get_dataset():
    # Read the dataset.
    # Dataset obtained from https://github.com/aadimangla/Restaurant-Reviews-Sentiment-Analysis/blob/master/Dataset/Restaurant_Reviews.tsv

    dataset = pd.read_csv("../datasets/training_dataset.tsv", delimiter = '\t')
    corpus = preprocess_dataset(dataset)

    X, y = to_bag_of_words(corpus, dataset)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    
    return X_train, X_test, y_train, y_test 


if __name__ == "__main__":
    get_dataset()
