"""config file for sharing common fixtures between test files"""

import pytest
import joblib
import pandas as pd

@pytest.fixture(scope="module")
def cv():
    """Loads the CountVectorizer model."""
    return joblib.load("models/count_vectorizer.joblib")

@pytest.fixture(scope="module")
def trained_model():
    """Loads the Naive Bayes model."""
    return joblib.load("models/naive_bayes.joblib")

@pytest.fixture(scope="module")
def off_data():
    """Loads the offline training dataset."""
    # Note: Ensure the path "datasets/training_dataset.tsv" is correct
    # relative to where pytest is run or use absolute paths/path manipulation.
    return pd.read_csv("data/raw/training_dataset.tsv", delimiter='\t')

@pytest.fixture(scope="module")
def on_data():
    """Loads the online dataset for monitoring."""
    # Note: Ensure the path "datasets/online_dataset.tsv" is correct.
    return pd.read_csv("data/raw/online_dataset.tsv", delimiter='\t')

@pytest.fixture
def stopwords_data():
    """Provides test cases for stopword removal functionality."""
    return pd.DataFrame({
        "input_text": [
            "this is an exampl sentenc", # Test stopwords removal
            "this is not an exampl sentenc", # Test that not is kept intact
            "THIS is NOT an exampl sentenc", # Test that capitalized stopwords are removed
            "a not and or not for", # Test that not is the only stopword remaining

            # Test that only 'not' remains.
            # List has been generated with the following GPT-o4 query:
            # Give me a string of all stopwords.
            "a about above after again against all am an "
            "and any are aren’t as at be because been before"
            " being below between both but by can’t couldn’t "
            "doing don’t down during each few for from further "
            "had hadn’t has hasn’t have haven’t having he he’d "
            "he’ll he’s her here here’s hers herself him himself his "
            "how how’s i i’d i’ll i’m i’ve if in into "
            "is isn’t it it’s its itself me more most "
            "mustn’t my myself no nor not of off on once"
            " only or other our ours ourselves out over "
            "own same shan’t she she’d she’ll she’s should"
            " shouldn’t so some such than that that’s the "
            "their theirs them themselves then there "
            "there’s these they they’d they’ll they’re they’ve "
            "this those through to too under until up very "
            "was wasn’t we we’d we’ll we’re we’ve were weren’t what "
            "what’s when when’s where where’s which "
            "while who who’s whom why why’s with won’t wouldn’t you "
            "you’d you’ll you’re you’ve your yours yourself yourselves"

        ],
        "expected_output": [
            "exampl sentenc", # Test removed stopwords and stemming
            "not exampl sentenc", # Test removed capitalization and stemming
            "not exampl sentenc", # Test caps & stemming
            "not not",
            "not"
        ]
    })

@pytest.fixture
def stemming_data():
    """Provides test cases for stemming functionality."""
    return pd.DataFrame({
        "input_text": [
            "example sentence", # Test that stopwords are removed and that stemming works
            "runners",
            "running",
            "EXAMPLE SENTENCE",
            "RUnNING"
        ],
        "expected_output": [
            "exampl sentenc",
            "runner",
            "run",
            "exampl sentenc",
            "run"
        ]
    })

@pytest.fixture
def sample_input():
    X = joblib.load("data/processed/X_train.joblib")
    return X[:100] 
