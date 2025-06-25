"""
Feature and data testing. We check that the preprocessed
input strings contain no caps and stop words other than 'not'."""

from lib_ml.preprocessing import preprocess
import nltk
import pytest

@pytest.mark.data_test
@pytest.mark.data_1
def test_stopwords_removal(stopwords_data):
    """Tests stopword removal from text, ensuring 'not' is preserved."""
    nltk.download('wordnet')
    # Remove stopwords for each input and check against expected output.
    for _, row in stopwords_data.iterrows():
        input_text = row["input_text"]
        expected = row["expected_output"]
        result = preprocess([input_text])
        assert result[0] == expected, \
            f"Failed on input: {input_text!r}. Expected {expected!r} but got {result!r}."

@pytest.mark.data_test
@pytest.mark.data_1
def test_stemming(stemming_data):
    """Tests stemming of words to their root form."""
    nltk.download('wordnet')
    # Do stemming on each input and check against expected output.
    for _, row in stemming_data.iterrows():
        input_text = row["input_text"]
        expected = row["expected_output"]
        result = preprocess([input_text])
        assert result[0] == expected, \
            f"Failed on input: {input_text!r}. Expected {expected!r} but got {result!r}."
