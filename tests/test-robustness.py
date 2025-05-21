import pytest
from joblib import load
import nltk
from nltk.corpus import wordnet
from nltk.stem.porter import *
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('wordnet')
nltk.download('omw-1.4')

"""
    Get synonyms by using the wordnet library
"""
def get_synonym(word):
    stemmer = PorterStemmer()

    synsets = wordnet.synsets(word)
    if not synsets:
        return None
    for lemma in synsets[0].lemmas():
        name = lemma.name().replace("_", " ")
        if name.lower() != word.lower() and stemmer.stem(name.lower()) != stemmer.stem(word.lower()) and ' ' not in name:
            return name
    return None

"""
    Replace words in text with synonym.
"""
def replace_with_synonyms(text):
    words = text.split()
    new_texts = []

    for i, word in enumerate(words):
        synonym = get_synonym(word)
        if synonym:
            new_text = words.copy()
            new_text[i] = synonym
            new_texts.append(" ".join(new_text))

    return new_texts

# Capture the trained model and the CV
@pytest.fixture(scope="module")
def model_and_vectorizer():
    # Load model and vectorizer (adjust paths accordingly)
    model = load("../models/naive_bayes.joblib")
    vectorizer = load("../models/count_vectorizer.joblib")
    return model, vectorizer

# The text inputs to provide synonyms for
@pytest.mark.parametrize("text", [
    "The food was good, service was fast",
    "The food was great, service was okay",
    "The food was bad, service was slow",
    "Bad vibes, great food",
    "Prices were cheap"
])


def test_synonyms_replacements(model_and_vectorizer, text):
    # Load model and CV
    model, vectorizer = model_and_vectorizer

    # Replace one word in a sentence with its synonyms
    # Do this for all words
    mutations = replace_with_synonyms(text)

    # Predict
    original_vec = vectorizer.transform([text]).toarray()
    original_pred = model.predict(original_vec)[0]

    # Compare the predictions for each mutation
    for mutation in mutations:
        mutation_vec = vectorizer.transform([mutation]).toarray()
        similarity = cosine_similarity(original_vec, mutation_vec)[0][0]

        # Would you keep this in?
        # assert similarity > 0.7, (
        #    f"Low vector similarity with variant:\n"
        #    f"Original: {text}\nMutation: {mutation}\nSim: {similarity:.2f}"
        # )

        mutation_pred = model.predict(mutation_vec)[0]
        assert mutation_pred == original_pred, (
            f"Prediction changed with change from\n"
            f"Original: {text} -> {original_pred}\n"
            "to\n"
            f"Variant: {mutation} -> {mutation_pred}"
        )

