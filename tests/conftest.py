"""config file for sharing common fixtures between test files"""

import os
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
    if not os.path.exists("data/raw/online_dataset.tsv"):
        pytest.skip("Online data not available – skipping all tests depending on it.")

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
    """Returns a sample of the training data (first 100 rows)."""
    x_sample = joblib.load("data/processed/X_train.joblib")
    return x_sample[:100]



def pytest_terminal_summary(terminalreporter):
    """ML Test Scores per Category """
    expected_tests = {
        "model_test": 
        {"model_1", "model_2", "model_3", "model_4", "model_5", "model_6", "model_7"},
        "infrastructure_test": 
        {"infra_1", "infra_2", "infra_3", "infra_4", "infra_5", "infra_6", "infra_7"},
        "monitoring_test": 
        {"monitor_1", "monitor_2", "monitor_3", "monitor_4", "monitor_5", "monitor_6", "monitor_7"},
        "data_test": 
        {"data_1", "data_2", "data_3", "data_4", "data_5", "data_6", "data_7"}
    }

    executed_tests = {
        "model_test": set(),
        "infrastructure_test": set(),
        "monitoring_test": set(),
        "data_test": set()
    }

    all_outcomes = terminalreporter.stats.get("passed", []) + \
               terminalreporter.stats.get("skipped", []) + \
               terminalreporter.stats.get("failed", [])

    for outcome in all_outcomes:
        for category, subtests in expected_tests.items():
            if category in outcome.keywords:
                for subtest in subtests:
                    if subtest in outcome.keywords:
                        executed_tests[category].add(subtest)

    print("\n\n=== ML TEST SCORE SUMMARY ===")

    adequacy_ratios = []
    for category, expected in expected_tests.items():
        executed = executed_tests[category]
        total = len(expected)
        passed = len(executed)
        adequacy = passed / total if total else 1.0
        adequacy_ratios.append(adequacy)

        print(f"\n{category.replace('_', ' ').title()}:")
        print(f"  {passed} of {total} required subtests implemented ({adequacy:.2f})")
        if executed:
            print(f"  Covered subtests: {sorted(executed)}")
        else:
            print("  No subtests passed.")


    ml_test_score = min(adequacy_ratios)
    print(f"\nML Test Score: {ml_test_score:.2f} / 1.0")
    print("Interpreted as:", interpret_score(ml_test_score))

def interpret_score(score):
    """ Defining print statement and qualitive ML Test Score"""
    if score == 0:
        return "More of a research project than a productionized system"
    if score <= 1:
        return "Not totally untested, but possibility of serious holes in reliability"
    if score <= 2:
        return "There’s first pass at basic productionization, additional investment needed."
    if score <= 3:
        return "Reasonably tested, but more of those tests and procedures may be automated."
    if score <= 5:
        return "Strong testing and monitoring, appropriate for mission-critical systems."
    return "Exceptional levels of automated testing and monitoring."
