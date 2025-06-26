# model-training

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project closely follows the [Cookiecutter template](https://github.com/drivendataorg/cookiecutter-data-science), a template that incorporates best practices for data science.


![Pylint Score](https://img.shields.io/badge/pylint-10.00%2F10-brightgreen)
![ml_score](https://img.shields.io/badge/ML_Score-2-red)
![coverage](https://img.shields.io/badge/Coverage-22%25-red)


## Overview

This repo contains the training pipeline for the Sentiment Analysis model

The training process is now organized using the Cookiecutter DS project structure. It separates the pipeline into 3 stages:
1. **Data preparation** (`data_processing.py`)
2. **Model training** (`training.py`)
3. **Model evaluation** (`evaluation.py`)

The trained model and vectorizer are saved as `.joblib` files and used by the `model-service` for inference.

---

## Project Structure

```
model-training/
├── data
├── docs
├── .dvc
├── .dvcignore
├── dvc.lock
├── dvc.yaml
├── .github
│   └── workflows
│       ├── coverage_test_score.yml
│       ├── prerelease.yml
│       ├── pylint.yml
│       └── release.yml
├── LICENSE
├── Makefile
├── metrics
│   └── evaluation_metrics.json
├── models
├── notebooks
├── pipeline
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── __init__.py
│   └── training.py
├── pylint_ml_smells
│   ├── hyperparameter_checker.py
│   ├── __init__.py
│   └── zero_division_checker.py
├── .pylintrc
├── pyproject.toml
├── pytest.ini
├── README.md
├── references
├── reports
├── requirements.txt
├── setup.cfg
├── setup_nltk.py
├── tests
│   ├── conftest.py
│   ├── generate_scores.py
│   ├── test_cost_of_features.py
│   ├── test_infrastructure.py
│   ├── test_memory.py
│   ├── test_metamorphic.py
│   ├── .test_metamorphic.py.swp
│   ├── test_model_development.py
│   ├── test_monitoring.py
│   ├── test_performance.py
│   ├── test_preprocessing.py
│   └── test_robustness.py
└──

```

#  Assignment 4: Model Training Setup Guide
This section walks you through everything you need to set up and run the pipeline for Assignment 4.

---

##  Step 1: Access to the Google Drive Remote (JSON Credential)
Our DVC setup uses a Google Cloud service account to access the dataset and models stored on Google Drive.
You need access to the service account credentials JSON file before running any pipeline code.
The Google Drive remote is fully configured via your DVC config file—no manual OAuth or env-vars needed.

There are three ways to do this:

### 1. Request the JSON File
- Email `y.chen-112@student.tudelft.nl` requesting the credentials file.
- Once received, save it as: `gdrive_sa_credentials.json` in the directory .dvc of this project.

### 2.  Use the Pre-Included Credential (Grading Only)
**For instructors and graders only**  
To simplify review and grading for the Release Engineering for Machine Learning Applications course, a working credential file is already included in the project submission `.zip`.

You can use it without requesting access:

1. Locate the file in the submission folder's `.dvc` directory:
   ```
   gdrive_sa_credentials.json
   ```
   You do not have to do anything, please leave it untouched. And most importantly, do not distribute this file publicly!
---

## Step 2: Set Up Virtual Environment
A virtual environment keeps dependencies isolated from your global Python setup so:

### Create and activate the virtual environment (Python 3.10):
```bash
python3.10 -m venv venv
source venv/bin/activate
```

### Install dependencies:
```bash
pip install -r requirements.txt
```

### Install NLTK corpora:
```bash
python setup_nltk.py
```

---


## Step 3: Run the Pipeline
There are two ways to run the training and evaluation pipeline.

### 1. Run Stages Manually
```bash
python3 pipeline/data_processing.py
python3 pipeline/training.py
python3 pipeline/evaluation.py
```

### 2. Use DVC
Pull all required files from remote:
```bash
dvc pull
```

Run all pipeline stages in order:
```bash
dvc repro
```
View the tracked metrics (accuracy, f1_score_weighted, precision_weighted, recall_weighted)
```bash
dvc metrics show
```

---
##  Rollback to Past Versions
To roll back to a specific previous version:

```bash
# Go to a previous Git commit
git log --oneline 
git checkout <commit_hash>

# Restore the respective data and model artifacts
dvc checkout

```

---
## Testing
Run all tests:
```bash
pytest -s tests/
```
This will run all test categories and display whether each one passes its subtests. The final score summarizes test adequacy. Output metrics are saved to `metrics/evaluation_metrics.json`
This project implements structured machine learning testing aligned with the ML Test Score framework from [Google’s ML Test Score rubric (2022)](https://storage.googleapis.com/gweb-research2023-media/pubtools/4156.pdf). The ML Test Score (test adequacy score) follows the suggested formula from said rubric.

---

## Test Coverage Report

To manually measure code coverage:

1. Install coverage:

```bash
pip install coverage
```

2. Run tests with coverage:

```bash
coverage run --source=pipeline -m pytest -v tests
coverage report
```

3. Optionally, generate an HTML report for easier reading:

```bash
coverage html
```

Then open `htmlcov/index.html` in your browser to explore detailed coverage.

---

## Test Category Overview

| Test Category         | Test File                    |
|-----------------------|------------------------------|
| Feature & Data        | `test_preprocessing.py`      |
| Model Development     | `test_model_development.py`  |
| ML Infrastructure     | `test_infrastructure.py`     |
| Monitoring            | `test_monitoring.py`         |
| Robustness            | `test_robustness.py`         |
| Memory Usage          | `test_memory.py`             |
| Performance           | `test_performance.py`        |
| Cost of Features      | `test_cost_of_features.py`   |
| Metamorphic Testing   | `test_metamorphic.py`        |

Each category is designed to cover one dimension of ML production readiness. The tests include:

- Functional validation
- Non-determinism checks using data slices
- Non-functional metrics like memory, speed, and cost
- Metamorphic consistency and auto-repair logic

---
## Comparing Experiments with DVC

To view and compare metrics across different experiments, run:
```bash
dvc exp show
```
This will output a table of all experiments and their metrics (accuracy, precision_weighted, recall_weighted, F1).

To create a new experiment:
```bash
dvc exp run -n <experiment-name>
```

To restore a particular experiment to your workspace:
```bash
dvc exp apply <experiment-name>
```


Note: make sure your DVC setup works by running `dvc pull` and `dvc repro` without issues.


## Code Quality

This project uses four linters and formatters to maintain clean and consistent code:

- Pylint: static analysis and scoring  
- Flake8: style guide enforcement  
- Black: automatic code formatting  
- isort: import sorting

### Run all checks

```bash
# Run Pylint on source and test code
pylint pipeline/ tests/

# Run Flake8 to check for style violations
flake8 pipeline/ tests/

# Check formatting (Black) without changing files
black --check pipeline/ tests/

# Check formatting (Black) and automatically fix issues
black pipeline/ tests/

# Check import sorting without changing files
isort --check-only pipeline/ tests/

# Check import sorting and apply changes automatically
isort pipeline/ tests/

#Check empty instantiations of GaussianNB without any hyperparameters and to check for non_zero_division argument in classification_report
pylint --load-plugins=pylint_ml_smells pipeline/


```
