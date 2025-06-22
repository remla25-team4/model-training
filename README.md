# model-training

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project closely follows the [Cookiecutter template](https://github.com/drivendataorg/cookiecutter-data-science), a template that incorporates best practices for data science.

![Pylint Score](https://img.shields.io/badge/pylint-9.79%2F10-brightgreen)

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
├── .dvc/
├── .github/
├── data/
│   ├── processed/
│   │   ├── .gitignore
│   │   ├── X_test.joblib
│   │   ├── X_train.joblib
│   │   ├── y_test.joblib
│   │   └── y_train.joblib
│   ├── raw/
│   │   ├── .gitignore
│   │   ├── training_dataset.tsv
│   │   └── training_dataset.tsv.dvc
│   └── .gitkeep
├── docs/
├── LICENSE
├── Makefile
├── metrics/
│   └── evaluation_metrics.json
├── models/
│   ├── count_vectorizer.joblib
│   └── naive_bayes.joblib
├── notebooks/
├── pipeline/
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── __init__.py
│   └── training.py
├── pyproject.toml
├── README.md
├── pylint_ml_smells/
│   ├── __init__.py
│   └── hyperparameter_checker.py
├── references/
├── reports/
│   └── figures/
├── requirements.txt
├── setup.cfg
└── tests/
    ├── test_model_development.py
    ├── test_monitoring.py
    ├── test_preprocessing.py
    └── test_robustness.py
    

```

#  Assignment 4: Model Training Setup Guide
This section walks you through everything you need to set up and run the pipeline for Assignment 4.

---

##  Step 1: Access to the Google Drive Remote (JSON Credential)
Our DVC setup uses a Google Cloud service account to access the dataset and models stored on Google Drive.
You need access to the service account credentials JSON file before running any pipeline code.

There are two ways to do this:

### 1. Request the JSON File
- Email `cguerreirodoes@tudelft.nl` requesting the credentials file.
- Once received, save it on your own device as `causal-root-460921-d8-0857aa4be999.json`.
- Then export the following environment variable in your terminal:
  ```bash
  export GOOGLE_APPLICATION_CREDENTIALS=causal-root-460921-d8-0857aa4be999.json
  ```

### 2. Request Access to the Service Account
- Email `cguerreirodoes@tudelft.nl` and ask to be granted access using your Google email address.
- Once access is granted:
  1. Go to [https://console.cloud.google.com/](https://console.cloud.google.com/)
  2. Navigate to "IAM & Admin" and then to "Service Accounts"
  3. Locate the service account `restaurant-sentiment@causal-root-460921-d8.iam.gserviceaccount.com`
  4. Create and download a new JSON key file
  5. Save it as `causal-root-460921-d8-0857aa4be999.json` 
  6. Export the environment variable:
     ```bash
     export GOOGLE_APPLICATION_CREDENTIALS=causal-root-460921-d8-0857aa4be999.json
     ```

---

## Step 2: Set Up Virtual Environment
A virtual environment keeps dependencies isolated from your global Python setup so:

### Create and activate the virtual environment:
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
Run unit tests:
```bash
pytest tests/
```

Output metrics are saved to `metrics/evaluation_metrics.json`
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
flake8

# Check formatting (Black) without changing files
black --check pipeline/ tests/

# Check import sorting without changing files
isort --check-only pipeline/ tests/

#Check empty instantiations of GaussianNB without any hyperparameters
pylint --load-plugins=pylint_ml_smells test.py


```
