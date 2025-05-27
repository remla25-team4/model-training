# model-training

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

This project closely follows the [Cookiecutter template](https://github.com/drivendataorg/cookiecutter-data-science), a template that incorporates best practices for data science.

![Pylint Score](https://img.shields.io/badge/pylint-7.42%2F10-yellowgreen)

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

---

## Installation
It is recommended to use a virtual environment with python 3.10
```bash
python3.10 -m venv venv
source venv/bin/activate
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

**Important!**

Then, run the following script to make sure needed nltk corpora is installed on your machine:

```bash
python setup_nltk.py
```


---

## Usage

### Preprocess the data


### Train the model:
```bash

python3 pipeline/training.py
```



### Evaluate the model:
```bash
python3 pipeline/evaluation.py
```

---

## Testing

Run tests with:
```bash
pytest tests/
```


---

## DVC
You can run the training and evaluation scripts manually, or use DVC to reproduce the full pipeline including versioned data, models, and metrics. 

### Remote Storage

DVC is configured to push data to a Google Drive remote using OAuth through a custom Google Cloud Project
.  
If you're contributing to this project, you’ll need to authenticate via the browser when running `dvc pull` or `dvc push` for the first time.
To reproduce the pipeline and download all required files:

### To clone and setup:

```bash
git clone git@github.com:remla25-team4/model-training.git
```

Pull all required files (dataset, models, metrics):
```bash
dvc pull
```

### To reproduce the full pipeline:

```bash
dvc repro
```

 We are currently using a service account on Google Cloud to access our data with dvc. In order to access our service account you will need the json key file.
---
