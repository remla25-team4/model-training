# model-training

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
── data
│   ├── external
│   ├── interim
│   ├── processed
│   └── raw
|    	└── training_dataset.tsv
|
├── docs
├── LICENSE
├── Makefile
├── models
│   ├── count_vectorizer.joblib
│   └── naive_bayes.joblib
├── model_training
│   ├── data_processing.py
│   ├── evaluation.py
│   ├── __init__.py
│   └── training.py
├── notebooks
├── pyproject.toml
├── README.md
├── references
├── reports
│   └── figures
├── requirements.txt
├── setup.cfg
├── metrics
│   └── evaluation_metrics.json
└── tests
    ├── test_preprocessing.py
    └── test_robustness.py

```

---

## Installation

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

python3 model_training/training.py
```



### Evaluate the model:
```bash
python3 model_training/evaluation.py
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