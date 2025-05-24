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


---

## Usage

### Train the model:
```bash
python model_training/training.py
```

This will:
- Load and preprocess the dataset in `data/raw/`
- Train a `GaussianNB` classifier
- Save the model and vectorizer in `models/`

### Evaluate the model:
```bash
python model_training/evaluation.py
```

---

## Testing

Run tests with:
```bash
pytest tests/
```


---
