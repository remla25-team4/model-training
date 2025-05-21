# model-training

## Overview

`model-training` is the training code for the Sentiment Analysis model, created as part of the REMLA `Restaurant Sentiment Analysis` application.

Its purpose is to train a model that is saved in a `.joblib` format, which can then be used by the `model-service` to predict the sentiments of specific inputs.

## Installation

To run the application, kindly first install dependencies via `pip install -r requirements.txt`.

## Usage
Once the project is ready to run, simply run [`create_model.py`](https://github.com/remla25-team4/model-training/blob/main/model_code/create_model.py) file, which may be found under the [`model_code`](https://github.com/remla25-team4/model-training/blob/main/model_code/) directory. The created model (and the bag of words encoding) will be saved as separate `.joblib` files in the [`models`](https://github.com/remla25-team4/model-training/tree/main/models) folder.

## Testing
Run the following:
```bash
pytest tests
```