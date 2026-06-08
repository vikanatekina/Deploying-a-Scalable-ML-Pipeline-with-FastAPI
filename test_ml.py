import numpy as np

from ml.data import apply_label
from ml.model import compute_model_metrics, inference, train_model
from sklearn.ensemble import RandomForestClassifier


def test_compute_model_metrics():
    y = np.array([1, 0, 1, 0])
    preds = np.array([1, 0, 1, 0])

    precision, recall, fbeta = compute_model_metrics(y, preds)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_train_model_returns_random_forest():
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)


def test_inference_returns_predictions():
    X_train = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
    y_train = np.array([0, 0, 1, 1])

    model = train_model(X_train, y_train)
    preds = inference(model, X_train)

    assert len(preds) == len(y_train)
    assert set(preds).issubset({0, 1})


def test_apply_label():
    assert apply_label([1]) == ">50K"
    assert apply_label([0]) == "<=50K"
