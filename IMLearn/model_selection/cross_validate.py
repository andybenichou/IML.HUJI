from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float], cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    total_train_score, total_validation_score = 0, 0
    X_split, y_split = np.array_split(X, cv), np.array_split(y, cv)
    for i in range(cv):
        X_fold, y_fold = X_split[i], y_split[i]

        X_others_fold = np.concatenate(X_split[:i] + X_split[i+1:])
        y_others_fold = np.concatenate(y_split[:i] + y_split[i+1:])

        estimator.fit(X_others_fold, y_others_fold)
        total_train_score += scoring(estimator.predict(X_others_fold),
                                     y_others_fold)
        total_validation_score += scoring(estimator.predict(X_fold),
                                          y_fold)

    train_score = total_train_score/cv
    validation_score = total_validation_score/cv

    return train_score, validation_score
