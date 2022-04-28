from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv

from ...metrics import loss_functions


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
    """
    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None, None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector, same covariance
        matrix with dependent features.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)

        self.mu_ = np.array([np.mean(X[y == k], axis=0)
                             for k in self.classes_])

        for i, y_i in enumerate(self.classes_):
            diff = X[y == y_i] - self.mu_[i]

            if self.cov_ is None:
                self.cov_ = np.dot(diff.T, diff)
                continue

            self.cov_ += np.dot(diff.T, diff)

        self.cov_ /= X.shape[0] - self.classes_.shape[0]

        self._cov_inv = np.linalg.inv(self.cov_)

        self.pi_ = np.array([X[y == k].size/X.shape[0] for k in self.classes_])

        self.fitted_ = True

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        y_pred = np.zeros(X.shape[0])

        for i, pred in enumerate(np.argmax(self.likelihood(X), axis=1)):
            y_pred[i] = self.classes_[pred]

        return y_pred

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError("Estimator must first be fitted before calling `likelihood` function")

        predictions = list()
        for k in range(self.classes_.shape[0]):
            mu_k = self.mu_[k]  # (1, features)
            mu_k_T = mu_k.T  # (features, 1)
            a_k = self._cov_inv @ mu_k_T  # (features, features) @ (features, 1) = (features, 1)
            b_k = np.log(self.pi_[k]) - 0.5 * (mu_k @ self._cov_inv @ mu_k_T)
            # scalar - 0.5 * ( (1, features) @ (features, features) @ (features, 1) )
            # = scalar - 0.5 * ( (1, features) @ (features, 1) )
            # = scalar - 0.5 * (1, 1)
            # = scalar

            predictions.append(np.array([a_k @ x + b_k for x in X]))

        return np.array(predictions).T

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """

        return loss_functions.misclassification_error(y, self._predict(X))
