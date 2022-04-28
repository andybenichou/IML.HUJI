from typing import NoReturn
from ...base import BaseEstimator
import numpy as np

from ...metrics import loss_functions


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """
    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

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

        self.vars_ = np.array([np.var(X[y == k], axis=0)
                               for k in self.classes_])

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

        predictions = list()  # (samples, classes)
        for k in range(self.classes_.shape[0]):
            cov_k = np.diag(self.vars_[k])  # (features, features)
            cov_k_det = np.linalg.det(cov_k)  # (1, 1)
            d = X.shape[1]
            mu_k = self.mu_[k]  # (1, features)
            pi_k = self.pi_[k]  # (1, 1)

            denominator = np.sqrt(((2 * np.pi) ** d) * cov_k_det)  # (classes, 1)

            y_pred = list()  # (samples, 1)
            for x in X:
                x_mu_k = x - mu_k  # (1, features)

                numerator = np.exp(-0.5 * x_mu_k
                                   @ np.linalg.inv(cov_k)
                                   @ x_mu_k.T) * pi_k
                # np.exp(- 0.5 * ( (1, features) @ (features, features) @ (features, 1) )) * scalar
                # = np.exp(- 0.5 * ( (1, features) @ (features, 1) )) * scalar
                # = np.exp(- 0.5 * ( (1, 1) )) * scalar
                # = scalar

                y_pred.append(numerator / denominator)

            predictions.append(y_pred)

        return np.array(predictions).T  # (classes, samples)

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
