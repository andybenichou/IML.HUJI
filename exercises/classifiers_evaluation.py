from numpy import ndarray

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
import numpy as np
from typing import Tuple
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.express as px

pio.templates.default = "simple_white"


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """

    load_data = np.load(filename)
    X, y = load_data[:, :2], load_data[:, 2]

    return X, y


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset

        X, y = load_dataset("/Users/andybenichou/Documents/Études/"
                            "Sciences Informatiques - HUJI/Année 3/"
                            "Année 3 - Semestre 2/67577 - Introduction to "
                            "Machine Learning/IML.HUJI/datasets/" + f)

        # Fit Perceptron and record loss in each fit iteration

        losses = []
        perceptron = Perceptron(include_intercept=True,
                                callback=
                                lambda fit, x_i, y_i:
                                losses.append(fit._loss(X, y)))
        perceptron._fit(X, y)

        # Plot figure

        print(losses)
        px.line(x=list(range(1, len(losses) + 1)), y=losses,
                title=n + " Losses during Fitting",
                labels={"x": "iteration", "y": "loss"}).show()


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        raise NotImplementedError()

        # Fit models and predict over training set
        raise NotImplementedError()

        # Plot a figure with two suplots, showing the Gaussian Naive Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and subplot titles should specify algorithm and accuracy
        from IMLearn.metrics import accuracy
        raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    # compare_gaussian_classifiers()
