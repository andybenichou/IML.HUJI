import numpy as np
import pandas as pd
from typing import Tuple, List
from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, \
    CrossEntropyLoss
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, FixedLR
from IMLearn.utils.utils import split_train_test
from utils import *

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

pio.templates.default = "simple_white"


def generate_nonlinear_data(
        samples_per_class: int = 100,
        n_features: int = 2,
        n_classes: int = 2,
        train_proportion: float = 0.8) -> \
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a multiclass non linearly-separable dataset. Adopted from Stanford CS231 course code.

    Parameters:
    -----------
    samples_per_class: int, default = 100
        Number of samples per class

    n_features: int, default = 2
        Data dimensionality

    n_classes: int, default = 2
        Number of classes to generate

    train_proportion: float, default=0.8
        Proportion of samples to be used for train set

    Returns:
    --------
    train_X : ndarray of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : ndarray of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : ndarray of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : ndarray of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    X, y = np.zeros((samples_per_class * n_classes, n_features)), np.zeros(
        samples_per_class * n_classes, dtype='uint8')
    for j in range(n_classes):
        ix = range(samples_per_class * j, samples_per_class * (j + 1))
        r = np.linspace(0.0, 1, samples_per_class)  # radius
        t = np.linspace(j * 4, (j + 1) * 4,
                        samples_per_class) + np.random.randn(
            samples_per_class) * 0.2  # theta
        X[ix], y[ix] = np.c_[r * np.sin(t), r * np.cos(t)], j

    split = split_train_test(pd.DataFrame(X), pd.Series(y), train_proportion)
    return tuple(map(lambda x: x.values, split))


def plot_decision_boundary(nn: NeuralNetwork, lims, X: np.ndarray = None,
                           y: np.ndarray = None, title=""):
    data = [decision_surface(nn.predict, lims[0], lims[1], density=40,
                             showscale=False)]
    if X is not None:
        col = y if y is not None else "black"
        data += [go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                            marker=dict(color=col, colorscale=custom,
                                        line=dict(color="black", width=1)))]

    return go.Figure(data,
                     go.Layout(
                         title=rf"$\text{{Network Decision Boundaries {title}}}$",
                         xaxis=dict(title=r"$x_1$"),
                         yaxis=dict(title=r"$x_2$"),
                         width=400, height=400))


def animate_decision_boundary(nn: NeuralNetwork, weights: List[np.ndarray],
                              lims, X: np.ndarray, y: np.ndarray,
                              title="", save_name=None):
    frames = []
    for i, w in enumerate(weights):
        nn.weights = w
        frames.append(go.Frame(data=[
            decision_surface(nn.predict, lims[0], lims[1], density=40,
                             showscale=False),
            go.Scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                       marker=dict(color=y, colorscale=custom,
                                   line=dict(color="black", width=1)))
            ],
                               layout=go.Layout(
                                   title=rf"$\text{{{title} Iteration {i + 1}}}$")))

    fig = go.Figure(data=frames[0]["data"], frames=frames[1:],
                    layout=go.Layout(title=frames[0]["layout"]["title"]))
    if save_name:
        animation_to_gif(fig, save_name, 200, width=400, height=400)


def question_1(train_X, train_y, test_X, test_y, n_features, lims):
    mod = [FullyConnectedLayer(input_dim=n_features,
                               output_dim=16,
                               activation=ReLU(),
                               include_intercept=True),
           FullyConnectedLayer(input_dim=16,
                               output_dim=16,
                               activation=ReLU(),
                               include_intercept=True),
           FullyConnectedLayer(input_dim=16,
                               output_dim=n_classes,
                               include_intercept=True)]

    network = NeuralNetwork(modules=mod,
                            loss_fn=CrossEntropyLoss(),
                            solver=GradientDescent(learning_rate=FixedLR(1e-1),
                                                   max_iter=5000))
    network.fit(train_X, train_y)

    print("Question 1 : ", accuracy(test_y, network.predict(test_X)))

    plot_decision_boundary(network,
                           lims,
                           train_X,
                           train_y,
                           title="Question 1").show()


def question_2(train_X, train_y, test_X, test_y, n_features, n_classes, lims):
    mod = [FullyConnectedLayer(input_dim=n_features,
                               output_dim=n_classes,
                               include_intercept=True)]

    network = NeuralNetwork(modules=mod,
                            loss_fn=CrossEntropyLoss(),
                            solver=GradientDescent(learning_rate=FixedLR(1e-1),
                                                   max_iter=5000))
    network.fit(train_X, train_y)

    print("Question 2 : ", accuracy(test_y, network.predict(test_X)))

    plot_decision_boundary(network,
                           lims,
                           train_X,
                           train_y,
                           title="Question 2").show()


def question_3_or_4(mod, train_X, train_y, lims, question):
    vals, gradients, weights = list(), list(), list()

    def cb(val, grad, w, **kwargs):
        vals.append(val)
        gradients.append(np.linalg.norm(grad))
        weights.append(w)

    network = NeuralNetwork(modules=mod,
                            loss_fn=CrossEntropyLoss(),
                            solver=GradientDescent(learning_rate=FixedLR(1e-1),
                                                   max_iter=5000,
                                                   callback=cb))
    network.fit(train_X, train_y)

    go.Figure(data=[go.Scatter(x=list(range(len(vals))),
                               y=vals,
                               mode='markers + lines',
                               name="Errors"),
                    go.Scatter(x=list(range(len(gradients))),
                               y=gradients,
                               mode='markers + lines',
                               name="Gradients")],
              layout=go.Layout(title=f"{question}",
                               xaxis=dict(title="Iterations"),
                               yaxis=dict(title="Values"))).show()

    animate_decision_boundary(network,
                              weights[::100],
                              lims,
                              train_X,
                              train_y,
                              save_name=f"{question}.gif")


def questions_3_4(train_X, train_y, n_features, n_classes, lims):
    question_3_or_4([FullyConnectedLayer(input_dim=n_features,
                                         output_dim=16,
                                         activation=ReLU(),
                                         include_intercept=True),
                     FullyConnectedLayer(input_dim=16,
                                         output_dim=16,
                                         activation=ReLU(),
                                         include_intercept=True),
                     FullyConnectedLayer(input_dim=16,
                                         output_dim=n_classes,
                                         include_intercept=True)],
                    train_X, train_y, lims, "Question 3")

    question_3_or_4([FullyConnectedLayer(input_dim=n_features,
                                         output_dim=6,
                                         activation=ReLU(),
                                         include_intercept=True),
                     FullyConnectedLayer(input_dim=6,
                                         output_dim=6,
                                         activation=ReLU(),
                                         include_intercept=True),
                     FullyConnectedLayer(input_dim=6,
                                         output_dim=n_classes,
                                         include_intercept=True)],
                    train_X, train_y, lims, "Question 4")


if __name__ == '__main__':
    np.random.seed(0)

    # Generate and visualize dataset
    n_features, n_classes = 2, 3
    train_X, train_y, test_X, test_y = generate_nonlinear_data(
        samples_per_class=500, n_features=n_features, n_classes=n_classes,
        train_proportion=0.8)
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    go.Figure(
        data=[go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode='markers',
                         marker=dict(color=train_y, colorscale=custom,
                                     line=dict(color="black", width=1)))],
        layout=go.Layout(title=r"$\text{Train Data}$",
                         xaxis=dict(title=r"$x_1$"),
                         yaxis=dict(title=r"$x_2$"),
                         width=400, height=400)) \
        .show()

    # ---------------------------------------------------------------------------------------------#
    # Question 1: Fitting simple network with two hidden layers                                    #
    # ---------------------------------------------------------------------------------------------#
    question_1(train_X, train_y, test_X, test_y, n_features, lims)

    # ---------------------------------------------------------------------------------------------#
    # Question 2: Fitting a network with no hidden layers                                          #
    # ---------------------------------------------------------------------------------------------#
    # question_2(train_X, train_y, test_X, test_y, n_features, n_classes, lims)
    #
    # # ---------------------------------------------------------------------------------------------#
    # # Question 3+4: Plotting network convergence process                                           #
    # # ---------------------------------------------------------------------------------------------#
    # questions_3_4(train_X, train_y, n_features, n_classes, lims)
