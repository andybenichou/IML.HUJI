import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """
    values = list()
    weights_list = list()

    def callback(solver: GradientDescent,
                 weights: np.ndarray,
                 val: np.ndarray,
                 grad: np.ndarray,
                 t: int,
                 eta: float,
                 delta: float):
        weights_list.append(weights)
        values.append(val)

    return callback, values, weights_list


def question_1_3(init, etas):
    def get_module_plots(module_name):
        for eta in etas:

            module = L1(init) if module_name == 'L1' else L2(init)

            callback, values, weights = get_gd_state_recorder_callback()

            module.weights = GradientDescent(FixedLR(eta),
                                             out_type="best",
                                             callback=callback).fit(module,
                                                                    None,
                                                                    None)

            plot_descent_path(L1 if module_name == 'L1' else L2,
                              np.array(weights),
                              f"{module_name} descent path with eta of {eta}"
                              ).show()

            go.Figure([go.Scatter(x=list(range(len(values))),
                                  y=values,
                                  mode='lines+markers')],
                      layout=go.Layout(
                          title=f"L2 norm with eta of {eta}",
                          xaxis_title={"text": "Iterations"},
                          yaxis_title={"text": "Values"})).show()

    get_module_plots("L1")
    get_module_plots("L2")


def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):
    return question_1_3(init, etas)


def question_5(init, eta, gammas):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    traces = []

    for gamma, colors in zip(gammas,
                             [("LightSkyBlue", "mediumblue"),
                              ("lightsalmon", "mediumvioletred"),
                              ("lightseagreen", "forestgreen"),
                              ("lightsteelblue", "darkblue")]):

        callback, values, weights = get_gd_state_recorder_callback()

        GradientDescent(learning_rate=ExponentialLR(base_lr=eta,
                                                    decay_rate=gamma),
                        out_type="best",
                        callback=callback).fit(L1(init), None, None)

        traces.append(go.Scatter(x=list(range(len(values))),
                                 y=values,
                                 mode='lines+markers',
                                 name=f"With gamma of {gamma}",
                                 marker=dict(color=colors[0],
                                             size=0.3,
                                             line=dict(
                                                 color=colors[1],
                                                 width=0.03
                                             ))
                                 )
                      )

    # Plot algorithm's convergence for the different values of gamma
    go.Figure(traces,
              layout=go.Layout(
                  title=f"All decay rates convergence",
                  xaxis_title={"text": "Iterations"},
                  yaxis_title={"text": "Values"})).show()


def question_7(init, eta):
    for module_name in ['L1', 'L2']:
        callback, values, weights = get_gd_state_recorder_callback()

        GradientDescent(learning_rate=ExponentialLR(base_lr=eta,
                                                    decay_rate=0.95),
                        out_type="best",
                        callback=callback).fit((L1
                                                if module_name == 'L1'
                                                else L2)(init),
                                               None, None)

        plot_descent_path(L1 if module_name == 'L1' else L2,
                          np.array(weights),
                          f"{module_name} descent path with gamma of 0.95"
                          ).show()


def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    question_5(init, eta, gammas)

    # Plot descent path for gamma=0.95
    question_7(init, eta)


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    # fit_logistic_regression()
