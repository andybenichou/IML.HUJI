import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def Question_1(train_X, train_y, test_X, test_y, n_learners, noise):
    # Question 1: Train- and test errors of AdaBoost in noiseless case

    adaboost = AdaBoost(DecisionStump, n_learners)
    adaboost.fit(train_X, train_y)

    losses = list()

    for t in range(1, n_learners):
        train_loss = adaboost.partial_loss(train_X, train_y, T=t)
        test_loss = adaboost.partial_loss(test_X, test_y, T=t)
        losses.append((train_loss, test_loss))

    losses = np.array(losses)
    x = list(range(n_learners))

    go.Figure([
        go.Scatter(x=x,
                   y=losses[:, 0],
                   mode='lines',
                   name=r'Training errors'),
        go.Scatter(x=x,
                   y=losses[:, 1],
                   mode='lines',
                   name=r'Test errors')]) \
        .update_layout(title=f"Training and test errors as a function of the "
                             f"number of fitted learners with noise = {noise}",
                       xaxis=dict(title="number of fitted learners",
                                  ticktext=x),
                       yaxis_title="errors rate")\
        .show()

    return adaboost, losses[:, 0], losses[:, 1]


def Question_2(train_X, test_X, test_y, noise, adaboost):
    # Question 2: Plotting decision surfaces

    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0), np.r_[train_X, test_X].max(axis=0)]).T + np.array([-.1, .1])

    data = [decision_surface(lambda x: adaboost.partial_predict(x, t), *lims,
                             showscale=False, dotted=False)
            for t in T]

    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[f"With {i} iterations"
                                        for i in T])

    for i, d in enumerate(data):
        fig.add_traces([d, go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                      mode="markers", showlegend=False,
                                      marker=dict(color=test_y,
                                                  colorscale=custom))],
                       rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(title=rf"Decision Boundaries with noise = {noise}")
    fig.show()

    return lims


def Question_3(test_X, test_y, noise, adaboost, test_loss, lims):
    # Question 3: Decision surface of best performing ensemble

    min_loss_classifier = np.argmin(test_loss) + 1
    accuracy = 1 - test_loss[min_loss_classifier - 1]

    dec_surf = decision_surface(lambda x:
                                adaboost.partial_predict(x,
                                                         min_loss_classifier),
                                *lims, showscale=False, dotted=False)

    fig = go.Figure([dec_surf, go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                          mode="markers", showlegend=False,
                                          marker=dict(color=test_y,
                                                      colorscale=custom))])

    fig.update_layout(title=rf"Best decision boundaries with noise = {noise}, "
                            rf"with ensemble size = {min_loss_classifier} and "
                            rf"accuracy = {accuracy}")

    fig.show()


def Question_4(train_X, train_y, noise, adaboost, lims):
    # Question 4: Decision surface with weighted samples

    #TODO: size_factor * adaboost.D_ / np.max(adaboost.D_) ??

    dec_surf = decision_surface(lambda x: adaboost._predict(x),
                                *lims, showscale=False, dotted=False)

    fact = 50 if noise == 0 else 15

    fig = go.Figure([dec_surf, go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                          mode="markers", showlegend=False,
                                          marker=dict(color=train_y,
                                                      symbol=class_symbols[train_y.astype(int)],
                                                      colorscale=[custom[0], custom[-1]],
                                                      size=fact * adaboost.D_ / np.max(adaboost.D_),
                                                      line=dict(width=0.5,
                                                                color="black")
                                                      ))])
    fig.update_layout(title=rf"Weighted train decision boundaries with "
                            rf"noise = {noise}")
    fig.show()


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000, test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size, noise), generate_data(test_size, noise)

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost, train_loss, test_loss = \
        Question_1(train_X, train_y, test_X, test_y, n_learners, noise)

    # Question 2: Plotting decision surfaces
    lims = Question_2(train_X, test_X, test_y, noise, adaboost)

    # Question 3: Decision surface of best performing ensemble
    Question_3(test_X, test_y, noise, adaboost, test_loss, lims)

    # Question 4: Decision surface with weighted samples
    Question_4(train_X, train_y, noise, adaboost, lims)


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(noise=0)

    # Question 5 : Repeat the steps with noise levels of 0.4
    fit_and_evaluate_adaboost(noise=0.4)
