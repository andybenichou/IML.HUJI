from __future__ import annotations
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go


def f_x(x):
    return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)


def question_1(n_samples: int = 100, noise: float = 5):
    samples = np.linspace(-1.2, 2, n_samples)
    f_X = f_x(samples)
    noisy_labels = f_X + np.random.normal(0, noise, n_samples)

    X_train, y_train, X_test, y_test = split_train_test(pd.DataFrame(samples),
                                                        pd.Series(
                                                            noisy_labels),
                                                        train_proportion=2 / 3)

    X_train, X_test = X_train.to_numpy().flatten(), X_test.to_numpy().flatten()
    y_train, y_test = y_train.to_numpy(), y_test.to_numpy()

    go.Figure(
        [go.Scatter(x=samples,
                    y=f_X,
                    mode="markers",
                    name="true",
                    marker=dict(color='LightSkyBlue',
                                size=20,
                                line=dict(
                                    color='mediumblue',
                                    width=2
                                ))),

         go.Scatter(x=X_train,
                    y=y_train,
                    mode="markers",
                    name="train",
                    marker=dict(color='lightsalmon',
                                size=20,
                                line=dict(
                                    color='mediumvioletred',
                                    width=2
                                ))),

         go.Scatter(x=X_test,
                    y=y_test,
                    mode="markers",
                    name="test",
                    marker=dict(color='lightseagreen',
                                size=20,
                                line=dict(
                                    color='forestgreen',
                                    width=2
                                ))),
         ],
        layout=go.Layout(
            title=f"With {n_samples} samples and noise of {noise}")
    ).show()

    return X_train, y_train, X_test, y_test


def question_2(X_train, y_train, n_samples: int = 100, noise: float = 5):
    Xs = list(range(11))
    all_scores = np.array([cross_validate(PolynomialFitting(i),
                                          X_train, y_train,
                                          mean_square_error)
                           for i in Xs])

    test_scores, validation_scores = all_scores[:, 0], all_scores[:, 1]

    go.Figure(
        [go.Scatter(x=Xs,
                    y=test_scores,
                    mode="lines+markers",
                    name="Train Scores",
                    marker=dict(color='LightSkyBlue',
                                size=20,
                                line=dict(
                                    color='mediumblue',
                                    width=2
                                ))),

         go.Scatter(x=Xs,
                    y=validation_scores,
                    mode="lines+markers",
                    name="Validation Scores",
                    marker=dict(color='lightsalmon',
                                size=20,
                                line=dict(
                                    color='mediumvioletred',
                                    width=2
                                ))),
         ],
        layout=go.Layout(title=f"Train Scores and Validation Scores with "
                               f"{n_samples} samples and noise of {noise}",
                         xaxis_title=dict(text="K"),
                         yaxis_title=dict(text="MSE"))
    ).show()

    return test_scores, validation_scores


def question_3(validation_scores,
               X_train, y_train, X_test, y_test,
               n_samples: int = 100, noise: float = 5):
    def round_with_2_decimals(value):
        return round(value, 2)

    k_min = int(np.argmin(validation_scores))
    PolynomialFitting(k_min).fit(X_train, y_train)
    MSE = mean_square_error(y_test,
                            PolynomialFitting(k_min).
                            fit(X_train, y_train).
                            predict(X_test))
    test_error = round_with_2_decimals(MSE)
    validation_error = round_with_2_decimals(validation_scores[k_min])

    print(f"---------- With {n_samples} samples and noise of {noise}, "
          f"we got : ----------")

    print(f"- k* = {k_min}")
    print(f"- Test error of the fitted model = {test_error}")
    print(f"- Validation error = {validation_error}\n\n")


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions

    X_train, y_train, X_test, y_test = question_1(n_samples, noise)

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10

    test_scores, validation_scores = question_2(X_train, y_train,
                                                n_samples, noise)

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    question_3(validation_scores,
               X_train, y_train, X_test, y_test,
               n_samples, noise)


def question_6(n_samples: int = 50):
    X, y = datasets.load_diabetes(return_X_y=True)

    return X[:n_samples], y[:n_samples], X[n_samples:], y[n_samples:]


def question_7(X_train, y_train,
               n_samples: int = 50,
               n_evaluations: int = 500):
    lams = np.linspace(0, 2, n_evaluations)

    models = [
        {
            "model": RidgeRegression,
            "validation_scores": list(),
            "text": "Ridge"
        },
        {
            "model": Lasso,
            "validation_scores": list(),
            "text": "Lasso",
        }
    ]

    for mod in models:
        all_scores = np.array([
            cross_validate(mod["model"](lam), X_train, y_train,
                           mean_square_error)
            for lam in lams
        ])

        train_scores = all_scores[:, 0]
        mod["validation_scores"] = all_scores[:, 1]

        go.Figure(
            [go.Scatter(x=lams,
                        y=train_scores,
                        mode="lines+markers",
                        name="Train Scores",
                        marker=dict(color='LightSkyBlue',
                                    size=3,
                                    line=dict(
                                        color='mediumblue',
                                        width=0.3
                                    ))),

             go.Scatter(x=lams,
                        y=mod["validation_scores"],
                        mode="lines+markers",
                        name="Validation Scores",
                        marker=dict(color='lightsalmon',
                                    size=3,
                                    line=dict(
                                        color='mediumvioletred',
                                        width=0.3
                                    ))),
             ],
            layout=go.Layout(title=f"Train Scores and Validation Scores of "
                                   f"{mod['text']} with {n_samples} samples",
                             xaxis_title=dict(text="Lambda"),
                             yaxis_title=dict(text="MSE"))
        ).show()

    return lams, models[0]["validation_scores"], models[1]["validation_scores"]


def question_8(lams, ridge_validation_scores, lasso_validation_scores,
               X_train, y_train, X_test, y_test):
    for mod in [
        {
            "model": RidgeRegression,
            "validation_scores": ridge_validation_scores,
            "text": "Ridge"
        },
        {
            "model": Lasso,
            "validation_scores": lasso_validation_scores,
            "text": "Lasso",
        }
    ]:
        print(f"---------- {mod['text']} Model, we got : ----------")

        min_validation = lams[np.argmin(mod["validation_scores"])]
        print(f"Minimal Validation Error : {min_validation}")

        MSE = mean_square_error(y_test,
                                mod["model"](min_validation).
                                fit(X_train, y_train).
                                predict(X_test))
        print(f"Test MSE : {MSE}")

    linear_regression_loss = \
        LinearRegression().fit(X_train,
                               y_train).fit(X_train,
                                            y_train).loss(X_test,
                                                          y_test)
    print(f"Linear Regression loss : {linear_regression_loss}")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X_train, y_train, X_test, y_test = question_6(n_samples)

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    lams, ridge_validation_scores, lasso_validation_scores = \
        question_7(X_train, y_train, n_samples, n_evaluations)

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    question_8(lams, ridge_validation_scores, lasso_validation_scores,
               X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)

    select_regularization_parameter()
