from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    # Question 1 - Draw samples and print fitted model
    X, Y = enumerate(np.random.normal(10, 1, size=1000))

    go.Figure([go.Scatter(y=X, mode='lines', marker=dict(color="black"), showlegend=False)],
              layout=go.Layout(title=r"$\text{(5) Estimation of Expectation As Function Of Number Of Samples}$",
                               xaxis_title="$m\\text{ - number of samples}$",
                               yaxis_title="r$\hat\mu$",
                               height=300)).show()

    univariate_gaussian = UnivariateGaussian(X)
    univariate_gaussian.fit()

    print(f"({univariate_gaussian.mu_}, {univariate_gaussian.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    raise NotImplementedError()

    # Question 3 - Plotting Empirical PDF of fitted model
    raise NotImplementedError()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
