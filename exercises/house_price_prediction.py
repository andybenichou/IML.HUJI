import os

import pandas
from pandas import DataFrame

from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """

    df = pandas.read_csv(filename)

    # price (int>0), square_foot_price: (price/sqft_living +
    # sqft_non_living * 0.2 + sqft_basement * 0.5) (float>0),
    # square_foot_price_mean_ratio: square_foot_price/square_foot_price_mean (float>0),
    # sqft_above (int>0), sqft_basement (int>=0),
    # sqft_non_living: sqft_lot - sqft_living (int>=0),
    # ratio_living15: sqft_living/sqft_living15 (int>0),
    # ratio_lot15: sqft_lot/sqft_lot15 (int>0),
    # bedrooms (int>0), bathrooms (int>0), floors (int), waterfront (0 or 1),
    # view (0-4), condition (1-5), grade (int>0),
    # built_years (int>=0), renovation_years (int>=0), zipcode (5 digit)

    df = df[(df.price > 0) & (df.bedrooms > 0) & (df.bathrooms > 0) &
            (df.floors > 0) & (df.waterfront.isin([0, 1])) &
            (df.view.isin([0, 1, 2, 3, 4])) &
            (df.condition.isin([1, 2, 3, 4, 5])) & (df.grade > 0) &
            (df.sqft_lot > 0) & (df.sqft_living > 0) & (df.sqft_above >= 0) &
            (df.sqft_basement >= 0) &
            (df.sqft_living == df.sqft_above + df.sqft_basement) &
            (df.yr_renovated >= 0) & (df.yr_renovated <= 2015) &
            (df.yr_built >= 1900) & (df.yr_built <= 2015) &
            (df.zipcode >= 10000) & (df.zipcode <= 99999) &
            (df.sqft_living15 > 0) & (df.sqft_lot15 > 0)]

    df["built_years"] = 2015 - df.yr_built
    df["renovation_years"] = np.where(df.yr_renovated > 1900,
                                      2015 - df.yr_renovated,
                                      df.built_years)
    df["ratio_living15"] = df.sqft_living / df.sqft_living15
    df["ratio_lot15"] = df.sqft_lot / df.sqft_lot15
    df["square_foot_price"] = np.where(df.sqft_lot > df.sqft_living,
                                       df.price /
                                       (df.sqft_above +
                                        (df.sqft_lot - df.sqft_living) * 0.2 +
                                        df.sqft_basement * 0.5),
                                       df.price / (df.sqft_above +
                                                   df.sqft_basement * 0.5))

    df["square_foot_price_zipcode_mean"] = \
        df.groupby('zipcode').square_foot_price.mean()

    mean_price_zipcode = df.groupby('zipcode').square_foot_price.mean()

    df = df.merge(mean_price_zipcode, on='zipcode', how='right')

    df = pd.get_dummies(df, prefix="zipcode", columns=["zipcode"])

    df["square_foot_price"] = df.square_foot_price_x
    df["square_foot_price_zipcode_mean"] = df.square_foot_price_y

    df["square_foot_price_zipcode_ratio"] = \
        df.square_foot_price / df.square_foot_price_zipcode_mean

    df["intercept"] = [1 for i in range(df.shape[0])]

    columns = ["intercept", "price", "square_foot_price",
               "square_foot_price_zipcode_mean",
               "square_foot_price_zipcode_ratio",
               "sqft_above", "sqft_basement", "sqft_lot",
               "ratio_living15", "ratio_lot15", "bedrooms", "bathrooms",
               "floors", "waterfront", "view", "condition", "grade",
               "built_years", "renovation_years"]

    for column in df.columns:
        if "zipcode_9" in column:
            columns.append(column)

    df = df[columns]

    # df.to_csv("data.csv", sep=',')

    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    std_y = y.std()

    for column in X.columns:
        std_x = X[column].std()
        cov = np.cov(X[column], y)[0, 1]

        if std_x * std_y:
            pearson_correlation = cov / (std_x * std_y)

            graph = go.Figure([go.Scatter(x=X[column], y=y, mode='markers')],
                              layout=go.Layout(
                                  title="Pearson correlation (" +
                                        str(pearson_correlation) + "): " +
                                        str(column) + "-" +
                                        "Price : }$",
                                  xaxis_title="X - " + str(column),
                                  yaxis_title="Y - Price"))
            graph.write_image(f"{output_path}/{str(column)}.png")


def question_3(X):
    tr_X, tr_y, t_X, t_y = \
        split_train_test(X.drop(["price"], axis=1), X.price, 0.75)

    # train_X.to_csv("train_X.csv")
    # train_y.to_csv("train_y.csv")
    # test_X.to_csv("test_X.csv")
    # test_y.to_csv("test_y.csv")

    return tr_X, tr_y, t_X, t_y


def question_4(tr_X, tr_y):
    losses_m = list()
    losses_std = list()

    percentages = range(10, 101)
    for p in percentages:
        losses_p = list()
        for i in range(10):
            samples_X = tr_X.sample(frac=p/100)
            samples_y = tr_y[samples_X.index]

            lr = LinearRegression()
            lr.fit(samples_X.to_numpy(), samples_y)

            loss = lr.loss(test_X, test_y)

            losses_p.append(loss)

        losses_m.append(np.mean(losses_p))
        losses_std.append(np.std(losses_p))

    go.Figure([go.Scatter(x=list(percentages), y=losses_m,
                          error_y=
                          dict(type='data', symmetric=True,
                               array=[2 * loss for loss in losses_std]),
                          mode='lines+markers')],
              layout=go.Layout(
                  title="Correlation between % samples, loss mean with std",
                  xaxis_title="% samples",
                  yaxis_title="loss mean")).show()


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of housing prices dataset
    X = load_data("/Users/andybenichou/Documents/Études/Sciences "
                  "Informatiques - HUJI/Année 3/Année 3 - Semestre 2/67577 -"
                  " Introduction to Machine Learning/IML.HUJI/datasets/"
                  "house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    # feature_evaluation(X.drop(["price"], axis=1), X.price,
    #                    "feat_evals")

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = question_3(X)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)

    question_4(train_X, train_y)
