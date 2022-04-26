import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
pio.templates.default = "simple_white"


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=["Date"])
    df = df.dropna()
    df = df[df.Temp >= -40]

    df['DayOfYear'] = df['Date'].dt.dayofyear
    df = df.drop(["Date", "Day"], axis=1)

    # df.to_csv("dt.csv", index=True, sep=",")

    return df


def question_2(data):
    df_israel = data[data.Country == "Israel"]

    px.scatter(x=df_israel["DayOfYear"], y=df_israel["Temp"],
               color=df_israel.Year.astype(str),
               title="Change daily temperature according to day of the"
                     " year in Israel",
               labels={"x": "day of the year", "y": "temperature"}).show()

    df_israel_month = df_israel.groupby("Month")
    df_israel_month_std = df_israel_month.agg("std")

    months = list(set(df_israel.Month))
    months.sort()

    px.bar(x=months,
           y=df_israel_month_std.Temp,
           title="Standard deviation by month in Israel",
           labels={"x": "months", "y": "temperature"}).show()


def question_3(data):
    df_country_month = data.groupby(["Country", "Month"])
    df_country_month_mean = df_country_month.agg("mean")
    df_country_month_std = df_country_month.agg("std")

    df_country_month_std = df_country_month_std.reset_index()

    px.line(x=df_country_month_std.Month, y=df_country_month_mean.Temp,
            color=df_country_month_std.Country,
            error_y=df_country_month_std.Temp,
            title="Mean temperature by Month for each country with std "
                  "calculated error bars",
            labels={"x": "months", "y": "temperature"}).show()


def question_4(data):
    df_israel = data[data.Country == "Israel"]

    train_X, train_y, test_X, test_y = \
        split_train_test(df_israel.DayOfYear,
                         df_israel.Temp, 0.75)
    losses = list()
    k = 1
    while k < 11:
        polynomial_model = PolynomialFitting(k)
        polynomial_model._fit(train_X.to_numpy(), train_y.to_numpy())

        pol_loss = polynomial_model._loss(test_X,
                                          test_y)
        losses.append(round(pol_loss, 2))

        k += 1

    px.bar(x=list(range(1, 11)), y=losses,
           title="Loss for each k between 1 and 11",
           labels={"x": "k", "y": "loss"}).show()


def question_5(data):
    df_israel = data[data.Country == "Israel"]

    polynomial_model = PolynomialFitting(5)
    polynomial_model._fit(df_israel.DayOfYear, df_israel.Temp)

    losses = list()
    for c in ['South Africa','The Netherlands', 'Jordan']:
        df_country = data[data.Country == c]
        pred = polynomial_model._predict(df_country.DayOfYear)

        losses.append(
            round(polynomial_model._loss(pred, df_country.Temp), 2))

    px.bar(x=['South Africa','The Netherlands', 'Jordan'], y=losses,
           title="Loss for k=5 for each country",
           labels={"x": f"country", "y": "loss"}).show()


if __name__ == '__main__':
    np.random.seed(0)

    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("/Users/andybenichou/Documents/Études/Sciences "
                     "Informatiques - HUJI/Année 3/Année 3 - Semestre 2/67577"
                     " - Introduction to Machine Learning/IML.HUJI/datasets/"
                     "City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    question_2(data)

    # Question 3 - Exploring differences between countries

    question_3(data)

    # Question 4 - Fitting model for different values of `k`

    question_4(data)

    # Question 5 - Evaluating fitted model on different countries

    question_5(data)
