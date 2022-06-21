from typing import Tuple, List, Callable, Type

from sklearn.metrics import roc_curve, auc

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.metrics import misclassification_error
from IMLearn.model_selection import cross_validate
from IMLearn.utils import split_train_test

from utils import *

LAMBDAS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


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
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange,
                                       density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1],
                                 mode="markers+lines",
                                 marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[
    Callable[[], None], List[np.ndarray], List[np.ndarray]]:
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


def questions_1_3_4(init, etas):
    def get_module_plots(module_name):
        lowest_loss = [np.inf, np.inf]
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

            if np.min(values) < lowest_loss[0]:
                lowest_loss[0] = np.min(values)

            if module.compute_output() < lowest_loss[1]:
                lowest_loss[1] = module.compute_output()

        print(f"Question 4 - {module_name} lowest loss : {str(lowest_loss)}")

    get_module_plots("L1")
    get_module_plots("L2")


def compare_fixed_learning_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        etas: Tuple[float] = (1, .1, .01, .001)):
    return questions_1_3_4(init, etas)


def questions_5_6(init, eta, gammas):
    # Optimize the L1 objective using different decay-rate values of the exponentially decaying learning rate
    traces = []
    lowest_norm = np.inf

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

        if np.min(values) < lowest_norm:
            lowest_norm = np.min(values)

    # Plot algorithm's convergence for the different values of gamma
    go.Figure(traces,
              layout=go.Layout(
                  title=f"All decay rates convergence",
                  xaxis_title={"text": "Iterations"},
                  yaxis_title={"text": "Values"})).show()

    print(f"Question 6 - Lowest norm for L1 : {lowest_norm}")


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


def compare_exponential_decay_rates(
        init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
        eta: float = .1,
        gammas: Tuple[float] = (.9, .95, .99, 1)):
    questions_5_6(init, eta, gammas)

    # Plot descent path for gamma=0.95
    question_7(init, eta)


def load_data(path: str = "../datasets/SAheart.data",
              train_portion: float = .8) -> \
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
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd,
                            train_portion)


def questions_8_9(X_train, y_train, X_test, y_test):
    # Plotting convergence rate of logistic regression over SA heart disease data
    mod = LogisticRegression(
        solver=GradientDescent(learning_rate=FixedLR(1e-4),
                               max_iter=20000)
    )

    mod._fit(X_train.to_numpy(), y_train.to_numpy())

    y, y_prob = y_train, mod.predict_proba(X_train.to_numpy())
    fpr, tpr, thresholds = roc_curve(y, y_prob)

    go.Figure([go.Scatter(x=[0, 1],
                          y=[0, 1],
                          mode="lines",
                          line=dict(color="black", dash='dash'),
                          name="Random Class Assignment"),
               go.Scatter(x=fpr,
                          y=tpr,
                          mode='markers+lines',
                          text=thresholds,
                          name="",
                          showlegend=False,
                          marker_size=5,
                          marker_color=custom[-1][1],
                          hovertemplate="<b>Threshold:</b>%{text:.3f}<br>FPR: "
                                        "%{x:.3f}<br>TPR: %{y:.3f}")],
              layout=go.Layout(
                  title=rf"$\text{{ROC Curve Of Fitted Model - LR}}="
                        rf"{auc(fpr, tpr):.6f}$",
                  xaxis=dict(title=r"$\text{False Positive Rate "
                                   r"(FPR)}$"),
                  yaxis=dict(title=r"$\text{True Positive Rate "
                                   r"(TPR)}$"))).show()

    # Best alpha
    arg_max = np.argmax(tpr - fpr)
    mod.alpha_ = thresholds[arg_max]

    print(f"Question 9 - Best alpha : {mod.alpha_} ",
          f"Score : {mod._loss(X_test, y_test)}")


def questions_10_11(X_train, y_train, X_test, y_test):
    for module in ["l1", "l2"]:
        question = 10

    scores, losses = list(), list()

    solver = GradientDescent(learning_rate=FixedLR(1e-4),
                             max_iter=20000)

    module = LogisticRegression(penalty=module,
                                solver=solver,
                                alpha=0.5)

    for lam in LAMBDAS:
        module.lam_ = lam
        module._fit(X_train.to_numpy(), y_train.to_numpy())

        scores.append(cross_validate(module,
                                     X_train.to_numpy(),
                                     y_train.to_numpy(),
                                     misclassification_error)[1])
        losses.append(module.loss(X_test.to_numpy(),
                                  y_test.to_numpy()))

    print(f"Question {question} - {module} best lambda : "
          f"{LAMBDAS[np.argmin(scores)]}, "
          f"corresponding loss : {losses[np.argmin(scores)]}")

    question += 1


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # questions_8_9(X_train, y_train, X_test, y_test)

    # Fitting l1- and l2-regularized logistic regression models,
    # using cross-validation to specify values of regularization parameter

    questions_10_11(X_train, y_train, X_test, y_test)


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    # compare_exponential_decay_rates()
    fit_logistic_regression()
