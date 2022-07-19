import time
import numpy as np
import gzip
from typing import Tuple

from IMLearn.metrics.loss_functions import accuracy
from IMLearn.learners.neural_networks.modules import FullyConnectedLayer, ReLU, \
    CrossEntropyLoss, softmax
from IMLearn.learners.neural_networks.neural_network import NeuralNetwork
from IMLearn.desent_methods import GradientDescent, StochasticGradientDescent, \
    FixedLR
from IMLearn.utils.utils import confusion_matrix

import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def load_mnist() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads the MNIST dataset

    Returns:
    --------
    train_X : ndarray of shape (60,000, 784)
        Design matrix of train set

    train_y : ndarray of shape (60,000,)
        Responses of training samples

    test_X : ndarray of shape (10,000, 784)
        Design matrix of test set

    test_y : ndarray of shape (10,000, )
        Responses of test samples
    """

    def load_images(path):
        with gzip.open(path) as f:
            # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
            raw_data = np.frombuffer(f.read(), 'B', offset=16)
        # converting raw data to images (flattening 28x28 to 784 vector)
        return raw_data.reshape(-1, 784).astype('float32') / 255

    def load_labels(path):
        with gzip.open(path) as f:
            # First 8 bytes are magic_number, n_labels
            return np.frombuffer(f.read(), 'B', offset=8)

    return (load_images('../datasets/mnist-train-images.gz'),
            load_labels('../datasets/mnist-train-labels.gz'),
            load_images('../datasets/mnist-test-images.gz'),
            load_labels('../datasets/mnist-test-labels.gz'))


def plot_images_grid(images: np.ndarray, title: str = ""):
    """
    Plot a grid of images

    Parameters
    ----------
    images : ndarray of shape (n_images, 784)
        List of images to print in grid

    title : str, default="
        Title to add to figure

    Returns
    -------
    fig : plotly figure with grid of given images in gray scale
    """
    side = int(len(images) ** 0.5)
    subset_images = images.reshape(-1, 28, 28)

    height, width = subset_images.shape[1:]
    grid = subset_images.reshape(side, side, height, width).swapaxes(1,
                                                                     2).reshape(
        height * side, width * side)

    return px.imshow(grid, color_continuous_scale="gray") \
        .update_layout(
        title=dict(text=title, y=0.97, x=0.5, xanchor="center", yanchor="top"),
        font=dict(size=16), coloraxis_showscale=False) \
        .update_xaxes(showticklabels=False) \
        .update_yaxes(showticklabels=False)


def question_5_or_8(mod, train_X, train_y, test_X, test_y):
    vals, gradients = list(), list()

    def cb(val, grad):
        vals.append(val)
        gradients.append(np.linalg.norm(grad))

    network = NeuralNetwork(modules=mod,
                            loss_fn=CrossEntropyLoss(),
                            solver=StochasticGradientDescent(
                                learning_rate=FixedLR(1e-1),
                                max_iter=10000,
                                batch_size=256,
                                callback=cb))

    network.fit(train_X, train_y)
    print(f"Question 5 : {accuracy(test_y, network.predict(test_X))}")

    return network, vals, gradients


def question_6(vals, grads):
    go.Figure(data=[go.Scatter(x=list(range(len(vals))),
                               y=vals,
                               mode='markers + lines',
                               name="Vals"),
                    go.Scatter(x=list(range(len(grads))),
                               y=grads,
                               mode='markers + lines',
                               name="Gradients")],
              layout=go.Layout(title="Convergence Process")).show()


def question_7(network, test_X, test_y, n_classes):
    go.Figure(data=[go.Heatmap(z=confusion_matrix(network.predict(test_X),
                                                  test_y),
                               x=list(range(n_classes)),
                               y=list(range(n_classes)))],
              layout=go.Layout(
                  title="Test True vs Predicted Confusion Matrix"
              )).show()


def question_9(network, test_X, test_y):
    true_7 = test_X[test_y == 7]
    pred = network.compute_prediction(true_7).argsort()

    plot_images_grid(true_7[pred[-64:]],
                     title="Most Confident").show()
    plot_images_grid(true_7[pred[:64]],
                     title="Least Confident").show()


def question_10():
    pass


if __name__ == '__main__':
    train_X, train_y, test_X, test_y = load_mnist()
    (n_samples, n_features), n_classes = train_X.shape, 10

    # ---------------------------------------------------------------------------------------------#
    # Question 5+6+7: Network with ReLU activations using SGD + recording convergence              #
    # ---------------------------------------------------------------------------------------------#
    # Initialize, fit and test network
    mod = [FullyConnectedLayer(input_dim=n_features,
                               output_dim=64,
                               activation=ReLU(),
                               include_intercept=True),
           FullyConnectedLayer(input_dim=64,
                               output_dim=64,
                               activation=ReLU(),
                               include_intercept=True),
           FullyConnectedLayer(input_dim=64,
                               output_dim=n_classes,
                               include_intercept=True)]

    network, vals, grads = question_5_or_8(mod, train_X, train_y,
                                           test_X, test_y)

    # Plotting convergence process
    question_6(vals, grads)

    # Plotting test true- vs predicted confusion matrix
    question_7(network, test_X, test_y, n_classes)

    # ---------------------------------------------------------------------------------------------#
    # Question 8: Network without hidden layers using SGD                                          #
    # ---------------------------------------------------------------------------------------------#
    mod = [FullyConnectedLayer(input_dim=n_features,
                               output_dim=n_classes,
                               include_intercept=True)]

    network = question_5_or_8(mod, train_X, train_y, test_X, test_y)[0]

    # ---------------------------------------------------------------------------------------------#
    # Question 9: Most/Least confident predictions                                                 #
    # ---------------------------------------------------------------------------------------------#
    question_9(network, test_X, test_y)

    # ---------------------------------------------------------------------------------------------#
    # Question 10: GD vs GDS Running times                                                         #
    # ---------------------------------------------------------------------------------------------#
    question_10()
