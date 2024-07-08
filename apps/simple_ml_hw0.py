from math import ceil
import struct
from typing import Tuple
import numpy as np
import gzip

NDArray = np.ndarray


def add(x, y):
    """A trivial 'add' function you should implement to get used to the
    autograder and submission system.  The solution to this problem is in the
    the homework notebook.

    Args:
        x (Python number or numpy array)
        y (Python number or numpy array)

    Return:
        Sum of x + y
    """
    ### BEGIN YOUR CODE
    return x + y
    ### END YOUR CODE


def parse_mnist(image_filename: str, label_filename: str) -> Tuple[NDArray, NDArray]:
    """Read an images and labels file in MNIST format.  See this page:
    http://yann.lecun.com/exdb/mnist/ for a description of the file format.

    Args:
        image_filename (str): name of gzipped images file in MNIST format
        label_filename (str): name of gzipped labels file in MNIST format

    Returns:
        Tuple (X,y):
            X (numpy.ndarray[np.float32]): 2D numpy array containing the loaded
                data.  The dimensionality of the data should be
                (num_examples x input_dim) where 'input_dim' is the full
                dimension of the data, e.g., since MNIST images are 28x28, it
                will be 784.  Values should be of type np.float32, and the data
                should be normalized to have a minimum value of 0.0 and a
                maximum value of 1.0 (i.e., scale original values of 0 to 0.0
                and 255 to 1.0).

            y (numpy.ndarray[dtype=np.uint8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.uint8 and
                for MNIST will contain the values 0-9.
    """
    ### BEGIN YOUR CODE
    with gzip.open(image_filename, "rb") as f:
        # MSB first, high endian, big endian
        magic_number, image_nums, rows, cols = struct.unpack(">4i", f.read(16))
        assert magic_number == 2051
        pixels = rows * cols
        X = np.vstack(
            [
                np.array(struct.unpack(f"{pixels}B", f.read(pixels)), dtype=np.float32)
                for _ in range(image_nums)
            ]
        )
        X /= 255  # 别忘了归一化

    with gzip.open(label_filename, "rb") as f:
        magic_number, label_nums = struct.unpack(">2i", f.read(8))
        assert magic_number == 2049
        Y = np.array(
            struct.unpack(f"{label_nums}B", f.read(label_nums)), dtype=np.uint8
        )

    return (X, Y)
    ### END YOUR CODE


def softmax_loss(Z: NDArray, y: NDArray) -> float:
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (np.ndarray[np.float32]): 2D numpy array of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (np.ndarray[np.uint8]): 1D numpy array of shape (batch_size, )
            containing the true label of each example.

    Returns:
        Average softmax loss over the sample.
    """
    ### BEGIN YOUR CODE
    B = Z.shape[0]
    return (np.sum(np.log(np.sum(np.exp(Z), axis=1))) - np.sum(Z[range(B), y])) / B
    ### END YOUR CODE


def softmax_regression_epoch(X: NDArray, y: NDArray, theta: NDArray, lr=0.1, batch=100):
    """Run a single epoch of SGD for softmax regression on the data, using
    the step size lr and specified batch size.  This function should modify the
    theta matrix in place, and you should iterate through batches in X _without_
    randomizing the order.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        theta (np.ndarrray[np.float32]): 2D array of softmax regression
            parameters, of shape (input_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # 拆分 batch，并遍历
    iterations = ceil(X.shape[0] / batch)
    for i in range(iterations):
        X_batch = X[i * batch : (i + 1) * batch, :]  # 越界会自动截断，不需要自己判断
        y_batch = y[i * batch : (i + 1) * batch]
        B = X_batch.shape[0]

        Z = np.exp(X_batch @ theta)  # 别忘了 softmax 取指数
        Z = Z / np.sum(Z, axis=1, keepdims=True)
        I_y = np.zeros_like(Z)
        I_y[range(B), y_batch] = 1
        grad: NDArray = X_batch.T @ (Z - I_y) / B

        assert grad.shape == theta.shape
        theta -= lr * grad
    ### END YOUR CODE


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).  It should modify the
    W1 and W2 matrices in place.

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (np.ndarray[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (np.ndarray[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD minibatch

    Returns:
        None
    """
    ### BEGIN YOUR CODE
    # 拆分 batch，并遍历
    iterations = ceil(X.shape[0] / batch)
    for i in range(iterations):
        X_batch = X[i * batch : (i + 1) * batch, :]  # 越界会自动截断，不需要自己判断
        y_batch = y[i * batch : (i + 1) * batch]
        B = X_batch.shape[0]

        z1 = X_batch @ W1
        z1[z1 < 0] = 0
        G2 = np.exp(z1 @ W2)
        G2 = G2 / np.sum(G2, axis=1, keepdims=True)
        Y = np.zeros_like(G2)
        Y[range(B), y_batch] = 1
        G2 -= Y
        G1 = np.zeros_like(z1)
        G1[z1 > 0] = 1
        G1 = G1 * (G2 @ W2.T)
        grad1 = X_batch.T @ G1 / B
        grad2 = z1.T @ G2 / B

        assert grad1.shape == W1.shape
        assert grad2.shape == W2.shape
        W1 -= lr * grad1
        W2 -= lr * grad2
    ### END YOUR CODE


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper funciton to compute both loss and error"""
    return softmax_loss(h, y), np.mean(h.argmax(axis=1) != y)


def train_softmax(X_tr, y_tr, X_te, y_te, epochs=10, lr=0.5, batch=100, cpp=False):
    """Example function to fully train a softmax regression classifier"""
    theta = np.zeros((X_tr.shape[1], y_tr.max() + 1), dtype=np.float32)
    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        if not cpp:
            softmax_regression_epoch(X_tr, y_tr, theta, lr=lr, batch=batch)
        else:
            # softmax_regression_epoch_cpp(X_tr, y_tr, theta, lr=lr, batch=batch)
            pass
        train_loss, train_err = loss_err(X_tr @ theta, y_tr)
        test_loss, test_err = loss_err(X_te @ theta, y_te)
        print(
            "|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |".format(
                epoch, train_loss, train_err, test_loss, test_err
            )
        )


def train_nn(X_tr, y_tr, X_te, y_te, hidden_dim=500, epochs=10, lr=0.5, batch=100):
    """Example function to train two layer neural network"""
    n, k = X_tr.shape[1], y_tr.max() + 1
    np.random.seed(0)
    W1 = np.random.randn(n, hidden_dim).astype(np.float32) / np.sqrt(hidden_dim)
    W2 = np.random.randn(hidden_dim, k).astype(np.float32) / np.sqrt(k)

    print("| Epoch | Train Loss | Train Err | Test Loss | Test Err |")
    for epoch in range(epochs):
        nn_epoch(X_tr, y_tr, W1, W2, lr=lr, batch=batch)
        train_loss, train_err = loss_err(np.maximum(X_tr @ W1, 0) @ W2, y_tr)
        test_loss, test_err = loss_err(np.maximum(X_te @ W1, 0) @ W2, y_te)
        print(
            "|  {:>4} |    {:.5f} |   {:.5f} |   {:.5f} |  {:.5f} |".format(
                epoch, train_loss, train_err, test_loss, test_err
            )
        )
