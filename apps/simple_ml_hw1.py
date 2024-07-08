"""hw1/apps/simple_ml.py"""

import struct
import gzip
import numpy as np
import needle as ndl
import math


def parse_mnist(image_filename, label_filename):
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
                maximum value of 1.0.

            y (numpy.ndarray[dypte=np.int8]): 1D numpy array containing the
                labels of the examples.  Values should be of type np.int8 and
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


def softmax_loss(Z, y_one_hot):
    """Return softmax loss.  Note that for the purposes of this assignment,
    you don't need to worry about "nicely" scaling the numerical properties
    of the log-sum-exp computation, but can just compute this directly.

    Args:
        Z (ndl.Tensor[np.float32]): 2D Tensor of shape
            (batch_size, num_classes), containing the logit predictions for
            each class.
        y (ndl.Tensor[np.int8]): 2D Tensor of shape (batch_size, num_classes)
            containing a 1 at the index of the true label of each example and
            zeros elsewhere.

    Returns:
        Average softmax loss over the sample. (ndl.Tensor[np.float32])
    """
    ### BEGIN YOUR SOLUTION
    B = Z.shape[0]
    return (ndl.log(ndl.exp(Z).sum((1,))).sum() - (y_one_hot * Z).sum()) / B
    ### END YOUR SOLUTION


def nn_epoch(X, y, W1, W2, lr=0.1, batch=100):
    """Run a single epoch of SGD for a two-layer neural network defined by the
    weights W1 and W2 (with no bias terms):
        logits = ReLU(X * W1) * W2
    The function should use the step size lr, and the specified batch size (and
    again, without randomizing the order of X).

    Args:
        X (np.ndarray[np.float32]): 2D input array of size
            (num_examples x input_dim).
        y (np.ndarray[np.uint8]): 1D class label array of size (num_examples,)
        W1 (ndl.Tensor[np.float32]): 2D array of first layer weights, of shape
            (input_dim, hidden_dim)
        W2 (ndl.Tensor[np.float32]): 2D array of second layer weights, of shape
            (hidden_dim, num_classes)
        lr (float): step size (learning rate) for SGD
        batch (int): size of SGD mini-batch

    Returns:
        Tuple: (W1, W2)
            W1: ndl.Tensor[np.float32]
            W2: ndl.Tensor[np.float32]
    """

    ### BEGIN YOUR SOLUTION
    cls_nums = W2.shape[1]
    # 拆分 batch，并遍历
    iterations = math.ceil(X.shape[0] / batch)
    for i in range(iterations):
        X_batch = ndl.Tensor(
            X[i * batch : (i + 1) * batch, :]
        )  # 越界会自动截断，不需要自己判断
        y_batch = y[i * batch : (i + 1) * batch]
        B = X_batch.shape[0]

        # forward
        Z = ndl.relu(X_batch @ W1) @ W2

        # label
        y_one_hot = np.zeros((B, cls_nums))
        y_one_hot[range(B), y_batch] = 1
        y_one_hot = ndl.Tensor(y_one_hot)

        # loss
        loss = softmax_loss(Z, y_one_hot)
        loss.backward()

        # update，这里不能用 Tensor 直接计算，会引入新的图
        # 这里是一个新 Tensor，不是 in-place change 了
        W1 = ndl.Tensor(W1.realize_cached_data() - lr * W1.grad.realize_cached_data())
        W2 = ndl.Tensor(W2.realize_cached_data() - lr * W2.grad.realize_cached_data())

    return W1, W2
    ### END YOUR SOLUTION


### CODE BELOW IS FOR ILLUSTRATION, YOU DO NOT NEED TO EDIT


def loss_err(h, y):
    """Helper function to compute both loss and error"""
    y_one_hot = np.zeros((y.shape[0], h.shape[-1]))
    y_one_hot[np.arange(y.size), y] = 1
    y_ = ndl.Tensor(y_one_hot)
    return softmax_loss(h, y_).numpy(), np.mean(h.numpy().argmax(axis=1) != y)
