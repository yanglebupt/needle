from typing import List, Optional, Tuple
from ..data_basic import Dataset
import struct
import gzip
import numpy as np


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


class MNISTDataset(Dataset):

    def __init__(
        self,
        image_filename: str,
        label_filename: str,
        transforms: Optional[List] = None,
        jobs: Optional[int] = None,
    ):
        ### BEGIN YOUR SOLUTION
        super().__init__(transforms)
        self.X, self.y = parse_mnist(image_filename, label_filename)
        ### END YOUR SOLUTION

    def __getitem__(self, index: int | List[int] | slice) -> Tuple:
        ### BEGIN YOUR SOLUTION
        images = self.X[index].reshape((-1, 28, 28, 1))
        labels = self.y[index]
        # apply transform
        t_images = np.vstack(
            [
                self.apply_transforms(images[i]).reshape((1, 28, 28, 1))
                for i in range(images.shape[0])
            ]
        )
        return t_images, labels
        ### END YOUR SOLUTION

    def __len__(self) -> int:
        ### BEGIN YOUR SOLUTION
        return self.X.shape[0]
        ### END YOUR SOLUTION
