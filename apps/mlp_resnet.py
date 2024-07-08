import needle as ndl
import needle.nn as nn
from needle.data import MNISTDataset ,DataLoader
import numpy as np
import time
import os

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    ### BEGIN YOUR SOLUTION
    block_fn = nn.Sequential(
        *[
            nn.Linear(dim, hidden_dim),
            norm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(hidden_dim, dim),
            norm(dim),
        ]
    )
    return nn.Sequential(nn.Residual(block_fn), nn.ReLU())
    ### END YOUR SOLUTION


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    ### BEGIN YOUR SOLUTION
    module_list = [nn.Linear(dim, hidden_dim),nn.ReLU(),]
    module_list += [
        ResidualBlock(hidden_dim, hidden_dim//2, norm=norm, drop_prob=drop_prob) for _ in range(num_blocks)
    ]
    module_list.append(nn.Linear(hidden_dim, num_classes))
    return nn.Sequential(*module_list)
    ### END YOUR SOLUTION


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    tot_loss, tot_error = [], 0.0
    loss_fn = nn.SoftmaxLoss()
    trainning = opt is not None
    if trainning:
        model.train()
    else:
        model.eval()
    for X, y in dataloader:
        h = model(X.reshape((X.shape[0],-1)))
        loss = loss_fn(h, y)
        tot_error += np.sum(h.numpy().argmax(axis=1)!=y.numpy())
        tot_loss.append(loss.numpy())
        if trainning:
            opt.reset_grad()
            loss.backward()
            opt.step()

    return tot_error / len(dataloader.dataset), np.mean(tot_loss)
    ### END YOUR SOLUTION


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="./apps/data",
):
    np.random.seed(4)
    ### BEGIN YOUR SOLUTION
    model = MLPResNet(28*28, hidden_dim=hidden_dim)
    opt = optimizer(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_dataset = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                                 f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_dataset = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    for _ in range(epochs):
        train_error, train_loss = epoch(train_loader, model, opt)
    test_error, test_loss = epoch(test_loader, model, None)
    return train_error, train_loss, test_error, test_loss
    ### END YOUR SOLUTION


if __name__ == "__main__":
    train_mnist(data_dir="./apps/data")
