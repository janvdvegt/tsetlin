import numpy as np


def noisy_XOR(n_rows=10000, number_noisy_features=10, flip_fraction=0.4):
    X = np.random.randint(0, 2, size=(n_rows, number_noisy_features + 2))
    y = np.logical_xor(X[:, 0:1], X[:, 1:2])
    flips = np.random.rand(n_rows, 1) < flip_fraction
    y = np.logical_xor(y, flips)
    return X, y


def MNIST():
    from keras.datasets import mnist
    from keras.utils import to_categorical
    (X, y), (val_X, val_y) = mnist.load_data()
    X = np.where(X < 127, 0, 1).reshape(X.shape[0], -1)
    val_X = np.where(val_X < 127, 0, 1).reshape(val_X.shape[0], -1)
    y = to_categorical(y, 10)
    val_y = to_categorical(val_y, 10)
    return X, y, val_X, val_y


def split_train_validate(X, y, n_rows):
    return X[:n_rows], y[:n_rows], X[n_rows:], y[n_rows:]
