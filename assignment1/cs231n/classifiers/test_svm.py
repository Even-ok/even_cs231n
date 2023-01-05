from linear_svm import svm_loss_vectorized
import numpy as np


if __name__ == '__main__':
    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001 

    X_dev = np.loadtxt('./classifiers/X_dev.txt')
    y_dev = np.loadtxt('./classifiers/y_dev.txt')
    # Compute the loss and its gradient at W.
    loss, grad = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)