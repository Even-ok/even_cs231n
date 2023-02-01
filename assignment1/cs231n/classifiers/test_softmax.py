import time
from softmax import softmax_loss_naive, softmax_loss_vectorized
import numpy as np

X_dev = np.loadtxt('classifiers\\X_dev.txt')
y_dev = np.loadtxt('classifiers\\y_dev.txt')

W = np.random.randn(3073, 10) * 0.0001

# tic = time.time()
# loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
# toc = time.time()
# print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

tic = time.time()
loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
toc = time.time()
print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))
