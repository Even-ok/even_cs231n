from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):  # for every sample
        scores = X[i].dot(W)   # (10,)  score for every class of 1 sample
        scores -= np.max(scores) 
        p = np.exp(scores) / np.sum(np.exp(scores)) # safe to do, gives the correct answer
        true_label = y[i]
        loss += -np.log(p[true_label])
        for j in range(num_classes):
            if j == y[i]:
                dW[:, j] += (p[j]-1) * X[i]
            else:
                dW[:, j] += p[j] * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 1/2 * reg * np.sum(W * W)   # Default C = 1/2 (\lambda = 1/2C = 1)
    dW += reg * W
    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    num_train = X.shape[0]
    scores = X.dot(W)   # (10,)  score for every class of 1 sample
    margin = np.max(scores, axis=1).reshape(-1, 1)
    scores = scores - margin

    p = np.exp(scores) / np.sum(np.exp(scores), axis=1).reshape(-1, 1) # safe to do, gives the correct answer  (500, 10)
    y = y.astype('int32')
    loss = -np.sum(np.log(p[range(num_train), y]))
    loss /= num_train
    loss += 1/2 * reg * np.sum(W * W) 

    p[range(num_train), y] -= 1
    dW += np.dot(X.T, p)
    dW /= num_train
    dW += reg * W


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
