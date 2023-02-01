from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength  1 or 2?

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    dW = np.zeros(W.shape)  # initialize the gradient as zero  (3073, 10)

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):  # for every sample
        scores = X[i].dot(W)   # (10,)  score for every class of 1 sample
        correct_class_score = scores[int(y[i])] # (1,) 
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1   hinge loss
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]  # For every class, sum them。前面有一个累加的符号，对于loss是这样，对于梯度也是这样。  意思是第i个样本，对于第j个W的梯度
                dW[:, int(y[i])] += -X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += 1/2 * reg * np.sum(W * W)   # Default C = 1/2 (\lambda = 1/2C = 1)
    dW += reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store in dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0.0
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the structured SVM loss, storing the    #
    # result in loss.                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    num_train = X.shape[0]
    num_classes = X.shape[1]
    scores = X.dot(W)  # (500, 10)
    
    y = y.astype('int32')
    correct_score = scores[np.arange(num_train), y].reshape(num_train, 1)  # 变成二维数组（一列），广播
    loss_class_wise = np.maximum(0, scores - correct_score + 1)
    loss_class_wise[np.arange(num_train), y] = 0
    loss = np.sum(loss_class_wise) / num_train

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    #############################################################################
    # TODO:                                                                     #
    # Implement a vectorized version of the gradient for the structured SVM     #
    # loss, storing the result in dW.                                           #
    #                                                                           #
    # Hint: Instead of computing the gradient from scratch, it may be easier    #
    # to reuse some of the intermediate values that you used to compute the     #
    # loss.                                                                     #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    margins = loss_class_wise
    margins[margins > 0] = 1.0                         # 示性函数的意义
    row_sum = np.sum(margins, axis=1)                  # 1 by N
    margins[np.arange(num_train), y] = -row_sum        # (500,)
    dW += np.dot(X.T, margins)/num_train + 2 * reg * W     # D by C

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
