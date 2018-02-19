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
  dW = 2*reg*W
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    denom = 0.0
    for j in xrange(num_classes):
      denom += np.exp(scores[j])
    for k in xrange(num_classes):
      dW[:,k] += X[i,:]*np.exp(scores[k])/(denom*num_train)  
    loss = loss - np.log(np.exp(correct_class_score)/denom)
    dW[:,y[i]] += -X[i,:]/num_train 
  loss = reg*np.sum(W*W) + loss/num_train
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  dW = 2*reg*W
  num_train = X.shape[0]
  num_classes = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  scores = np.dot(X,W)
  expo = np.exp(scores)
  totals = expo.sum(axis = 1)
  loss += (np.log(totals) - scores[np.arange(num_train),y]).sum()
  loss /= num_train
  loss += reg*np.sum(W*W)

  expo /= np.reshape(totals,(num_train,1))
  expo[np.arange(num_train),y] -= 1  #for the -fyi term
  expo /= num_train
  dW += np.dot(X.T,expo)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW