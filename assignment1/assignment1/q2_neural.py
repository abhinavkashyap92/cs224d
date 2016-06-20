import numpy as np
import random
import sys

from q1_softmax import softmax
from q2_sigmoid import sigmoid, sigmoid_grad
from q2_gradcheck import gradcheck_naive

def forward_backward_prop(data, labels, params, dimensions, only_forward=False):
    """
    Forward and backward propagation for a two-layer sigmoidal network

    Compute the forward propagation and for the cross entropy cost,
    and backward propagation for the gradients for all parameters.

    Input:
        data: N * Dx
        labels: N * Dy # The proper labels for all the data
        labels have one at the indexes where it is the true class
        params: All the parameters packed
        dimensions: A list [Dx, H, Dy]
    """

    ### Unpack network parameters (do not modify)
    ofs = 0
    Dx, H, Dy = (dimensions[0], dimensions[1], dimensions[2])
    X = data
    N, Dx = X.shape

    ############################################################################
    # X = data = N * Dx
    # W1 of shape(Dx, H)
    # b1 of shape (1, H)
    # W2 of shape (H, Dy)
    # b2 of shape (1, Dy)
    ############################################################################
    W1 = np.reshape(params[ofs:ofs+ Dx * H], (Dx, H))
    ofs += Dx * H
    b1 = np.reshape(params[ofs:ofs + H], (1, H))
    ofs += H
    W2 = np.reshape(params[ofs:ofs + H * Dy], (H, Dy))
    ofs += H * Dy
    b2 = np.reshape(params[ofs:ofs + Dy], (1, Dy))
    gradW1 = np.zeros(W1.shape)
    gradW2 = np.zeros(W2.shape)
    gradb1 = np.zeros(b1.shape)
    gradb2 = np.zeros(b2.shape)


    # dot1 = X.W1 - shape(N * H)
    dot1 = X.dot(W1)

    # add1 = dot1 + b1  - shape (N * H)
    add1 = dot1 + b1

    # h1 = sigmoid(add1) - shape(N * H)
    h1 = sigmoid(add1)

    # dot2 = h1.dot(W2) - shape(N * Dy)
    dot2 =  h1.dot(W2)

    # add2 = dot2 + b2 - shape(N * Dy)
    add2 = dot2 + b2

    # scores = softmax(add2) - shape(N * Dy)
    scores = softmax(add2)
    correct_labels = np.where(labels == 1) # tuples of indices

    loss = -np.sum(np.log(scores[correct_labels]))

    if only_forward == True:
        return loss

    # Backward pass for the two layer neural network
    dadd2 = scores
    dadd2[correct_labels] -= 1 # CROSS ENTROPY LOSS

    # add2 = dot2 + b2
    # dadd2 - shape(N * Dy)
    # dot2 - shape(N * Dy)
    # b2 - shape(Dy)
    db2 = np.sum(dadd2, axis=0)
    ddot2 = dadd2

    # dot2 =  h1.dot(W2)
    # ddot2 - shape(N * dy)
    # h1 - shape(N * H)
    # W2 - shape(H * Dy)
    dW2 = h1.T.dot(ddot2)
    dh1 = ddot2.dot(W2.T)

    # h1 = sigmoid(add1)
    # h1 - shape(N * H)
    # add1 - shape(N * H)
    dadd1 = dh1 * sigmoid_grad(h1)

    # add1 = dot1 + b1
    # add1 - shape(N * H)
    # dot1 - shape(N * H)
    # b1 - shape(H)
    db1 = np.sum(dadd1, axis=0)
    ddot1 = dadd1

    # dot1 = X.dot(W1)
    # dot1 - shape(N * H)
    # X - shape(N * Dx)
    # W1 - shape(Dx * H)
    dW1 = X.T.dot(ddot1)
    dX = ddot1.dot(W1.T)

    gradW1 += dW1
    gradb1 += db1
    gradW2 += dW2
    gradb2 += db2



    ### Stack gradients (do not modify)
    grad = np.concatenate((gradW1.flatten(), gradb1.flatten(),
        gradW2.flatten(), gradb2.flatten()))

    return loss, grad

def sanity_check():
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."

    N = 20
    ############################################################################
    # The following dimensions are Dx, H, Dy                                   #
    ############################################################################
    dimensions = [10, 5, 10]
    data = np.random.randn(N, dimensions[0])  # each row will be a datum
    labels = np.zeros((N, dimensions[2]))   # lables are 1 only where it is True
    for i in xrange(N):
        labels[i,random.randint(0,dimensions[2]-1)] = 1

    ############################################################################
    # All the params are packed here                                           #
    # I think this a bad way - You need to initialize using normal distribution#
    # See the number of parameters in the q1.pdf                               #
    ############################################################################
    params = np.random.randn((dimensions[0] + 1) * dimensions[1] + (
        dimensions[1] + 1) * dimensions[2], )

    gradcheck_naive(lambda params: forward_backward_prop(data, labels, params,
        dimensions), params)

def your_sanity_checks():
    """
    Use this space add any additional sanity checks by running:
        python q2_neural.py
    This function will not be called by the autograder, nor will
    your additional tests be graded.
    """
    print "Running your sanity checks..."
    """
    Set up fake data and parameters for the neural network, and test using
    gradcheck.
    """
    print "Running sanity check..."
    pass    
    ### END YOUR CODE

if __name__ == "__main__":
    sanity_check()
    # your_sanity_checks()
