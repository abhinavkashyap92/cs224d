import numpy as np
import random
import sys
from cs224d.data_utils import *

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q3_sgd import load_saved_params

def getSentenceFeature(tokens, wordVectors, sentence):
    """ Obtain the sentence feature for sentiment analysis by averaging its word vectors """
    # Implement computation for the sentence features given a sentence.

    # Inputs:
    # - tokens: a dictionary that maps words to their indices in
    #          the word vector list
    # - wordVectors: word vectors (each row) for all tokens
    # - sentence: a list of words in the sentence of interest

    # Output:
    # - sentVector: feature vector for the sentence

    sentVector = np.zeros((wordVectors.shape[1],))

    ### YOUR CODE HERE
    for word in sentence:
        word_index = tokens[word]
        word_vector = wordVectors[word_index]
        sentVector += word_vector
    ### END YOUR CODE
    sentVector = (1.0/len(sentence)) * sentVector
    return sentVector

def softmaxRegression(features, labels, weights, regularization = 0.0, nopredictions = False):
    """ Softmax Regression """
    # Implement softmax regression with weight regularization.

    # Inputs:
    # - features: feature vectors, each row is a feature vector (N * D)
    # - labels: labels corresponding to the feature vectors (N,)
    # - weights: weights of the regressor (D * C)
    # - regularization: L2 regularization constant

    # Output:
    # - cost: cost of the regressor
    # - grad: gradient of the regressor cost with respect to its
    #        weights
    # - pred: label predictions of the regressor (you might find
    #        np.argmax helpful)


    # calculate the scores
    # scores shape (N, C)
    dot1 = features.dot(weights)
    prob = softmax(dot1)
    # print "prob shape %s" % (prob.shape, )
    # print "weights shape %s" % (weights.shape, )
    # print "features shape %s" % (features.shape, )

    if len(features.shape) > 1:
        N = features.shape[0]
    else:
        N = 1
    # A vectorized implementation of    1/N * sum(cross_entropy(x_i, y_i)) + 1/2*|w|^2
    cost = np.sum(-np.log(prob[range(N), labels])) / N
    cost += 0.5 * regularization * np.sum(weights ** 2)

    ### YOUR CODE HERE: compute the gradients and predictions
    ddot = prob.copy()
    ddot[range(N), labels] -= 1
    ddot /= N

    # dot1 = features.dot(weights)
    # weights shape D* C
    # feature shape N, D
    # dot shape N * C
    dweights = features.T.dot(ddot)
    grad = dweights
    grad += (regularization * weights)

    ### END YOUR CODE
    pred = np.argmax(prob, axis=1)
    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred

def accuracy(y, yhat):
    """ Precision for classifier """
    assert(y.shape == yhat.shape)
    return np.sum(y == yhat) * 100.0 / y.size

def softmax_wrapper(features, labels, weights, regularization = 0.0):
    cost, grad, _ = softmaxRegression(features, labels, weights,
        regularization)
    return cost, grad

def sanity_check():
    """
    Run python q4_softmaxreg.py.
    """
    random.seed(314159)
    np.random.seed(265)

    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    _, wordVectors0, _ = load_saved_params()
    wordVectors = (wordVectors0[:nWords,:] + wordVectors0[nWords:,:])
    dimVectors = wordVectors.shape[1]

    dummy_weights = 0.1 * np.random.randn(dimVectors, 5)
    dummy_features = np.zeros((10, dimVectors))
    dummy_labels = np.zeros((10,), dtype=np.int32)
    for i in xrange(10):
        words, dummy_labels[i] = dataset.getRandomTrainSentence()
        dummy_features[i, :] = getSentenceFeature(tokens, wordVectors, words)
    print "==== Gradient check for softmax regression ===="
    gradcheck_naive(lambda weights: softmaxRegression(dummy_features,
        dummy_labels, weights, 1.0, nopredictions = True), dummy_weights)

    print "\n=== Results ==="
    print softmaxRegression(dummy_features, dummy_labels, dummy_weights, 1.0)

if __name__ == "__main__":
    sanity_check()
