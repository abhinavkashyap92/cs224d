import numpy as np
import random
import sys

from q1_softmax import softmax
from q2_gradcheck import gradcheck_naive
from q2_sigmoid import sigmoid, sigmoid_grad

def normalizeRows(x):
    """ Row normalization function """
    # Implement a function that normalizes each row of a matrix to have unit length

    ### YOUR CODE HERE
    axis = x.ndim - 1
    x = x.astype('float64') # Just to make sure - making it float 64
    norm_x = np.linalg.norm(x, axis=axis, keepdims=True)
    x = x / norm_x
    ### END YOUR CODE

    return x

def test_normalize_rows():
    print "Testing normalizeRows..."
    x = normalizeRows(np.array([[3.0,4.0],[1, 2]]))
    # the result should be [[0.6, 0.8], [0.4472, 0.8944]]
    print x
    assert (x.all() == np.array([[0.6, 0.8], [0.4472, 0.8944]]).all())
    print ""


def softmaxCostAndGradient(predicted, target, outputVectors, dataset):
    """ Softmax cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, assuming the softmax prediction function and cross
    # entropy loss.

    # Inputs:
    # - predicted: numpy ndarray, predicted word vector (\hat{v} in
    #   the written component or \hat{r} in an earlier version) |1| * D
    # - target: integer, the index of the target word
    # - outputVectors: "output" vectors (as rows) for all tokens
    #                   dimension - (|V| * D)
    # - dataset: needed for negative sampling, unused here.

    # Outputs:
    # - cost: cross entropy cost for the softmax word prediction
    # - gradPred: the gradient with respect to the predicted word
    #        vector
    # - grad: the gradient with respect to all the other word
    #        vectors

    D = predicted.shape[0]
    N, D = outputVectors.shape # N is the size of the vocabulary
    z = outputVectors.dot(predicted) # 1 * N
    scores = softmax(z) # N * 1
    loss = -np.log(scores[target])
    dz = scores
    dz[target] -= 1

    # z = outputVectors.dot(predicted)
    # predicted - shape(D,)
    # outputVectors - shape(N, D)
    # dz - shape(N,)
    gradPred = outputVectors.T.dot(dz)
    grad = np.outer(dz, predicted)
    return loss, gradPred, grad

def negSamplingCostAndGradient(predicted, target, outputVectors, dataset,
    K=10):
    """ Negative sampling cost function for word2vec models """

    # Implement the cost and gradients for one predicted word vector
    # and one target word vector as a building block for word2vec
    # models, using the negative sampling technique. K is the sample
    # size. You might want to use dataset.sampleTokenIdx() to sample
    # a random word index.
    #
    # Note: See test_word2vec below for dataset's initialization.
    #
    # Input/Output Specifications: same as softmaxCostAndGradient
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!
    # Input:
    #   predicted: (D,)
    #   target: index of the predicted word
    #   outputVectors: (N * D)
    #   dataset is an object that contains various methods for data manipulation

    ### YOUR CODE HERE
    klist = []
    for k in range(K):
        randomId = dataset.sampleTokenIdx()
        while randomId == target:
            randomId = dataset.sampleTokenIdx()
        klist.append(randomId)

    # Find the negative sampling loss now
    u0 = outputVectors[target]
    dot1 = u0.dot(predicted)
    first_term = np.log(sigmoid(dot1))

    second_term = 0.0

    # This can be vectorised for a faster implementation
    # This is just for better understanding
    for k in range(K):
        index = klist[k]
        uk = outputVectors[index]
        vc = predicted
        dot2 = uk.dot(vc)
        second_term += np.log(sigmoid(-dot2))

    cost = -(first_term + second_term)

    # Lets do the derivatives now
    gradPred = np.zeros(predicted.shape)
    grad = np.zeros(outputVectors.shape)

    # Gradient with respect to vc

    temp = sigmoid(u0.dot(predicted)) - 1
    gradPred += temp * outputVectors[target]
    grad[target] += temp * predicted

    for k in xrange(K):
        index = klist[k]
        uk = outputVectors[index]
        vc = predicted
        temp = sigmoid(-(uk.dot(vc))) - 1
        gradPred += -temp *  uk
        grad[index] += - temp * predicted






    ### END YOUR CODE

    return cost, gradPred, grad

def skipgram(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ Skip-gram model in word2vec """

    # Implement the skip-gram model in this function.

    # Inputs:
    # - currrentWord: a string of the current center word
    # - C: integer, context size
    # - contextWords: list of no more than 2*C strings, the context words
    # - tokens: a dictionary that maps words to their indices in
    #      the word vector list
    # - inputVectors: "input" word vectors (as rows) for all tokens
    # - outputVectors: "output" word vectors (as rows) for all tokens
    # - word2vecCostAndGradient: the cost and gradient function for
    #      a prediction vector given the target word vectors,
    #      could be one of the two cost functions you
    #      implemented above

    # Outputs:
    # - cost: the cost function value for the skip-gram model
    # - grad in and grad out: the gradient with respect to the word vectors
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    currentWordIndex = tokens[currentWord]
    v_hat = inputVectors[currentWordIndex]
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)
    cost = 0.0
    ### YOUR CODE HERE
    for contextWord in contextWords:
        # You need to predict the context word using the center word
        target = tokens[contextWord]
        # The predicted vector is vc
        # gradPred is the gradient with respect to vc - see the notes for
        # the derivation
        # grad is the gradient with repsect to uo
        c, gradPred, grad = word2vecCostAndGradient(v_hat, target,
                                                      outputVectors, dataset)
        cost += c
        gradIn[currentWordIndex] += gradPred
        gradOut += grad
    ### END YOUR CODE

    return cost, gradIn, gradOut

def cbow(currentWord, C, contextWords, tokens, inputVectors, outputVectors,
    dataset, word2vecCostAndGradient = softmaxCostAndGradient):
    """ CBOW model in word2vec """

    # Implement the continuous bag-of-words model in this function.
    # Input/Output specifications: same as the skip-gram model
    # We will not provide starter code for this function, but feel
    # free to reference the code you previously wrote for this
    # assignment!

    #################################################################
    # IMPLEMENTING CBOW IS EXTRA CREDIT, DERIVATIONS IN THE WRIITEN #
    # ASSIGNMENT ARE NOT!                                           #
    #################################################################

    cost = 0
    gradIn = np.zeros(inputVectors.shape)
    gradOut = np.zeros(outputVectors.shape)

    ### YOUR CODE HERE
    raise NotImplementedError
    ### END YOUR CODE

    return cost, gradIn, gradOut

#############################################
# Testing functions below. DO NOT MODIFY!   #
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C, word2vecCostAndGradient = softmaxCostAndGradient):
    ############################################################################
    # Input:
        # word2vecModel: it is either skipgram or CBOW that is constructed
        # tokens: The token to index mappig in a list of tuples
        # wordVectors - This is the vector that has to be learnt :p
        # dataset: This is the dataset object that may contain utility function
        # C - The size of the context
        # word2vecCostAndGradient - The function object that calculates
        # The cost of prediction and the gradients of the loss function
    # The input vectors is the V matrix of dim(|V|*D) where D=dim(embedding space)
    # The output vectors is the U matrix, of dim(|V| * D)
    # Both of them are contained in the wordVectors
    ############################################################################

    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:N/2,:] # The first half rows are input vectors
    outputVectors = wordVectors[N/2:,:] # The second half row are the output vectors
    for i in xrange(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(centerword, C1, context, tokens, inputVectors, outputVectors, dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:N/2, :] += gin / batchsize / denom
        grad[N/2:, :] += gout / batchsize / denom

    return cost, grad

def test_word2vec():
    # Interface to the dataset for negative sampling
    dataset = type('dummy', (), {})()
    def dummySampleTokenIdx():
        return random.randint(0, 4)

    def getRandomContext(C):
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] \
           for i in xrange(2*C)]
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalizeRows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])
    print "==== Gradient check for skip-gram ===="
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    # print "\n==== Gradient check for CBOW      ===="
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5), dummy_vectors)
    # gradcheck_naive(lambda vec: word2vec_sgd_wrapper(cbow, dummy_tokens, vec, dataset, 5, negSamplingCostAndGradient), dummy_vectors)
    #
    print "\n=== Results ==="
    # print skipgram("c", 3, ["a", "b", "e", "d", "b", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    # print skipgram("c", 1, ["a", "b"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)
    # print cbow("a", 2, ["a", "b", "c", "a"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset)
    # print cbow("a", 2, ["a", "b", "a", "c"], dummy_tokens, dummy_vectors[:5,:], dummy_vectors[5:,:], dataset, negSamplingCostAndGradient)

if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
