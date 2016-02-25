#!/usr/bin/env python3

# Based directly on the Youtube tutorials by Welch Labs
# Original source code @ github.com/stephencwelch/Neural-Networks-Demystified
# A simple artificial neural network
# by Marshall Ehlinger
# 2-22-16

import numpy as np

class Neural_Network(object):
    def __init__(self):
        # Define Hyper Parameters (hyperParameters do not change)
        self.inputLayerSize = 2
        self.outputLayerSize = 1
        self.hiddenLayerSize = 3

        #Weights / Parameters (these change as the ann learns)
        self.W1 = np.random.randn(self.inputLayerSize, \
                self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize, \
                self.outputLayerSize)
    
    def forward(self, X):
        # Propagate inputs thru network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):
        #Apply sigmoid activation function to scalar, vector, or matric
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        # Derivative/gradient of sigmoid function
        return np.exp(-z)/((1+np.exp(-z))**2)

    def costFunction(self, X, y):
        # Computer cost for given X,y use weights already stored in ANN object
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        # Compute derivative with respect to weight layer 1, W1, and weight layer 2, W2
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)
        
        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    # Helper functions

    def getParams(self):
        # Return W1 and W2 as a single vector
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        # Set W1 and W2 as a single vector
        W1_startIndex = 0
        W1_endIndex = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_startIndex:W1_endIndex], \
                (self.inputLayerSize, self.hiddenLayerSize))
        W2_endIndex = W1_endIndex + self.hiddenLayerSize * self.outputLayerSize
        self.W2 = np.reshape(params[W1_endIndex:W2_endIndex], \
                (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))
    
def computeNumericalGradient(N, X, y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-4

    for p in range(len(paramsInitial)):
        # Set perturbation vector
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        loss2 = N.costFunction(X, y)

        N.setParams(paramsInitial - perturb)
        loss1 = N.costFunction(X, y)

        # Compute numerical Gradient
        numgrad[p] = (loss2-loss1) / (2*e)

        # Return the value we changed back to zero
        perturb[p] = 0

    # Return params to original value
    N.setParams(paramsInitial)

    return numgrad

if __name__ == "__main__":
    # X is input data, X1 as hours slept, and X2 hours studied
    # Y represents output data, test score 0-100
    X = np.array(([3,5], [5,1], [10,2]), dtype=float)
    y = np.array(([75], [82], [93]), dtype=float)   
    X = X/np.amax(X, axis=0)
    y = y/100 # Max output value is 100
    
    NN = Neural_Network()
    # yHat is our prediction(s)
    yHat = NN.forward(X)

    print(yHat)
    print(y)
    print("----------------")

    numgrad = computeNumericalGradient(NN, X, y)
    grad = NN.computeGradients(X,y)
    print(numgrad)
    print(grad)
    print(np.linalg.norm(grad-numgrad)/np.linalg.norm(grad+numgrad))
    print('--------')

