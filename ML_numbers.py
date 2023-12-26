import numpy as np
import pickle
import os.path
import mnist
from mnist import MNIST
import random
import math


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def getRandomVal01():
    return round(random.uniform(0, 1), 2)

def seeNeuralNetworkVals():
    ## see neurons
    # print("inputLayer: ", inputLayer)
    # print()
    print("hiddenLayer1: ", hiddenLayer1)
    print()
    print("hiddenLayer2: ", hiddenLayer2)
    print()
    print("outputLayer: ", outputLayer)
    print()


    ## see weights and biases
    print("weightsInputLayer: ", weightsInputLayer)
    print()
    print("weightsHiddenLayer1: ", weightsHiddenLayer1)
    print()
    print("weightsHiddenLayer2: ", weightsHiddenLayer2)
    print()
    print("biasVector: ", biasVector)
    print()


    ## see other data
    lenH1 = len(hiddenLayer1)
    lenH2 = len(hiddenLayer2)
    lenOut = len(outputLayer)
    lenBias = len(biasVector)

    print("lenH1: ", lenH1)
    print("lenH2: ", lenH2)
    print("lenOut: ", lenOut)
    print("lenBias: ", lenBias)


def normalizedXavierWeightInit(numNodesPrevLayer, numNodesCurrentLayer):
    lower, upper = -(math.sqrt(6.0) / math.sqrt(numNodesPrevLayer + numNodesCurrentLayer)), (math.sqrt(6.0) / math.sqrt(numNodesPrevLayer + numNodesCurrentLayer))
    return round(random.uniform(lower, upper), 2)


mndata = MNIST('samples')
images, labels = mndata.load_training()
numLayersNN = 4 # input, Hidden 1, Hidden 2, Output

index = 0
currentImage = images[index]
currentImageLabel = labels[index]

inputLayerLen = len(currentImage)
hiddenLayer1Len = 16
hiddenLayer2Len = 16
outputLayerLen = 10

# change data values from mnist (0-255) to our (0-1)
for x in range(inputLayerLen):
    currentImage[x] = currentImage[x] / 255


# neurons at each layer
inputLayer = np.array(currentImage)
hiddenLayer1 = np.array([getRandomVal01() for x in range(hiddenLayer1Len)])
hiddenLayer2 = np.array([getRandomVal01() for x in range(hiddenLayer2Len)])
outputLayer = np.array([getRandomVal01() for x in range(outputLayerLen)])

# weights of connections between each layer
weightsInputLayer = np.array([[normalizedXavierWeightInit(inputLayerLen, hiddenLayer1Len) for col in range(inputLayerLen)] for row in range(hiddenLayer1Len)])
weightsHiddenLayer1 = np.array([[normalizedXavierWeightInit(hiddenLayer1Len, hiddenLayer2Len) for col in range(hiddenLayer1Len)] for row in range(hiddenLayer2Len)])
weightsHiddenLayer2 = np.array([[normalizedXavierWeightInit(hiddenLayer2Len, outputLayerLen) for col in range(hiddenLayer2Len)] for row in range(outputLayerLen)])

# doesn't need to be np array itself b/c we will just be using this to hold the np arrays of the weights
weightVector = [weightsInputLayer, weightsHiddenLayer1, weightsHiddenLayer2]

numBiases = hiddenLayer1Len + hiddenLayer2Len + outputLayerLen

# bias values can be initialized to 0 b/c asymmetry breaking is provided by the small random numbers in the weights
biasVector = np.array([0 for x in range(numBiases)])

seeNeuralNetworkVals()


def computeOutput():
    
    # just work on computing one layer
    # start with simpler system to make sure works then do with entire system
    print("compute output")


def main():
    computeOutput()

main()





