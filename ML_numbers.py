import numpy as np
import pickle
import os.path
import mnist
from mnist import MNIST
import random

def initStructure():
    mndata = MNIST('samples')
    images, labels = mndata.load_training()

    index = 0
    currentImage = images[index]

    inputLayerLen = len(currentImage)
    hiddenLayer1Len = 16
    hiddenLayer2Len = 16
    outputLayerLen = 10


    # change data values from mnist (0-255) to our (0-1)
    for x in range(inputLayerLen):
        currentImage[x] = currentImage[x] / 255

    # holds neuron values for each layer
    inputLayer = currentImage
    hiddenLayer1 = [0] * hiddenLayer1Len
    hiddenLayer2 = [0] * hiddenLayer2Len
    outputLayer = [0] * outputLayerLen

    numWeights = (len(inputLayer) * hiddenLayer1Len) + (hiddenLayer1Len * hiddenLayer2Len) + (hiddenLayer2Len * outputLayerLen)
    numBiases = hiddenLayer1Len + hiddenLayer2Len + outputLayerLen

    # holds weights and bias values 
    weightVector = [0] * numWeights
    biasVector = [0] * numBiases

    # initialize weights and biases with random values for training 
    for i in range(numWeights):
        weightVector[i] = round(random.uniform(0, 1), 2)

    for i in range(numBiases):
        biasVector[i] = round(random.uniform(0, 1), 2)



def main():
    initStructure()






main()


