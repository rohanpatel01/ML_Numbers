import numpy as np
import pickle
import os.path
import mnist
from mnist import MNIST
from typing import Type
import random
import math
import copy

"""
TODO: Forward prop - do with actual image and see output - won't know issues until after

"""


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def getRandomVal01():
    return round(random.uniform(0, 1), 2)

def seeNeuralNetworkVals():

    for layer in layers:
        print("Neurons")
        print("Num Neurons: ", layer.numNeurons)
        print(layer.neurons)
        print()

        print("Weights")
        print(layer.weights)
        print()

        print("Biases")
        print(layer.biases)
        print()

        print("------------------------------------------------------------")

def seeLastLayerOutput():
    print("Last Layer: ", layers[-1].neurons)

def normalizedXavierWeightInit(numNodesPrevLayer, numNodesCurrentLayer):
    lower, upper = -(math.sqrt(6.0) / math.sqrt(numNodesPrevLayer + numNodesCurrentLayer)), (math.sqrt(6.0) / math.sqrt(numNodesPrevLayer + numNodesCurrentLayer))
    return round(random.uniform(lower, upper), 2)


mndata = MNIST('samples')
images, labels = mndata.load_training()
numLayersNN = 4 # input, Hidden 1, Hidden 2, Output

index = 0
# ^^ Don't delete - for actual image
# currentImage = images[index]
# currentImageLabel = labels[index]


class Layer:

    def __init__(self, previousLayer: Type['Layer'], numNeurons, neurons = None) -> None:
        self.previousLayer = previousLayer
        self.numNeurons = numNeurons

        # done to allow for input layer to manually write neuron values
        if neurons is not None:
            self.neurons = neurons
        else:
            self.neurons = np.array([0 for x in range(numNeurons)])

        if not previousLayer:
            self.weights = None
        else:
            self.weights = np.array([[normalizedXavierWeightInit(previousLayer.numNeurons, numNeurons) for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
        
        self.biases = np.array([0 for x in range(numNeurons)])

    def forwardPropigation(self):
        self.neurons = sigmoid(np.matmul(self.weights, self.previousLayer.neurons) + self.biases)

inputLayer = Layer(previousLayer=None, numNeurons=3, neurons=np.array([0.3, 0.5, 0.7]))

inputLayerLen = inputLayer.numNeurons        # was len(currentImage)
hiddenLayer1Len = 2      # was 16
hiddenLayer2Len = 2    # was 16
outputLayerLen = 2     # was 10

# ^^ Don't delete - for actual image
# change data values from mnist (0-255) to our (0-1)
# for x in range(inputLayerLen):
#     currentImage[x] = currentImage[x] / 255


# neurons at each layer - init vals remain 0 b/c their av is calculated not set to random
# ^^ Don't delete - for actual image
# inputLayer = np.array(currentImage)
hiddenLayer1 = Layer(previousLayer=inputLayer, numNeurons=hiddenLayer1Len)
hiddenLayer2 = Layer(previousLayer=hiddenLayer1, numNeurons=hiddenLayer2Len)
outputLayer = Layer(previousLayer=hiddenLayer2, numNeurons=outputLayerLen)

layers = [inputLayer, hiddenLayer1, hiddenLayer2, outputLayer]

inputLayer.weights = None
hiddenLayer1.weights = np.array([[normalizedXavierWeightInit(inputLayer.numNeurons, hiddenLayer1.numNeurons) for col in range(inputLayer.numNeurons)] for row in range(hiddenLayer1.numNeurons)])
hiddenLayer2.weights = np.array([[normalizedXavierWeightInit(hiddenLayer1.numNeurons, hiddenLayer2.numNeurons) for col in range(hiddenLayer1.numNeurons)] for row in range(hiddenLayer2.numNeurons)])
outputLayer.weights = np.array([[normalizedXavierWeightInit(hiddenLayer2.numNeurons, outputLayer.numNeurons) for col in range(hiddenLayer2.numNeurons)] for row in range(outputLayer.numNeurons)])


# TODO: Compute new layer value as method within layers. then have this function just call that for all layers and print output to check
def forwardPropigation():
    for layer in layers[1:]:
        layer.forwardPropigation()

    
def main():
    forwardPropigation()


main()





