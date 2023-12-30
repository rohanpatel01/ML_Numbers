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
TODO: Back prop
TODO: fix: each neuron contains a z not a layer so self.z doesn't make sense, would need to make an array similar to "neurons"


"""


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):
    return np.exp(-x) / math.pow((1 + np.exp(-x)), 2)

def getRandomVal01():
    return round(random.uniform(0, 1), 2)

def seeNeuralNetworkVals():

    print("Weights Hidden Layer 1: ", end="")
    print(layers[1].weights)

    print("Neurons Hidden Layer 1: ")
    print(layers[1].neurons)

    print("------------------------------------------")
    print("------------------------------------------")

    print("Weights Hidden Layer 2: ", end="")
    print(layers[2].weights)

    print("Neurons Hidden Layer 2: ")
    print(layers[2].neurons)

    print("------------------------------------------")
    print("------------------------------------------")
    
    print("Weights Output Layer 2:", end="")
    print(layers[3].weights)

    print("Neurons Output Layer 1: ")
    print(layers[3].neurons)

    print("------------------------------------------")
    print("------------------------------------------")

    print("Expected Layer: ", end="")
    print(expected)

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

    # z = None

    def __init__(self, numNeurons, previousLayer: Type['Layer'] = None, nextLayer: Type['Layer'] = None, neurons = None) -> None:
        self.numNeurons = numNeurons

        self.previousLayer = previousLayer
        self.nextLayer = nextLayer

        # For other layers - input layer has default neurons, all other layers default to value of 0 b/c will be calculated
        if neurons is not None:
            self.neurons = neurons
        else:
            self.neurons = np.array([0.0 for x in range(numNeurons)])

        # to handle input layer not having weights b/c has no previous layer
        if not previousLayer:
            self.weights = None
            self.gradient = None
        else:
            self.weights = np.array([[normalizedXavierWeightInit(previousLayer.numNeurons, numNeurons) for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            self.gradient = np.array([[0.0 for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            

        self.biases = np.array([0.0 for x in range(numNeurons)])

        

    def forwardPropigation(self):
        # note: self.z is a vector not a single value. It holds the Z values for each neuron in the layer
        self.z = np.matmul(self.weights, self.previousLayer.neurons) + self.biases
        self.neurons = sigmoid(self.z)

    def backPropigation(self):
        #^ note: each element in self.gradient represents the dC_dW (how much each weight should be changed)
        
        # train same model vals with a batch of training examples, then take sum of all those to get actual cost function

        # is ouput layer
        # todo: make sure to save cost so we can pass down to next layer
        if not self.nextLayer:
            sumCost = 0
            for x in range(len(self.neurons)):
                sumCost += math.pow(self.neurons[x] - expected[x], 2)


            # for every neuron in current layer find how much to change each weight
            #^^ don't delete b/c for actual full run
            # for currentLayerNeuronIndex in range(len(self.neurons)):
                
            #     #^^ Note: number of weights attacked to each neuron in current layer = number of neurons in previous layer
            #     for prevLayerWeightIndex in range(len(self.previousLayer.neurons)):

            #         prevLayerNeuronOutput = self.previousLayer.neurons[prevLayerWeightIndex]
            #         dSigmoid_ZL = derivative_sigmoid(self.z[currentLayerNeuronIndex])
            #         self.gradient[currentLayerNeuronIndex][prevLayerWeightIndex] = prevLayerNeuronOutput * dSigmoid_ZL * sumCostd

            # ^^^^ Testing purposes just do first neuron in output layer
            for currentLayerNeuronIndex in range(1):
                
                for prevLayerWeightIndex in range(len(self.previousLayer.neurons)):

                    prevLayerNeuronOutput = self.previousLayer.neurons[prevLayerWeightIndex]
                    dSigmoid_ZL = derivative_sigmoid(self.z[currentLayerNeuronIndex])
                    self.gradient[currentLayerNeuronIndex][prevLayerWeightIndex] = prevLayerNeuronOutput * dSigmoid_ZL * sumCost

        pass

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

# for testing - change expected to match label for image
expected = [0] * len(outputLayer.neurons)
expected[0] = 1  # for test: expected will be first neuron to be 1 and all others to be 0

# TODO: Compute new layer value as method within layers. then have this function just call that for all layers and print output to check
def forwardPropigation():
    for layer in layers[1:]:
        layer.forwardPropigation()

def backPropigation():
    outputLayer = layers[-1]
    print("output gradient: ", outputLayer.gradient)
    outputLayer.backPropigation()
    print("output gradient after: ", outputLayer.gradient)

    
def main():
   
    forwardPropigation()
    print("before")
    seeNeuralNetworkVals()

    print("----------------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------------")

    print("Starting Back prop")
    backPropigation()

    print("----------------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------------")
    print("----------------------------------------------------------------------------------------------------------------")

    print("after")
    seeNeuralNetworkVals()

    

    

main()





