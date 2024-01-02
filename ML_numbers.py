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

implement saving feature for dC/dAL and dC/dZL for current layer
make sure to use it correctly during back prop by grabbing correct value when looking at each neuron
hand check values
plug in images and all other real parts - make sure working / no errors
start training algorithm and test if works

"""


def sigmoid(x):
    return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):
    return np.exp(-x) / math.pow((1 + np.exp(-x)), 2)

def getRandomVal01():
    return round(random.uniform(0, 1), 2)

def seeNeuralNetworkVals():

    print("Weights Output Layer:", end="")
    print(layers[3].weights)

    print("gradient_weight Output Layer: ", end="")
    print(layers[3].gradient_weight)

    print("gradient_bias Output Layer: ", end="")
    print(layers[3].gradient_bias)

    print("Z Output Layer: ", end="")
    print(layers[3].z)

    print("Neurons Output Layer: ")
    print(layers[3].neurons)

    print("------------------------------------------")
    print("------------------------------------------")

    print("Weights Hidden Layer 2: ", end="")
    print(layers[2].weights)

    print("gradient_weight Hidden Layer 2: ", end="")
    print(layers[2].gradient_weight)

    print("gradient_bias Hidden Layer 2: ", end="")
    print(layers[2].gradient_bias)

    print("Z Hidden Layer 2: ", end="")
    print(layers[2].z)

    print("Neurons Hidden Layer 2: ")
    print(layers[2].neurons)

    print("------------------------------------------")
    print("------------------------------------------")

    print("Weights Hidden Layer 1: ", end="")
    print(layers[1].weights)

    print("gradient_weight Hidden Layer 1: ", end="")
    print(layers[1].gradient_weight)

    print("gradient_bias Hidden Layer 1: ", end="")
    print(layers[1].gradient_bias)

    print("Z Hidden Layer 1: ", end="")
    print(layers[1].z)

    print("Neurons Hidden Layer 1: ")
    print(layers[1].neurons)

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
            self.gradient_weight = None
            self.gradient_bias = None
        else:
            # TODO: make sure below line of code is necessary - might not be b/c already have weights being initialized by xavier
            self.weights = np.array([[normalizedXavierWeightInit(previousLayer.numNeurons, numNeurons) for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            self.gradient_weight = np.array([[0.0 for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            self.gradient_bias = np.array([0.0] * numNeurons)

        self.biases = np.array([0.0 for x in range(numNeurons)])

        

    def forwardPropigation(self):
        # note: self.z is a vector not a single value. It holds the Z values for each neuron in the layer
        self.z = np.matmul(self.weights, self.previousLayer.neurons) + self.biases
        self.neurons = sigmoid(self.z)



    def backPropigation(self):
        
        # train same model vals with a batch of training examples, then take sum of all those to get actual cost function

        if not self.nextLayer:

            # calculate gradient for weight
            self.dC_dAL = np.array([0.0] * self.numNeurons) 

            for currentLayerNeuronIndex in range(len(self.neurons)):
                
                #^^ Note: number of weights attacked to each neuron in current layer = number of neurons in previous layer
                for prevLayerWeightIndex in range(len(self.previousLayer.neurons)):

                    # sumCost = 0.0
                    sumCost = float((2 * (self.neurons[currentLayerNeuronIndex] - expected[currentLayerNeuronIndex]))) # changed to = from += 
                    self.dC_dAL[currentLayerNeuronIndex] = sumCost
                    prevLayerNeuronOutput = self.previousLayer.neurons[prevLayerWeightIndex]
                    dSigmoid_ZCurrent = derivative_sigmoid(self.z[currentLayerNeuronIndex])
                    
                    self.gradient_weight[currentLayerNeuronIndex][prevLayerWeightIndex] = prevLayerNeuronOutput * dSigmoid_ZCurrent * sumCost

                
                # calculate gradient for bias
                dSigmoid_ZCurrent_for_bias = derivative_sigmoid(self.z[currentLayerNeuronIndex])
                sumCost_bias = float(2 * (self.neurons[currentLayerNeuronIndex] - expected[currentLayerNeuronIndex]))
                self.gradient_bias[currentLayerNeuronIndex] = dSigmoid_ZCurrent_for_bias * sumCost_bias


        else:
            # calculate gradient for weight
            self.dC_dAL = np.array([0.0] * self.numNeurons)

            for currentLayerNeuronIndex in range(len(self.neurons)):
                for prevLayerWeightIndex in range(len(self.previousLayer.neurons)): # this is how many weights per neuron in this layer

                    prevLayerNeuronOutput = self.previousLayer.neurons[prevLayerWeightIndex]
                    dSigmoid_ZCurrent = derivative_sigmoid(self.z[currentLayerNeuronIndex])

                    sumCost = 0.0
                    # sum the influence of current neuron's activation on cost: dC_dAL-1 (hidden layer)
                    for nextLayerNeuronIndex in range(len(self.nextLayer.neurons)):
                        nextLayerWeightForCurrentNeuron = self.nextLayer.weights[nextLayerNeuronIndex][currentLayerNeuronIndex]
                        dSigmoid_ZNext = derivative_sigmoid(self.nextLayer.z[nextLayerNeuronIndex])
                        sumCost += nextLayerWeightForCurrentNeuron * dSigmoid_ZNext * self.nextLayer.dC_dAL[nextLayerNeuronIndex]     # was self.nextLayer.dC_dAL
                        
                    self.dC_dAL[currentLayerNeuronIndex] = sumCost
                    
                    self.gradient_weight[currentLayerNeuronIndex][prevLayerWeightIndex] = prevLayerNeuronOutput * dSigmoid_ZCurrent * sumCost # this sumCost is fine b/c it's for this neuron not next

                # find bias for each neuron in hidden layer
                dSigmoid_ZCurrent_for_bias = derivative_sigmoid(self.z[currentLayerNeuronIndex])
                costCurrentNeuron = self.dC_dAL[currentLayerNeuronIndex]
                self.gradient_bias[currentLayerNeuronIndex] = dSigmoid_ZCurrent_for_bias * costCurrentNeuron

                

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

# do this after b/c cannot do in initialization b/c not delcared
hiddenLayer1.nextLayer = hiddenLayer2
hiddenLayer2.nextLayer = outputLayer
outputLayer.nextLayer = None

# for testing

layers = [inputLayer, hiddenLayer1, hiddenLayer2, outputLayer]

inputLayer.weights = None
#^^ DO NOT DELETE - for actual 
# hiddenLayer1.weights = np.array([[normalizedXavierWeightInit(inputLayer.numNeurons, hiddenLayer1.numNeurons) for col in range(inputLayer.numNeurons)] for row in range(hiddenLayer1.numNeurons)])
# hiddenLayer2.weights = np.array([[normalizedXavierWeightInit(hiddenLayer1.numNeurons, hiddenLayer2.numNeurons) for col in range(hiddenLayer1.numNeurons)] for row in range(hiddenLayer2.numNeurons)])
# outputLayer.weights = np.array([[normalizedXavierWeightInit(hiddenLayer2.numNeurons, outputLayer.numNeurons) for col in range(hiddenLayer2.numNeurons)] for row in range(outputLayer.numNeurons)])

# for test - manually forcing values / environment that would be seen after forward propigation
# hiddenLayer1.neurons = np.array([0.35663485, 0.30767744])
# hiddenLayer2.neurons = np.array([[0.52180225, 0.41313151]])
# outputLayer.neurons = np.array([[0.61935516, 0.30040983]])

hiddenLayer1.weights = np.array([[ 0.36, -0.57, -0.59],[-0.78, -0.58, -0.41]])
hiddenLayer2.weights = np.array([[ -0.48,  0.84],[-0.32, -0.77]])
outputLayer.weights = np.array([[ 0.64,  0.37],[-1.05, -0.72]])

# for testing - change expected to match label for image
expected = [0] * len(outputLayer.neurons)
expected[0] = 1  # for test: expected will be first neuron to be 1 and all others to be 0

# TODO: Compute new layer value as method within layers. then have this function just call that for all layers and print output to check
def forwardPropigation():
    for layer in layers[1:]:
        layer.forwardPropigation()

def backPropigation():

    for x in range(len(layers) - 1, 0, -1):
        layers[x].backPropigation()


    
def main():
   
    forwardPropigation()
    backPropigation()
    seeNeuralNetworkVals()


    
main()





