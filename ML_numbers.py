import numpy as np
import mnist
from mnist import MNIST
from typing import Type
import random
import math
import matplotlib.pyplot as plt

"""

plug in images and all other real parts - make sure working / no errors
start training algorithm and test if works

"""

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def derivative_sigmoid(x):
    return np.exp(-x) / math.pow((1 + np.exp(-x)), 2)

def normalizeInput(input):
    minInput = min(input)
    maxInput = max(input)

    difference = maxInput - minInput

    for element in input:
        element = (element - minInput) / difference


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

def xaviewWeightInit(numNodesPrevLayer):
    lower, upper = -(1 / math.sqrt(numNodesPrevLayer)), (1 / math.sqrt(numNodesPrevLayer))
    return round(random.uniform(lower, upper), 2)

mndata = MNIST('samples')
images, labels = mndata.load_training()

learning_rate = 0.2            # was 0.002125    - 0.1 in video


index = 0
# ^^ Don't delete - for actual image
currentImage = images[index]
currentImageLabel = labels[index]
numTrainingImages = len(images)


class Layer:

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
        if previousLayer is None:
            self.weights = None
            self.gradient_weight = None
            self.gradient_bias = None
        else:

            self.weights = np.array([[normalizedXavierWeightInit(previousLayer.numNeurons, numNeurons) for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            # self.weights = np.array([[xaviewWeightInit(previousLayer.numNeurons) for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            
            self.gradient_weight = np.array([[0.0 for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            self.gradient_bias = np.array([0.0] * numNeurons)

        self.biases = np.array([0.0 for x in range(numNeurons)])

        

    def forwardPropigation(self):
        # note: self.z is a vector not a single value. It holds the Z values for each neuron in the layer
        self.z = np.matmul(self.weights, self.previousLayer.neurons) + self.biases


        # check it output layer and don't apply sigmoid

        if not self.nextLayer:
            self.neurons = self.z
        else:
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
                    
                    self.gradient_weight[currentLayerNeuronIndex][prevLayerWeightIndex] += (prevLayerNeuronOutput * dSigmoid_ZCurrent * sumCost)    # was learning_rate *

                
                # calculate gradient for bias
                dSigmoid_ZCurrent_for_bias = derivative_sigmoid(self.z[currentLayerNeuronIndex])
                sumCost_bias = float(2 * (self.neurons[currentLayerNeuronIndex] - expected[currentLayerNeuronIndex]))
                self.gradient_bias[currentLayerNeuronIndex] += (dSigmoid_ZCurrent_for_bias * sumCost_bias)                                          # was learning_rate * 
                # print("Current bias: ", self.gradient_bias[currentLayerNeuronIndex])

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
                    
                    self.gradient_weight[currentLayerNeuronIndex][prevLayerWeightIndex] +=  (prevLayerNeuronOutput * dSigmoid_ZCurrent * sumCost) # was learning_rate *

                # find bias for each neuron in hidden layer
                dSigmoid_ZCurrent_for_bias = derivative_sigmoid(self.z[currentLayerNeuronIndex])
                costCurrentNeuron = self.dC_dAL[currentLayerNeuronIndex]
                self.gradient_bias[currentLayerNeuronIndex] += (dSigmoid_ZCurrent_for_bias * costCurrentNeuron)             # was learning_rate *

                
#^^ For testing
# inputLayer = Layer(previousLayer=None, numNeurons=3, neurons=np.array([0.3, 0.5, 0.7]))


inputLayerLen = len(currentImage)       # was inputLayer.numNeurons 
hiddenLayer1Len = 16      # was 16
hiddenLayer2Len = 16    # was 16
outputLayerLen = 10     # was 10



# ^^ Don't delete - for actual image
# change data values from mnist (0-255) to our (0-1)
for x in range(inputLayerLen):
    currentImage[x] = currentImage[x] / 255


# neurons at each layer - init vals remain 0 b/c their av is calculated not set to random
# ^^ Don't delete - for actual image
# inputLayer = np.array(currentImage, previousLayer= None)     #^^ make sure this is correct

inputLayer = Layer(numNeurons=len(currentImage), neurons=currentImage)
hiddenLayer1 = Layer(previousLayer=inputLayer, numNeurons=hiddenLayer1Len)
hiddenLayer2 = Layer(previousLayer=hiddenLayer1, numNeurons=hiddenLayer2Len)
outputLayer = Layer(previousLayer=hiddenLayer2, numNeurons=outputLayerLen)

# do this after b/c cannot do in initialization b/c not delcared
inputLayer.nextLayer = hiddenLayer1
hiddenLayer1.nextLayer = hiddenLayer2
hiddenLayer2.nextLayer = outputLayer
outputLayer.nextLayer = None

layers = [inputLayer, hiddenLayer1, hiddenLayer2, outputLayer]

#^^ DO NOT DELETE - for actual 
inputLayer.weights = None
hiddenLayer1.weights = np.array([[normalizedXavierWeightInit(inputLayer.numNeurons, hiddenLayer1.numNeurons) for col in range(inputLayer.numNeurons)] for row in range(hiddenLayer1.numNeurons)])
hiddenLayer2.weights = np.array([[normalizedXavierWeightInit(hiddenLayer1.numNeurons, hiddenLayer2.numNeurons) for col in range(hiddenLayer1.numNeurons)] for row in range(hiddenLayer2.numNeurons)])
outputLayer.weights = np.array([[normalizedXavierWeightInit(hiddenLayer2.numNeurons, outputLayer.numNeurons) for col in range(hiddenLayer2.numNeurons)] for row in range(outputLayer.numNeurons)])

# hiddenLayer1.weights = np.array([[xaviewWeightInit(inputLayer.numNeurons) for col in range(inputLayer.numNeurons)] for row in range(hiddenLayer1.numNeurons)])
# hiddenLayer2.weights = np.array([[xaviewWeightInit(hiddenLayer1.numNeurons) for col in range(hiddenLayer1.numNeurons)] for row in range(hiddenLayer2.numNeurons)])
# outputLayer.weights = np.array([[xaviewWeightInit(hiddenLayer2.numNeurons) for col in range(hiddenLayer2.numNeurons)] for row in range(outputLayer.numNeurons)])

# for test - manually forcing values / environment that would be seen after forward propigation
# hiddenLayer1.neurons = np.array([0.35663485, 0.30767744])
# hiddenLayer2.neurons = np.array([[0.52180225, 0.41313151]])
# outputLayer.neurons = np.array([[0.61935516, 0.30040983]])

#^ For testing
# hiddenLayer1.weights = np.array([[ 0.36, -0.57, -0.59],[-0.78, -0.58, -0.41]])
# hiddenLayer2.weights = np.array([[ -0.48,  0.84],[-0.32, -0.77]])
# outputLayer.weights = np.array([[ 0.64,  0.37],[-1.05, -0.72]])

#^ for testing
# expected = [0.0] * len(outputLayer.neurons)
# expected[0] = 1  # for test: expected will be first neuron to be 1 and all others to be 0

expected = [0.0] * len(outputLayer.neurons)
expected[currentImageLabel] = 1.0


def forwardPropigation():
    for layer in layers[1:]:
        layer.forwardPropigation()

def backPropigation():

    for x in range(len(layers) - 1, 0, -1):
        layers[x].backPropigation()


    
def main():
    
    # TODO: learning rate
    

    global index
    global expected
    global layers

    numCorrect = 0


    batchImageCounter = 0
    batchSize = 100.0            # was 100
    numBatchesToProcess = 5     # was 5

    x = np.array([x] for x in range(0))
    plt.xlim(0, batchSize * numBatchesToProcess)
    plt.ylim(-5, 5)
    plt.grid()
    
    # testOneImageIndex = 0

    while ( (index < numTrainingImages) and (index < (numBatchesToProcess * batchSize))   ):      # and (testOneImageIndex < (numBatchesToProcess * batchSize) )
        
        # process image with back and forward propigation
        currentImage = images[index]
        currentImageLabel = labels[index]

        expected = [0.0] * len(outputLayer.neurons)
        expected[currentImageLabel] = 1.0

        # turn input image from 0-255 into 0-1
        for x in range(inputLayerLen):
            currentImage[x] = currentImage[x] / 255

        # normalizeInput(currentImage) # bad performance with normalizing the input

        # print(currentImage)

        inputLayer.neurons = currentImage
        print("Current Image Index: ", index)
        # print("Test Image Index: ", testOneImageIndex)


        print("Current Image label: ", currentImageLabel)
        # print("output weight 1 gradient: ", layers[-1].gradient_weight[0][0])
        # print("output bias 2 gradient: ", layers[-2].biases[0])

        print("--------------------------------------------")

        forwardPropigation()


        # find accuracy

        prediction = list((layers[-1].neurons)).index(max(layers[-1].neurons))

        # print("output: ", layers[-1].neurons)

        # print("prediction: ", list((layers[-1].neurons)).index(max(layers[-1].neurons)))


        if (prediction == currentImageLabel):
            numCorrect += 1

        plt.plot(index, (numCorrect // (index + 1)), marker="o", markersize=1, markeredgecolor="blue", markerfacecolor="green")
        
        
        expected = [0.0] * len(outputLayer.neurons)
        expected[currentImageLabel] = 1.0
        # compute cost for output layer

        # print("output neurons: ", layers[-1].neurons)

        cost = 0.0
        for f in range(layers[-1].numNeurons):
            cost += math.pow(layers[-1].neurons[f] - expected[f],2)


        plt.plot(index, cost, marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")
        # plt.plot(testOneImageIndex, cost, marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")


        backPropigation()
        
        index += 1
        # testOneImageIndex += 1      #^ just to test if the cost is updating or random

        batchImageCounter += 1

        # if we are done with a batch process the gradient and clear it for next
        if ( (batchImageCounter >= batchSize) ):
            """
            batch has been processed. new weight / bias = old - gradient
            clear all gradients to 0.0
            reset batchImageCounter to 0
            """
            # print("Done with batch - apply and reset")
            for layer in layers[1:]:
                layer.weights = np.subtract(layer.weights, np.multiply(np.divide(layer.gradient_weight, batchSize), learning_rate)  )
                layer.biases = np.subtract(layer.biases,   np.multiply(np.divide(layer.gradient_bias, batchSize), learning_rate) )

                # reset gradient values to 0 for next batch
                # layer.gradient_weight = np.array([[0.0 for col in range(layer.previousLayer.numNeurons)] for row in range(layer.numNeurons)])
                # layer.gradient_bias = np.array([0.0] * layer.numNeurons)

                batchImageCounter = 0

            print("--------------------------------------------")
            print("Weight gradient output layer: ", np.divide(layer.gradient_weight, batchSize)[0][0])
            print("--------------------------------------------")
    

    # test model on an image
    testImage = images[index + 1]       
    testImageLabel = labels[index + 1]

    
    # turn input image from 0-255 into 0-1
    for x in range(inputLayerLen):
        testImage[x] = testImage[x] / 255

    inputLayer.neurons = testImage

    expected = [0.0] * len(outputLayer.neurons)
    expected[testImageLabel] = 1.0


    forwardPropigation()

    print("Output Layer: ", outputLayer.neurons)
    print("-------------------------------------")
    print("Test Label: ", testImageLabel)
    print("Expected array: ", expected)

    plt.show()


    forwardPropigation


    
main()





