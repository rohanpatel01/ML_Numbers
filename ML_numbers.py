import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)

import mnist
from mnist import MNIST
from typing import Type
import random
import math
import matplotlib.pyplot as plt

"""

* look at asian guy's code on youtube to see what went wrong

* check to make sure we have computed and updated the gradient bias and biases correctly

* try soft max instead of sigmoid for output layer - probably will not help

Issue: vanishing gradient problem?
- gradients are too small to begin with and remain the same


"""

def sigmoid(x):
    return (1.0/(1.0 + np.exp(-x)))


def derivative_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))
    # return (np.exp(-x) / (math.pow((1 + np.exp(-x)), 2)))

def reLu(x):
    if x < 0:
        return x * 0.1      # leaky reLu below 0 => 0.1 * x - try leaky reLu of 0.01 * x for < 0
    else:
        return x          
    
def derivative_reLu(x):
    if x < 0:
        return 0
    else:
        return 1




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


#^ try not rounding the weight init
def heWeightInit(numNodesPrevLayer):
    lower = 0
    upper = math.sqrt(2 / numNodesPrevLayer)
    return random.uniform(lower, upper)


mndata = MNIST('samples')
images, labels = mndata.load_training()

learning_rate = 0.01            # was 0.002125    - 0.1 in video

offset = 0
index = 0 + offset
# ^^ Don't delete - for actual image

currentImage = images[index]
# print("current image: ", currentImage)
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
            
            #^ weight init
            self.weights = np.array([[normalizedXavierWeightInit(previousLayer.numNeurons, numNeurons) for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            # self.weights = np.array([[xaviewWeightInit(previousLayer.numNeurons) for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            # self.weights = np.array([[heWeightInit(previousLayer.numNeurons) for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            

            self.gradient_weight = np.array([[0.0 for col in range(previousLayer.numNeurons)] for row in range(numNeurons)])
            self.gradient_bias = np.array([0.0] * numNeurons)

        self.biases = np.array([0.0 for x in range(numNeurons)])
        

        

    def forwardPropigation(self):
        # note: self.z is a vector not a single value. It holds the Z values for each neuron in the layer
        
        # z = a(l-1) * w
        self.z = [0.0] * self.numNeurons

        for x in range(self.numNeurons):
            sum = 0.0
            for y in range(self.previousLayer.numNeurons):
                sum += (self.previousLayer.neurons[y] * self.weights[x][y]) 

            self.z[x] = sum


        # self.z = np.dot(self.weights, self.previousLayer.neurons) #+ self.biases # was matmul - keeping dot since that's what the video does
        #^ apply sigmoid on output layer only when using reLu activation function
        for x in range(self.numNeurons):
            if not self.nextLayer:
                self.neurons[x] = sigmoid(self.z[x] + self.biases[x] )          # added bias
            else:
                self.neurons[x] = reLu(self.z[x] + self.biases[x])              # added bias
        

        


    def backPropigation(self):

        self.dC_dAL = np.array([0.0] * self.numNeurons) 
        
        for currentLayerNeuronIndex in range(len(self.neurons)):
            dSigmoid_ZCurrent = derivative_reLu(self.z[currentLayerNeuronIndex])   # da / dZ        # derivative_sigmoid

            

            # output layer
            if not self.nextLayer:

                outputCost = ((self.neurons[currentLayerNeuronIndex] - expected[currentLayerNeuronIndex]))  # remove 2*
                self.dC_dAL[currentLayerNeuronIndex] = outputCost

                # calculate bias
                # self.gradient_bias[currentLayerNeuronIndex] += (dSigmoid_ZCurrent * outputCost)            # added bias

            else:
                for nextLayerIndex in range(self.nextLayer.numNeurons):
                    weight = self.nextLayer.weights[nextLayerIndex][currentLayerNeuronIndex]
                    dSigmoid_ZNext = derivative_reLu(self.nextLayer.z[nextLayerIndex])      # derivative_sigmoid
                    costNextNeuron = self.nextLayer.dC_dAL[nextLayerIndex]
                    self.dC_dAL[currentLayerNeuronIndex] += (weight * dSigmoid_ZNext * costNextNeuron)



            self.gradient_bias[currentLayerNeuronIndex] += (dSigmoid_ZCurrent * self.dC_dAL[currentLayerNeuronIndex] )     # added bias
                
            
            for prevLayerWeightIndex in range(len(self.previousLayer.neurons)):
                prevNeuron = self.previousLayer.neurons[prevLayerWeightIndex]
                
                self.gradient_weight[currentLayerNeuronIndex][prevLayerWeightIndex] += prevNeuron * dSigmoid_ZCurrent * self.dC_dAL[currentLayerNeuronIndex]    # added +=

        
                # find bias for each neuron in hidden layer
                # dSigmoid_ZCurrent_for_bias = derivative_reLu(self.z[currentLayerNeuronIndex])    # derivative_sigmoid
                # costCurrentNeuron = self.dC_dAL[currentLayerNeuronIndex]
                # self.gradient_bias[currentLayerNeuronIndex] += (dSigmoid_ZCurrent_for_bias * costCurrentNeuron)             # was learning_rate *



                
#^^ For testing
# inputLayer = Layer(previousLayer=None, numNeurons=3, neurons=np.array([0.3, 0.5, 0.7]))

#^ num neurons per layer
inputLayerLen = len(currentImage)       # was inputLayer.numNeurons 
hiddenLayer1Len = 70      # was 30 - cut down number of neurons per layer and try printing them out to see them better
hiddenLayer2Len = 70    # was  30
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

#^ weight init
inputLayer.weights = None
hiddenLayer1.weights = np.array([[normalizedXavierWeightInit(inputLayer.numNeurons, hiddenLayer1.numNeurons) for col in range(inputLayer.numNeurons)] for row in range(hiddenLayer1.numNeurons)])
hiddenLayer2.weights = np.array([[normalizedXavierWeightInit(hiddenLayer1.numNeurons, hiddenLayer2.numNeurons) for col in range(hiddenLayer1.numNeurons)] for row in range(hiddenLayer2.numNeurons)])
outputLayer.weights = np.array([[normalizedXavierWeightInit(hiddenLayer2.numNeurons, outputLayer.numNeurons) for col in range(hiddenLayer2.numNeurons)] for row in range(outputLayer.numNeurons)])

# hiddenLayer1.weights = np.array([[heWeightInit(inputLayer.numNeurons) for col in range(inputLayer.numNeurons)] for row in range(hiddenLayer1.numNeurons)])
# hiddenLayer2.weights = np.array([[heWeightInit(hiddenLayer1.numNeurons) for col in range(hiddenLayer1.numNeurons)] for row in range(hiddenLayer2.numNeurons)])
# outputLayer.weights = np.array([[heWeightInit(hiddenLayer2.numNeurons) for col in range(hiddenLayer2.numNeurons)] for row in range(outputLayer.numNeurons)])


# hiddenLayer1.weights = np.array([[xaviewWeightInit(inputLayer.numNeurons) for col in range(inputLayer.numNeurons)] for row in range(hiddenLayer1.numNeurons)])
# hiddenLayer2.weights = np.array([[xaviewWeightInit(hiddenLayer1.numNeurons) for col in range(hiddenLayer1.numNeurons)] for row in range(hiddenLayer2.numNeurons)])
# outputLayer.weights = np.array([[xaviewWeightInit(hiddenLayer2.numNeurons) for col in range(hiddenLayer2.numNeurons)] for row in range(outputLayer.numNeurons)])

# for test - manually forcing val ues / environment that would be seen after forward propigation
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

    global index
    global expected
    global layers

    numCorrect = 0

    batchImageCounter = 0
    batchSize = 10.0            # was 100
    numBatchesToProcess = 100.0     # was 5

    x = np.array([x] for x in range(0))
    plt.xlim(offset, (batchSize * numBatchesToProcess) + offset)
    plt.ylim(-5, 5)
    plt.grid()

    random.shuffle(images)
    
    #^ Note: change btwn one image and all images: add statement to while loop, change what you plot, comment back in the index += 1 instead of "testOneImageIndex"
    
    testOneImageIndex = 0
    predictionSum = 0.0
    while ( (index < numTrainingImages) and (index - offset < (numBatchesToProcess * batchSize)) and (testOneImageIndex < (numBatchesToProcess * batchSize) )  ):      # 
        
        # process image with back and forward propigation
        currentImage = images[index]
        currentImageLabel = labels[index]

        expected = [0.0] * len(outputLayer.neurons)
        expected[currentImageLabel] = 1.0

        # normalize input to [0,1]
        for x in range(inputLayerLen):
            currentImage[x] = currentImage[x] / 255 # try 255.0

        # normalizeInput(currentImage) # bad performance with normalizing the input

        # print(currentImage)

        inputLayer.neurons = np.array(currentImage)
       

        forwardPropigation()


        # find accuracy

        prediction = list((layers[-1].neurons)).index(max(layers[-1].neurons))
        predictionSum += prediction



        if (prediction == currentImageLabel):
            numCorrect += 1

        #^ plot accuracy
        # plt.plot(index, (numCorrect / (index + 1)), marker="o", markersize=1, markeredgecolor="blue", markerfacecolor="green")
        plt.plot(testOneImageIndex, (numCorrect / (testOneImageIndex + 1)), marker="o", markersize=1, markeredgecolor="blue", markerfacecolor="green")
        
        # max_value = layers[1].gradient_weight[0][0]
        # for row in layers[1].gradient_weight:
        #     for value in row:
        #         max_value = max(max_value, value)


        # plot max of hidden layer 1 weights b/c most will be 0 b/c most of input neuron vals are 0 
        # plt.plot(index, max_value, marker="o", markersize=1, markeredgecolor="purple", markerfacecolor="green")
        
        #^ plot gradients
        # plt.plot(index, max_value, marker="o", markersize=1, markeredgecolor="green", markerfacecolor="green")
        # plt.plot(index, layers[2].gradient_weight[0][0], marker="o", markersize=1, markeredgecolor="black", markerfacecolor="green")
        # plt.plot(index, layers[3].gradient_weight[0][0], marker="o", markersize=1, markeredgecolor="orange", markerfacecolor="green")

        # plot biases
        # plt.plot(index, layers[3].gradient_bias[0], marker="o", markersize=1, markeredgecolor="green", markerfacecolor="green")
        # plt.plot(index, layers[2].gradient_bias[0], marker="o", markersize=1, markeredgecolor="cyan", markerfacecolor="green")


        expected = [0.0] * len(outputLayer.neurons)
        expected[currentImageLabel] = 1.0
        # compute cost for output layer

        # print("output neurons: ", layers[-1].neurons)

        cost = 0.0
        for f in range(layers[-1].numNeurons):
            # cost += math.pow(layers[-1].neurons[f] - expected[f],2)   #^ remove square from cost
            cost += layers[-1].neurons[f] - expected[f]



        # plt.plot(index, cost, marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")
        plt.plot(testOneImageIndex, cost, marker="o", markersize=1, markeredgecolor="red", markerfacecolor="green")


        backPropigation()
        
        # index += 1
        testOneImageIndex += 1      #^ just to test if the cost is updating or random

        batchImageCounter += 1

        #^ after every batch
        if ( (batchImageCounter >= batchSize) ):
            
            for layer in layers[1:]:
                # newWeight = (currentWeight - ((learning rate) * average gradient)) + bias

                for row in range(layer.numNeurons):
                    for col in range(layer.previousLayer.numNeurons):
                        layer.weights[row][col] = layer.weights[row][col] - (learning_rate * (layer.gradient_weight[row][col] / batchSize))
                
                for col in range(layer.numNeurons):
                    layer.biases[col] = layer.biases[col] - (learning_rate * (layer.gradient_bias[col] / batchSize))        # added bias

                # layer.weights = np.subtract(layer.weights, np.multiply(np.divide(layer.gradient_weight, batchSize), learning_rate)  ) 
                # layer.biases = np.subtract(layer.biases,   np.multiply(np.divide(layer.gradient_bias, batchSize), learning_rate) )   

                # reset gradient values to 0 for next batch

                layer.gradient_weight = np.array([[0.0 for col in range(layer.previousLayer.numNeurons)] for row in range(layer.numNeurons)])
                layer.gradient_bias = np.array([0.0] * layer.numNeurons)



            batchImageCounter = 0


    # print("Average output: ", predictionSum / index + 1)

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


    forwardPropigation()


    
main()





