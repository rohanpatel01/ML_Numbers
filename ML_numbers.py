from Layer import Layer

import numpy as numpy
from mnist import MNIST

mndata = MNIST('samples')
images, labels = mndata.load_training()

numNeuronsInput = len(images[0])
numNeuronsH1 = 50
numNeuronsH2 = 50
numNeuronsOutput = 10

print("numNeuronsInput: ", numNeuronsInput)

inputLayer = Layer(numNeurons=numNeuronsInput)
h1Layer = Layer(numNeurons=numNeuronsH1)
h2Layer = Layer(numNeurons=numNeuronsH2)
outputLayer = Layer(numNeurons=numNeuronsOutput)

inputLayer.nextLayer = h1Layer
h1Layer.nextLayer = h2Layer
h2Layer.nextLayer = outputLayer
outputLayer.nextLayer = None






