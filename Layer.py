from typing import Type
import math
import numpy as np


"""
Return weight 2D array with guassian distribution.
Each row in weight corresponds to all weight connected from previous layer to ith neuron in current layer
"""
def heWeightInit(numNeuronsPrevLayer, numNeuronsCurrentLayer):
    stdDeviation = math.sqrt(2.0 / numNeuronsPrevLayer)
    weightInit = [ (np.randn(numNeuronsCurrentLayer) * stdDeviation) for row in range(numNeuronsCurrentLayer)]
    return weightInit


class Layer:

    def __init__(self, numNeurons, neurons = None, previousLayer: Type['Layer'] = None, nextLayer: Type['Layer'] = None) -> None:
        
        self.numNeurons = numNeurons
        self.nextLayer = nextLayer
        self.previousLayer = previousLayer

        if self.previousLayer is None:   # input layer
            self.neurons = neurons
            self.weights = None
            self.biases = None
            self.z = None
        else:
            self.neurons = [0.0 for x in range(self.numNeurons)]
            self.weights = heWeightInit(self.previousLayer.numNeurons, self.numNeurons)
            self.biases = [0.0 for x in range(self.numNeurons)]
            self.z = [0.0 for x in range(self.numNeurons)]




        