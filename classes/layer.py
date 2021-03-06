from matrix import Matrix
from random import uniform
from math import sqrt
import copy


class Layer:
    def __init__(self, number_of_neurons, previous_layer, weights=[]) -> None:
        self.number_of_neurons: int = number_of_neurons
        self.previous_layer: Matrix = previous_layer
        if weights == []:
            self.create_weights()

    def create_weights(self):
        weights = []
        for i in range(len(self.previous_layer.data[0])):
            row = []
            for j in range(self.number_of_neurons):
                row.append(uniform(0, 30))
            weights.append(row)
        self.weights = Matrix(weights)

    def forward(self):
        self.weightedInputs: Matrix = copy.deepcopy(self.previous_layer).multiply(
            self.weights)
        self.current_layer: Matrix = copy.deepcopy(self.weightedInputs).activate()
        return self.current_layer
