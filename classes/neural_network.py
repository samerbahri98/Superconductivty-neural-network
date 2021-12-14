from matrix import Matrix
from random import uniform
from math import sqrt
from layer import Layer
import copy



class NeuralNetwork:
    def __init__(self, train_input, train_output, hidden_layers) -> None:
        self.train_input: Matrix = train_input
        self.train_output: Matrix = train_output
        # hidden layers is an array of the number of neurons in the layer[i]
        self.hidden_layers: list[int] = hidden_layers
        self.layers: list[Layer] = []
        self.first_wave()

    def first_wave(self):
        for i in range(len(self.hidden_layers)):
            if i == 0:
                self.layers.append(
                    Layer(self.hidden_layers[i], self.train_input))
            else:
                self.layers.append(
                    Layer(self.hidden_layers[i], self.layers[i-1].forward()))
        self.layers[-1].forward()
        self.train_predictions = Matrix(self.layers[-1].current_layer.data)

    def forward(self):
        self.layers[0].forward()
        for i in range(1, len(self.hidden_layers)):
            self.layers[i].previous_layer = copy.deepcopy(self.layers[i-1].current_layer)
            self.layers[i].forward()
        self.train_predictions = Matrix(self.layers[-1].current_layer.data)

    # rmse
    def error(self):
        sum_result = 0
        for i in range(len(self.train_output.data)):
            sum_result += (self.train_output.data[i]
                           [0]-self.train_predictions.data[i][0])**2
        return sqrt(sum_result/len(self.train_output.data))

    def backpropagation(self):
        self.d_weights: list[Matrix] = []
        self.J = self.error()
        # ititialize deltas[max] = (y-yhat) ° f'(Z[n])/(JN)
        delta = copy.deepcopy(self.train_predictions).scale(-1)
        delta = copy.deepcopy(delta).add(self.train_output)
        dw = copy.deepcopy(
            self.layers[-1].weightedInputs).derivative()
        delta = copy.deepcopy(delta).hadamard(dw)
        ratio = 1/(self.J*len(self.train_predictions.data))
        delta = copy.deepcopy(delta).scale(
            ratio)
        deltas: list[Matrix] = [delta]
        for i in range(len(self.layers)-1, -1, -1):
            # dJ/dW[i] = a[i-1]^T × delta[i]
            a_T = copy.deepcopy(self.layers[i].previous_layer).transpose()
            d_weight = copy.deepcopy(a_T).multiply(delta)
            self.d_weights.append(d_weight)
            if i == 0:
                break
            # deltas[n] = deltas[n+1] × (W[n]^T ° f'(Z[n])) in matrices
            f_prime = copy.deepcopy(
                self.layers[i-1].weightedInputs).derivative()
            w_T = copy.deepcopy(self.layers[i].weights).transpose()
            delta = copy.deepcopy(deltas[-1]).multiply(w_T)
            delta = copy.deepcopy(delta).hadamard(f_prime)
            deltas.append(delta)
        self.d_weights.reverse()

    def adjust(self):
        for i in range(len(self.layers)):
            d = copy.deepcopy(self.d_weights[i]).scale(self.rate)
            self.layers[i].weights = self.layers[i].weights.add(d)

    def learn(self, target, rate=0.01):
        self.rate = rate
        while True:
            self.backpropagation()
            self.adjust()
            # self.rate /= 10
            print(self.J)
            if self.J < 10:
                break
            self.forward()
