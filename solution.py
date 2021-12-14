from random import uniform
from math import sqrt
import json
import copy

train_input_array = []
train_output_array = []
test_input_array = []
maxes = [0 for _ in range(81)]
mins = [0 for _ in range(81)]


### DATA INPUT ###
for _ in range(17011):
    line = input()
    line = line.split("\t")
    fields = []
    for i in range(len(line)):
        value = float(line[i])
        fields.append(value)
        if(maxes[i] < value):
            maxes[i] = value
        if(mins[i] > 0):
            mins[i] = value

    train_input_array.append(fields)

for i in range(17011):
    line = input()
    train_output_array.append([float(line)])
    for j in range(len(train_input_array[i])):
        train_input_array[i][j] = (train_input_array[i][j]-mins[j])/maxes[j]


for _ in range(4252):
    line = input()
    line = line.split("\t")
    fields = []
    for i in range(len(line)):
        value = float(line[i])
        fields.append((value-mins[i])/maxes[j])
    test_input_array.append(fields)


### CLASSES ###

class Matrix:
    def __init__(self, data):
        self.data = data

    def multiply(self, b):
        result = []
        for i in range(len(self.data)):
            row = []
            for j in range(len(b.data[0])):
                current = 0
                for k in range(len(b.data)):
                    current += self.data[i][k] * b.data[k][j]

                row.append(current)
            result.append(row)
        return Matrix(result)

    # Leaky Relu
    def activate(self, beta=0.1):
        current_data = self.data
        for i in range(len(current_data)):
            for j in range(len(current_data[0])):

                current_data[i][j] = max(
                    current_data[i][j]*beta, current_data[i][j])
        return Matrix(current_data)

    def derivative(self, beta=0.1):
        current_data = self.data
        for i in range(len(current_data)):
            for j in range(len(current_data[0])):
                current_data[i][j] = int(
                    current_data[i][j] >= 0) + int(current_data[i][j] < 0)*beta
        return Matrix(current_data)

    def add(self, b):
        result = self.data
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                result[i][j] += b.data[i][j]
        return Matrix(result)

    def scale(self, b):
        result = self.data
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                result[i][j] *= b
        return Matrix(result)

    def hadamard(self, b):
        result = self.data
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                result[i][j] *= b.data[i][j]
        return Matrix(result)

    def transpose(self):
        result = [[] for i in range(len(self.data[0]))]
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                result[j].append(self.data[i][j])
        return Matrix(result)


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
                row.append(uniform(-100, 100))
            weights.append(row)
        self.weights = Matrix(weights)

    def forward(self):
        self.weightedInputs: Matrix = copy.deepcopy(self.previous_layer).multiply(
            self.weights)
        self.current_layer: Matrix = copy.deepcopy(
            self.weightedInputs).activate()
        return self.current_layer


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
            self.layers[i].previous_layer = copy.deepcopy(
                self.layers[i-1].current_layer)
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

    def learn(self, target, rate=1):
        self.rate = rate
        while True:
            self.backpropagation()
            self.adjust()
            # self.rate /= 10
            print(self.J)
            f = open("./output.txt", "a")
            f.write(json.dumps(copy.deepcopy(self.train_predictions.data)))
            f.close()
            if self.J < 10:
                break
            self.forward()


### PREPARE NETWORK ###
train_input = Matrix(train_input_array)
train_output = Matrix(train_output_array)
test_input = Matrix(test_input_array)

nn = NeuralNetwork(train_input, train_output, [60,1])
nn.learn(10)

print(nn.train_predictions, nn.train_output)
