from random import uniform
from math import sqrt

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
    train_output_array.append(float(line))
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
                if current_data[i][j] <= 0:
                    current_data[i][j] *= beta
        return Matrix(current_data)

    def derivative(self, beta=0.1):
        current_data = self.data
        for i in range(len(current_data)):
            for j in range(len(current_data[0])):
                if current_data[i][j] < 0:
                    current_data[i][j] = -beta
                else:
                    current_data[i][j] = 1
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

    def s_multiply(self, b):
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
    def __init__(self, number_of_neurons, previous_layer) -> None:
        self.number_of_neurons: Matrix = number_of_neurons
        self.previous_layer: Matrix = previous_layer
        weights = []

        for i in range(len(maxes)):
            row = []
            for j in range(number_of_neurons):
                row.append(uniform(0, 30))
            weights.append(row)
        self.weights = Matrix(weights)

    def forward(self):
        self.weightedInputs: Matrix = self.previous_layer.multiply(
            self.weights)
        self.current_layer: Matrix = self.weightedInputs.activate()
        return self.current_layer


class NeuralNetwork:
    def __init__(self, train_input, train_output) -> None:
        self.train_input: Matrix = train_input
        self.train_output: Matrix = train_output
        self.layers: list[Layer] = []
        self.d_weights: list[Matrix] = []
        self.unit_delta_matrix = self.create_unit_delta_matrix()

    def forward(self, hidden_layers):
        for i in range(len(hidden_layers)):
            if i == 0:
                self.layers.append(Layer(hidden_layers[i], self.train_input))
            else:
                self.layers.append(
                    Layer(hidden_layers[i], self.layers[i-1].forward()))
        self.layers[-1].forward()
        self.train_predictions = Matrix(
            [p[0] for p in self.layers[-1].current_layer.data])

    # rmse
    def error(self):
        sum_result = 0
        for i in range(len(self.train_output.data)):
            sum_result += (self.train_output.data[i]-self.train_predictions.data[i])*(
                self.train_output.data[i]-self.train_predictions.data[i])
        return sqrt(sum_result/len(self.train_output.data))

    def create_unit_delta_matrix(self):
        unit_matrix = []
        for i in range(len(self.train_output)):
            row = []
            for j in range(len(self.train_output)):
                if i==j:row.append(1)
                else: row.append(0)
            unit_matrix.append(row)
        return Matrix(unit_matrix)
    
    def backpropagation(self):
        deltas: list[Matrix] = [self.unit_delta_matrix]
        for i in range(len(self.layers), 0, -1):
            # delta[n] = delta[n+1] * W[n]^T * f'(Z[n]) in matrices
            delta = deltas[-1].multiply(self.layers[i].weights.transpose().multiply(
                self.layers[i].weightedInputs.derivative()))
            deltas.append(delta)
            weight = self.layers[i].previous_layer.multiply(delta)
            self.d_weights.append(weight)


### PREPARE NETWORK ###
train_input = Matrix(train_input_array)
train_output = Matrix(train_output_array)
test_input = Matrix(test_input_array)

nn = NeuralNetwork(train_input, train_output)
