from random import uniform

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
        fields.append[value]
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


for _ in range(4253):
    line = input()
    line = line.split("\t")
    fields = []
    for i in range(len(line)):
        value = float(line[i])
        fields.append[(value-mins[i])/maxes[j]]
    test_input_array.append(fields)


# Classes
class Matrix:
    def __init__(self, data):
        self.data = data

    def multiply(self, b):
        result = []
        for i in range(len(self.data)):
            for j in range(len(b.data[0])):
                for k in range(len(b)):
                    result[i][j] += self.data[i][k] * b.data[k][j]
        return Matrix(result)

    # Leaky Relu
    def activate(self, beta=0.1):
        current_data = self.data
        for i in range(len(current_data)):
            for j in range(len(current_data[0])):
                if current_data[i][j] <= 0:
                    current_data[i][j] *= beta
        return Matrix(current_data)


class Layer:
    def __init__(self, number_of_neurons, previous_layer) -> None:
        self.number_of_neurons: Matrix = number_of_neurons
        self.previous_layer: Matrix = previous_layer
        weights = []
        for i in range(len(maxes)):
            for j in range(number_of_neurons):
                weights[i, j] = uniform(0, 5000)
        self.weights = Matrix(weights)

    def forward(self):
        multiplication: Matrix = self.previous_layer.multiply(self.weights)
        self.current_layer = multiplication.activate()
        return self.current_layer


# PREPARE NETWORK


train_input = Matrix(train_input_array)
train_output = Matrix(train_output_array)
test_input = Matrix(test_input_array)

result = []

for i in result:
    if(i != ""):
        print(i)
