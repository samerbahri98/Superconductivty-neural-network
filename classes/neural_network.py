from matrix import Matrix
from random import uniform
from math import sqrt
from layer import Layer


class NeuralNetwork:
    def __init__(self, train_input, train_output) -> None:
        self.train_input: Matrix = train_input
        self.train_output: Matrix = train_output
        self.layers: list[Layer] = []

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
                if i == j:
                    row.append(1)
                else:
                    row.append(0)
            unit_matrix.append(row)
        return Matrix(unit_matrix)

    def backpropagation(self):
        self.d_weights: list[Matrix] = []
        J = self.error()
        # ititialize deltas[max] = I/(JN)
        delta = self.unit_delta_matrix.scale(1/(J*len(self.train_output.data)))
        deltas: list[Matrix] = [delta]
        for i in range(len(self.layers), 0, -1):
            # deltas[n] = deltas[n+1] × (W[n]^T ° f'(Z[n])) in matrices
            f_prime = self.layers[i].weightedInputs.derivative()
            w_T = self.layers[i].weights.transpose()
            delta = deltas[-1].multiply(w_T.hadamard(f_prime))
            deltas.append(delta)
            # dJ/dW[i] = a[i-1]^T × delta[i]
            a_T = self.layers[i].previous_layer.transpose()
            d_weight = a_T.multiply(delta)
            self.d_weights.append(d_weight)
