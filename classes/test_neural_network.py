import unittest
from matrix import Matrix
from layer import Layer
from neural_network import NeuralNetwork
import matplotlib.pyplot as plt

# class TestNeuralNetwork(unittest.TestCase):
#     def test_feetforward(self):
#         train_input_array = [[1,2],[3,4],[5,6]]
#         train_output_array = [8,8,8]
#         train_input = Matrix(train_input_array)
#         train_output = Matrix(train_output_array)

#         nn = NeuralNetwork(train_input, train_output)
#         nn.forward([7])
#         nn.backpropagation()
#         self.assertEqual()


train_input_array = [[1, 2], [3, 4], [5, 6]]
train_output_array = [[8], [8], [8]]
train_input = Matrix(train_input_array)
train_output = Matrix(train_output_array)

nn = NeuralNetwork(train_input, train_output, [7, 1])
nn.layers[0].weights = Matrix(
    [[1, 1, 1, 2, 2, 2, 3], [-3, -2, -2, -2, -1, -1, -1]])
nn.layers[1].weights = Matrix([[1], [2], [1], [-1], [-2], [-1], [7]])
nn.forward()
nn.learn(10)
print(nn.J)
