import unittest
from matrix import Matrix
from layer import Layer

# class TestLayer(unittest.TestCase):
    
#     def test_init(self):
#         previous = [[1,0],[0,1]]
#         currentLayer = Layer(10,previous)
#         print(currentLayer)


previous = [[1,0],[0,1],[1,1],[1,0],[0,1],[1,1]]
currentLayer = Layer(10,Matrix(previous))
currentLayer.forward()
print(currentLayer)