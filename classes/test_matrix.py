import unittest
import matrix as matrix

class TestMatrix(unittest.TestCase):

    def test_derivative(self):
        m1 = matrix.Matrix([[1,2,-1]])
        m2 = matrix.Matrix([[1,1,-0.1]])
        self.assertEqual(m1.derivative().data,m2.data)
        
    def test_transpose(self):
        m1 = matrix.Matrix([[1,2,3],[4,5,6]])
        m2 = matrix.Matrix([[1,4],[2,5],[3,6]])
        self.assertEqual(m1.transpose().data,m2.data)
