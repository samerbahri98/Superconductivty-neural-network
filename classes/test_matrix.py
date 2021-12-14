import unittest
import copy
from matrix import Matrix

class TestMatrix(unittest.TestCase):

    def setUp(self):
        self.m1 = Matrix([[1,2,3],[4,5,6]])
        self.m2 = Matrix([[1,2,3],[4,5,6]])
        self.m3 = copy.deepcopy(self.m1).scale(-1)
        self.m4 = Matrix([[-1,-2,-3],[-4,-5,-6]])

    # def test_derivative(self):
    #     m1 = matrix.Matrix([[1,2,-1]])
    #     m2 = matrix.Matrix([[1,1,-0.1]])
    #     self.assertEqual(m1.derivative().data,m2.data)
        
    # def test_transpose(self):
    #     m1 = matrix.Matrix([[1,2,3],[4,5,6]])
    #     m2 = matrix.Matrix([[1,4],[2,5],[3,6]])
    #     self.assertEqual(m1.transpose().data,m2.data)
 
    def test_scale(self):
        self.assertEqual(self.m3.data,self.m4.data)
        
    def test_mutation(self):
        self.assertEqual(self.m1.data,self.m2.data)

if __name__ == '__main__':
    unittest.main()