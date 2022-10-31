import unittest
from MyImplementations.Mean_Shift import Mean_Shift
import numpy as np

class TestMeanShift(unittest.TestCase):
    def test_distance(self):
        # Create object
        obj = Mean_Shift()

        # Create 1D and 2D Matrices
        A = np.array([[1,2]])
        B = np.array([[1,4]])

        AA = np.array([[1,2],
                       [1,2]])
        BB = np.array([[1,4],
                       [1,4]])

        # Test cases for 1D Matrices
        self.assertAlmostEqual(obj.distance(A, B), 2)
        self.assertAlmostEqual(obj.distance(A, A), 0)
        self.assertAlmostEqual(obj.distance(B, B), 0)
        self.assertAlmostEqual(obj.distance(B, A), 2)

        # Test cases for 2D Matrices
        self.assertEqual(np.allclose(obj.distance(AA, BB), np.array([2, 2])), True)
        self.assertEqual(np.allclose(obj.distance(AA, AA), np.array([0, 0])), True)
        self.assertEqual(np.allclose(obj.distance(BB, BB), np.array([0, 0])), True)
        self.assertEqual(np.allclose(obj.distance(BB, AA), np.array([2, 2])), True)


    def test_gaussian_kernel(self):
        # Create object
        obj = Mean_Shift()

        # Test Functionality for scalar input
        self.assertAlmostEqual(obj.gaussian_kernel(0), 1)
        self.assertAlmostEqual(obj.gaussian_kernel(1), np.exp(-1/2))
        self.assertAlmostEqual(obj.gaussian_kernel(2), np.exp(-1))

        # Test Functionality for array input
        self.assertEqual(np.allclose(obj.gaussian_kernel(np.array([0, 0])), np.array([1, 1])), True)
        self.assertEqual(np.allclose(obj.gaussian_kernel(np.array([0, 1])), np.array([1, np.exp(-1/2)])), True)
        self.assertEqual(np.allclose(obj.gaussian_kernel(np.array([2, 0])), np.array([np.exp(-1), 1])), True)

        # Test Restrictions
        self.assertRaises(ValueError, obj.gaussian_kernel, -2) # input cannot be negative, is ValueError being raised?

if __name__ == '__main__':
    unittest.main()
