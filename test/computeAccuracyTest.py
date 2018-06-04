import unittest
import numpy as np
from common.computeAccuracy import computeAccuracyUsingDistances

class TestComputeAccuracy(unittest.TestCase):

    def setUp(self):
        self.distances = np.array([
            [4, 20, 2, 12],
            [36, 300, 232, 999],
            [44, 2000, 25, 23]
        ])
        self.shop_labels = np.array(['id_1', 'id_2', 'id_3', 'id_4'])
        self.consumer_labels = np.array(['id_2', 'id_1', 'id_3'])

    def testAccuracyKEquals1(self):
        accuracies = computeAccuracyUsingDistances(self.distances, self.consumer_labels, self.shop_labels, k = [1])
        self.assertEqual(accuracies[0], (1, 3, 1/3))

    def testAccuracyKEquals2(self):
        accuracies = computeAccuracyUsingDistances(self.distances, self.consumer_labels, self.shop_labels, k = [2])
        self.assertEqual(accuracies[0], (2, 3, 2/3))

    def testAccuracyKEquals4(self):
        accuracies = computeAccuracyUsingDistances(self.distances, self.consumer_labels, self.shop_labels, k=[4])
        self.assertEqual(accuracies[0], (3, 3, 1))

    def testAccuracyKEquals3and4(self):
        accuracies = computeAccuracyUsingDistances(self.distances, self.consumer_labels, self.shop_labels, k=[4, 3])
        self.assertEqual(accuracies[0], (3, 3, 1))
        self.assertEqual(accuracies[1], (2, 3, 2/3))


if __name__ == '__main__':
    unittest.main()