import unittest

from common.helpers.computeDistances import *


class TestExtractFeatures(unittest.TestCase):

    def setUp(self):
        self.image = np.array([
            [1, 3, 5, 2],
            [5, 10, 2, 4],
            [1, 1, 1, 4]
        ])

    def testExtractingPretrainedFeatures(self):
        pass

    def testExtractingWhiteBoxFeatures(self):
        pass
if __name__ == '__main__':
    unittest.main()