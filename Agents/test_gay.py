import unittest
import numpy as np
from gay import GAY,Vector_GAE,reversed_GAE


class TestGay(unittest.TestCase):

    def test_2(self):
        rewards = np.ones(2)
        values = np.full(2,0.5)
        result = GAY(rewards,values)
        self.assertEqual(round(np.sum(result),3),1.99)

    def test_3(self):
        rewards = np.array([1,2,3])
        values = np.full(3,1.5)
        result = GAY(rewards,values)
        self.assertEqual(round(np.sum(result),3),8.674)


class TestGAEReversed(unittest.TestCase):

    def test_3(self):
        rewards = np.array([1,2,3])
        values = np.full(3,1.5)
        result = reversed_GAE(rewards,values)
        self.assertEqual(round(np.sum(result),3),8.674)

if __name__ == '__main__':
    unittest.main()