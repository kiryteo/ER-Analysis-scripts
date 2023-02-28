import numpy as np
import unittest
from junction_analysis import get_junc_patches

class TestGetJuncPatches(unittest.TestCase):

    def test_get_junc_patches(self):
        img = np.zeros((128, 128))
        img[50:53, 50:53] = 1
        newps = [(51, 51), (52, 51), (51, 52), (52, 52)]
        patches = get_junc_patches(newps, img)
        expected_patches = [[0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0, 0, 0, 0, 0]]
        self.assertEqual(patches, expected_patches)

if __name__ == '__main__':
    unittest.main()
