#!/usr/bin/env python3

import unittest
import numpy as np
import logging

from .context import imagedata
from imagedata.series import Series

class TestSeries(unittest.TestCase):

    #@unittest.skip("skipping test_create_series")
    def test_create_series(self):
        a = np.eye(128)
        si = Series(a)
        self.assertEqual(si.dtype, np.float64)
        self.assertEqual(si.shape, (128, 128))

    #@unittest.skip("skipping test_print_header")
    def test_print_header(self):
        a = np.eye(128)
        si = Series(a)
        print(si.slices)
        si.spacing = (1, 1, 1)
        print(si.spacing)
        self.assertEqual(si.slices, 1)
        np.testing.assert_array_equal(si.spacing, np.array((1, 1, 1)))

    #@unittest.skip("skipping test_copy_series")
    def test_copy_series(self):
        a = np.eye(128)
        si1 = Series(a)
        si1.spacing = (1, 1, 1)

        si2 = si1
        print(si2.slices)
        print(si2.spacing)
        self.assertEqual(si2.slices, 1)
        np.testing.assert_array_equal(si2.spacing, np.array((1, 1, 1)))

    #@unittest.skip("skipping test_copy_series2")
    def test_copy_series2(self):
        a = np.eye(128)
        si1 = Series(a)
        si1.spacing = (1, 1, 1)

        si2 = si1.copy()
        print(si2.slices)
        print(si2.spacing)
        self.assertEqual(si2.slices, 1)
        np.testing.assert_array_equal(si2.spacing, np.array((1, 1, 1)))

    #@unittest.skip("skipping test_subtract_series")
    def test_subtract_series(self):
        a = Series(np.eye(128))
        b = Series(np.eye(128))
        a.spacing = (1, 1, 1)
        b.spacing = a.spacing

        si = a - b
        print(si.slices)
        print(si.spacing)
        print(type(si.spacing))
        self.assertEqual(si.slices, 1)
        np.testing.assert_array_equal(si.spacing, np.array((1, 1, 1)))

if __name__ == '__main__':
    unittest.main()
