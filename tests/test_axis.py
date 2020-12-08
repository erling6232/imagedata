#!/usr/bin/env python3

import unittest
import numpy as np
import logging

# from .context import imagedata
from imagedata.axis import UniformAxis, UniformLengthAxis, VariableAxis


class TestAxis(unittest.TestCase):

    # @unittest.skip("skipping test_uniform_axis")
    def test_uniform_axis(self):
        axis = UniformAxis('space', 1, 21, 2)
        # axis has values: 1 3 5 7 9 11 13 15 17 19
        logging.debug('test_uniform_axis: {}'.format(axis.slice))
        all_axis = axis[:]
        self.assertEqual((1, 21, 2), all_axis.slice)
        self.assertEqual(10, len(all_axis))
        logging.debug('test_uniform_axis: all_axis: %s' % all_axis.name)
        logging.debug('test_uniform_axis: start,stop,step: %d,%d,%d' % (all_axis.start, all_axis.stop, all_axis.step))
        new_axis = axis[2:]
        self.assertEqual((5, 21, 2), new_axis.slice)
        self.assertEqual(8, len(new_axis))
        logging.debug('test_uniform_axis: new_axis: %s' % new_axis.name)
        logging.debug('test_uniform_axis: start,stop,step: %d,%d,%d' % (new_axis.start, new_axis.stop, new_axis.step))
        stride_axis = axis[2:6:2]
        self.assertEqual((5, 13, 4), stride_axis.slice)
        self.assertEqual(2, len(stride_axis))

    # @unittest.skip("skipping test_uniform_length_axis")
    def test_uniform_length_axis(self):
        axis = UniformLengthAxis('x', 0.1, 10)
        # axis has values: 0.1 1.1 2.1 3.1 4.1 5.1 6.1 7.1 8.1 9.1
        logging.debug('test_uniform_length_axis: {}'.format(axis.slice))
        all_axis = axis[:]
        self.assertEqual((0.1, 10.1, 1), all_axis.slice)
        self.assertEqual(10, len(all_axis))
        new_axis = axis[2:5]
        # new_axis has values: 2.1 3.1 4.1
        self.assertEqual((2.1, 5.1, 1), new_axis.slice)

        axis = UniformLengthAxis('y', 0, 128)
        # axis has values: 0..127
        self.assertEqual((0, 128, 1), axis.slice)
        self.assertEqual(len(axis), 128)

    # @unittest.skip("skipping test_uniform_length_axis_step2")
    def test_uniform_length_axis_step2(self):
        axis = UniformLengthAxis('x', 1, 10, 2)
        # axis has values: 1 3 5 7 9 11 13 15 17 19
        logging.debug('test_uniform_length_axis: {}'.format(axis.slice))
        all_axis = axis[:]
        self.assertEqual(len(all_axis), 10)
        stride_axis = axis[2:6:2]
        # stride_axis has values: 5 9
        self.assertEqual((5, 13, 4), stride_axis.slice)
        self.assertEqual(2, len(stride_axis))

    # @unittest.skip("skipping test_variable_axis")
    def test_variable_axis(self):
        axis = VariableAxis('b', [50, 400, 800])
        # axis has values: 50 400 800
        high_b = axis[1:]
        # high_b has values: 400 800
        np.testing.assert_array_equal(high_b.values, np.array([400, 800]))
        self.assertEqual(len(high_b), 2)

    # @unittest.skip("skipping test_variable_axis2")
    def test_variable_axis2(self):
        axis = VariableAxis('b', [0, 10, 20, 30, 50, 100, 200, 400, 700, 1000])
        # axis has values: 0 10 20 30 50 100 200 400 700 1000
        low_b = axis[1:5]
        # low_b has values: 10 20 30 50
        np.testing.assert_array_equal(low_b.values, np.array([10, 20, 30, 50]))
        stride_b = axis[1:8:2]
        # stride_b has values: 10 30 100 400
        np.testing.assert_array_equal(stride_b.values, np.array([10, 30, 100, 400]))

    # @unittest.skip("skipping test_variable_axis_rgb")
    def test_variable_axis_rgb(self):
        axis = VariableAxis('color', ['R', 'G', 'B'])
        np.testing.assert_array_equal(axis.values, np.array(['R', 'G', 'B']))
        self.assertEqual(axis.values[0], 'R')
        self.assertEqual(axis.values[1], 'G')
        self.assertEqual(axis.values[2], 'B')


if __name__ == '__main__':
    unittest.main()
