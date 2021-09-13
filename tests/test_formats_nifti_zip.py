#!/usr/bin/env python3

"""Test zip archive
"""

import unittest
import os.path
import tempfile
import numpy as np
import argparse

from .context import imagedata
import imagedata.cmdline
import imagedata.readdata
import imagedata.formats
from imagedata.series import Series


class TestNiftiZipRead(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['nifti']

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            os.path.join(
                'data',
                'nifti',
                'time_all.zip?time/time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_read_single_file_wildcard")
    def test_read_single_file_wildcard(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip?*/*_14.nii.gz'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_read_single_file_relative")
    def test_read_single_file_relative(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip?time/time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip?*Image_0000[01].mha'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 3, 192, 152))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip?time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_read_all_files")
    def test_read_all_files(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))


class TestNiftiZipWrite(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['nifti']

    # @unittest.skip("skipping test_write_single_file")
    def test_write_single_file(self):
        si1 = Series(os.path.join(
            'data',
            'nifti',
            'time_all.zip?time/time_all_fl3d_dynamic_20190207140517_14.nii.gz'))
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'nifti.zip'), formats=['nifti'])
            si2 = Series(os.path.join(d, 'nifti.zip?Image.nii.gz'))
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_write_all_files")
    def test_write_all_files(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'nifti.zip'), formats=['nifti'])
            si2 = Series(os.path.join(d, 'nifti.zip'))
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)


if __name__ == '__main__':
    unittest.main()
