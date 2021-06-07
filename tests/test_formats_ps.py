#!/usr/bin/env python3

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
# from .compare_headers import compare_headers


class Test2DPSPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'ps', '--serdes', '1'])
        self.opts_gray = parser.parse_args(['--of', 'ps', '--serdes', '1',
                                            '--psopt', 'driver=pnggray'])

        plugins = imagedata.formats.get_plugins_list()
        self.ps_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'ps':
                self.ps_plugin = pclass
        self.assertIsNotNone(self.ps_plugin)

    # @unittest.skip("skipping test_read_single_file_gray")
    def test_read_single_file_gray(self):
        si1 = Series(
            os.path.join('data', 'ps', 'pages', 'A_Lovers_Complaint_1.ps'),
            'none',
            self.opts_gray)
        self.assertEqual(si1.input_format, 'ps')
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (1754, 1240))

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            os.path.join('data', 'ps', 'pages', 'A_Lovers_Complaint_1.ps'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (1754, 1240, 3))

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [
                os.path.join('data', 'ps', 'pages', 'A_Lovers_Complaint_1.ps'),
                os.path.join('data', 'ps', 'pages', 'A_Lovers_Complaint_2.ps')
            ],
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (2, 1754, 1240, 3))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'ps', 'pages'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (6, 1754, 1240, 3))
        # for axis in si1.axes:
        #    logging.debug('test_read_single_directory: axis {}'.format(axis))

    # @unittest.skip("skipping test_read_large_file")
    def test_read_large_file(self):
        si1 = Series(
            os.path.join('data', 'ps', 'A_Lovers_Complaint.ps'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (6, 1754, 1240, 3))

    # @unittest.skip("skipping test_zipread_single_file")
    def test_zipread_single_file(self):
        si1 = Series(
            os.path.join('data', 'ps', 'pages.zip?pages/A_Lovers_Complaint_1.ps'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (1754, 1240, 3))

    # @unittest.skip("skipping test_zipread_two_files")
    def test_zipread_two_files(self):
        si1 = Series(
            os.path.join('data', 'ps', 'pages.zip?pages/A_Lovers_Complaint_[12].ps'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (2, 1754, 1240, 3))

    # @unittest.skip("skipping test_zipread_a_directory")
    def test_zipread_a_directory(self):
        si1 = Series(
            os.path.join('data', 'ps', 'pages.zip?pages/'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (6, 1754, 1240, 3))

    # @unittest.skip("skipping test_zipread_all")
    def test_zipread_all(self):
        si1 = Series(
            os.path.join('data', 'ps', 'pages.zip'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (6, 1754, 1240, 3))

    # @unittest.skip("skipping test_write_single_file")
    def test_write_single_file(self):
        si1 = Series(
            os.path.join('data', 'ps', 'pages.zip?pages/A_Lovers_Complaint_1.ps'),
            'none',
            self.opts)
        try:
            with tempfile.TemporaryDirectory() as d:
                si1.write(d + '?Image%05d.ps', formats=['ps'])
        except imagedata.formats.WriteNotImplemented:
            pass

    # @unittest.skip("skipping test_write_single_directory")
    def test_write_single_directory(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time'),
            'none',
            self.opts)
        try:
            with tempfile.TemporaryDirectory() as d:
                si1.write(d + '?Image%05d.ps', formats=['ps'])
        except imagedata.formats.WriteNotImplemented:
            pass


if __name__ == '__main__':
    unittest.main()
