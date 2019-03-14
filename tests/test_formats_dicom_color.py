#!/usr/bin/env python3

import unittest
import sys
import shutil
import numpy as np
import logging
import argparse

from .context import imagedata
import imagedata.cmdline
import imagedata.readdata
import imagedata.formats
from imagedata.series import Series
from .compare_headers import compare_headers

class test_dicom_color(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'dicom', '--serdes', '1'])
        self.opts_gray = parser.parse_args(['--of', 'dicom', '--serdes', '1',
            '--psopt', 'pnggray'])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        shutil.rmtree('ttdc', ignore_errors=True)

    #@unittest.skip("skipping test_read_dicom_color")
    def test_read_dicom_color(self):
        si1 = Series('data/lena_color.jpg')
        si2 = Series('data/dicom/lena_color.dcm')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        self.assertEqual(si1.color, si2.color)
        np.testing.assert_array_equal(si1, si2)

    #@unittest.skip("skipping test_write_dicom_color")
    def test_write_dicom_color(self):
        si1 = Series(
            'data/lena_color.jpg',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (512, 512, 3))
        si1.write('ttdc', formats=['dicom'])
        si2 = Series('ttdc')
        np.testing.assert_array_equal(si1, si2)

if __name__ == '__main__':
    unittest.main()
