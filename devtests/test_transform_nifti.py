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

class TestTransformNifti(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #sys.argv[1:] = ['aa', 'bb']
        #self.opts = parser.parse_args(['--order', 'none'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['nifti']

    def tearDown(self):
        shutil.rmtree('tttn', ignore_errors=True)

    #@unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'data/dicom/time/time00',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (40, 192, 152))

        si1.write('tttn', formats=['nifti'])
        si2 = Series('tttn/')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_almost_equal(si1.transformationMatrix, si2.transformationMatrix, decimal=2)

        # The following file was created using MRIConvert
        si3 = Series('data/nifti/MRIConvert/PHANTOM_T1_20190207_001_014_fl3d_dynamic.hdr')
        np.testing.assert_array_almost_equal(si1.transformationMatrix, si3.transformationMatrix, decimal=2)

if __name__ == '__main__':
    unittest.main()
