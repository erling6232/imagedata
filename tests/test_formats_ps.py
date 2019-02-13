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

class Test2DPSPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'ps', '--serdes', '1'])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        try:
            shutil.rmtree('ttp2d')
        except FileNotFoundError:
            pass

    #@unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            'data/ps/pages/A_Lovers_Complaint_1.ps',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (842, 595))

    #@unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [
            'data/ps/pages/A_Lovers_Complaint_1.ps',
            'data/ps/pages/A_Lovers_Complaint_2.ps'
            ],
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (2, 842, 595))

    #@unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'data/ps/pages',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (6, 842, 595))

    #@unittest.skip("skipping test_read_large_file")
    def test_read_large_file(self):
        si1 = Series(
            'data/ps/A_Lovers_Complaint.ps',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (6, 842, 595))

    #@unittest.skip("skipping test_zipread_single_file")
    def test_zipread_single_file(self):
        si1 = Series(
            'data/ps/pages.zip?pages/A_Lovers_Complaint_1.ps',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (842, 595))

    #@unittest.skip("skipping test_zipread_two_files")
    def test_zipread_two_files(self):
        si1 = Series(
            'data/ps/pages.zip?pages/A_Lovers_Complaint_[12].ps',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (2, 842, 595))

    #@unittest.skip("skipping test_zipread_a_directory")
    def test_zipread_a_directory(self):
        si1 = Series(
            'data/ps/pages.zip?pages/',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (6, 842, 595))

    #@unittest.skip("skipping test_zipread_all")
    def test_zipread_all(self):
        si1 = Series(
            'data/ps/pages.zip',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (7, 6, 842, 595))

    @unittest.skip("skipping test_read_PS")
    def test_read_PS(self):
        log = logging.getLogger("Test2DPSPlugin.test_read_PS")
        log.debug("test_read_PS")
        try:
            shutil.rmtree('ttp2d')
        except FileNotFoundError:
            pass
        try:
            si1 = Series(
                'data/ps/A_Lovers_Complaint.ps',
                0,
                self.opts)
        except Exception as e:
            logging.debug('test_read_PS: read si1 exception {}'.format(e))
            raise
        logging.debug('test_read_PS: si1 {} {} {} {}'.format(type(si1), si1.dtype, si1.min(), si1.max()))
        logging.debug('test_read_PS: si1.slices {}'.format(si1.slices))

        si1.spacing = (5, 0.41015625, 0.41015625)
        for slice in range(si1.shape[0]):
            si1.imagePositions = {
                slice:
                        np.array([slice,1,0])
            }
        si1.orientation=np.array([1, 0, 0, 0, 1, 0])
        logging.debug('test_read_PS: si1.tags {}'.format(si1.tags))
        si1.write('ttp2d/biff', 'Image_%05d', formats=['biff'], opts=self.opts)
        logging.debug('test_read_PS: si1 {} {} {}'.format(si1.dtype, si1.min(), si1.max()))

        si2 = Series(
                'ttp2d/biff/Image_00000.biff',
                0,
                self.opts)
        logging.debug('test_read_PS: si2 {} {} {}'.format(si2.dtype, si2.min(), si2.max()))

        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

        logging.debug('test_read_PS: Get si1.slices {}'.format(si1.slices))
        logging.debug('test_read_PS: Set s3')
        s3 = si1.astype(np.float64)
        logging.debug('test_read_PS: s3 {} {} {} {}'.format(type(s3),
            issubclass(type(s3), Series), s3.dtype, s3.shape))
        logging.debug('test_read_PS: s3 {} {} {}'.format(s3.dtype,
            s3.min(), s3.max()))
        logging.debug('test_read_PS: s3.slices {}'.format(s3.slices))
        si3 = Series(s3)
        np.testing.assert_array_almost_equal(si1, si3, decimal=4)
        logging.debug('test_read_PS: si3.slices {}'.format(si3.slices))
        logging.debug('test_read_PS: si3 {} {} {}'.format(type(si3), si3.dtype, si3.shape))
        si3.write('ttp2d/biff', 'Image_%05d.real', formats=['biff'], opts=self.opts)

        s3 = si1-si2
        s3.write('ttp2d/diff', 'Image_%05d.mha', formats=['itk'], opts=self.opts)

if __name__ == '__main__':
    unittest.main()
