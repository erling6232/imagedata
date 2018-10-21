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

class Test3DBiffPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'biff', '--serdes', '1'])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        try:
            shutil.rmtree('ttb3d')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('ttb4d')
        except FileNotFoundError:
            pass

    #@unittest.skip("skipping test_read_3d_biff")
    def test_read_3d_biff(self):
        log = logging.getLogger("TestWritePluginITK.test_read_3d_biff")
        log.debug("test_read_3d_biff")
        try:
            shutil.rmtree('ttb3d')
        except FileNotFoundError:
            pass
        try:
            si1 = Series(
                'tests/dicom/NYRE_151204_T1/_fl3d1_0005',
                0,
                self.opts)
        except Exception as e:
            logging.debug('test_read_3d_biff: read si1 exception {}'.format(e))
        logging.debug('test_read_3d_biff: si1 {} {} {} {}'.format(type(si1), si1.dtype, si1.min(), si1.max()))
        logging.debug('test_read_3d_biff: si1.slices {}'.format(si1.slices))

        si1.spacing = (5, 0.41015625, 0.41015625)
        for slice in range(si1.shape[1]):
            si1.imagePositions = {
                slice:
                        np.array([slice,1,0])
            }
        si1.orientation=np.array([1, 0, 0, 0, 1, 0])
        logging.debug('test_write_3d_biff: si1.tags {}'.format(si1.tags))
        si1.write('ttb3d/biff', 'Image_%05d', formats=['biff'], opts=self.opts)
        logging.debug('test_write_3d_biff: si1 {} {} {}'.format(si1.dtype, si1.min(), si1.max()))

        si2 = Series(
                'ttb3d/biff/Image_00000.biff',
                0,
                self.opts)
        logging.debug('test_write_3d_biff: si2 {} {} {}'.format(si2.dtype, si2.min(), si2.max()))

        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

        logging.debug('test_write_3d_biff: Get si1.slices {}'.format(si1.slices))
        logging.debug('test_write_3d_biff: Set s3')
        s3 = si1.astype(np.float64)
        logging.debug('test_write_3d_biff: s3 {} {} {} {}'.format(type(s3),
            issubclass(type(s3), Series), s3.dtype, s3.shape))
        logging.debug('test_write_3d_biff: s3 {} {} {}'.format(s3.dtype,
            s3.min(), s3.max()))
        logging.debug('test_write_3d_biff: s3.slices {}'.format(s3.slices))
        si3 = Series(s3)
        np.testing.assert_array_almost_equal(si1, si3, decimal=4)
        logging.debug('test_write_3d_biff: si3.slices {}'.format(si3.slices))
        logging.debug('test_write_3d_biff: si3 {} {} {}'.format(type(si3), si3.dtype, si3.shape))
        si3.write('ttb3d/biff', 'Image_%05d.real', formats=['biff'], opts=self.opts)

        s3 = si1-si2
        s3.write('ttb3d/diff', 'Image_%05d.mha', formats=['itk'], opts=self.opts)

class Test4DBiffPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'biff', '--input_shape', '8x30'])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        try:
            shutil.rmtree('ttb3d')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('ttb4d')
        except FileNotFoundError:
            pass

    #@unittest.skip("skipping test_write_4d_biff")
    def test_write_4d_biff(self):
        log = logging.getLogger("TestWritePlugin.test_write_4d_biff")
        log.debug("test_write_4d_biff")
        si1 = Series(
                'tests/dicom/NYRE_151204_T1',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (8, 30, 192, 192))

        si1.sort_on = imagedata.formats.SORT_ON_SLICE
        log.debug("test_write_4d_biff: si1.sort_on {}".format(
            imagedata.formats.sort_on_to_str(si1.sort_on)))
        si1.output_dir = 'single'
        #si1.output_dir = 'multi'
        try:
            shutil.rmtree('ttb4d')
        except FileNotFoundError:
            pass
        si1.write('ttb4d/biff', 'Image_%05d.us', formats=['biff'], opts=self.opts)

        # Read back the DICOM data and verify that the header was modified
        si2 = Series(
                'ttb4d/biff/Image_00000.us',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

if __name__ == '__main__':
    unittest.main()
