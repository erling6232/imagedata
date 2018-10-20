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

class Test3DMatPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #sys.argv[1:] = ['aa', 'bb']
        #self.opts = parser.parse_args(['--order', 'none'])
        #self.opts = parser.parse_args(['--of', 'nifti', '--sort', 'tag'])
        #self.opts = parser.parse_args(['--of', 'nifti'])
        self.opts = parser.parse_args(['--of', 'mat', '--serdes', '1'])
        #self.opts = parser.parse_args([])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    #def tearDown(self):
    #    try:
    #        shutil.rmtree('ttb3d')
    #    except FileNotFoundError:
    #        pass
    #    try:
    #        shutil.rmtree('ttb4d')
    #    except FileNotFoundError:
    #        pass

    #@unittest.skip("skipping test_read_3d_mat")
    def test_write_3d_mat(self):
        log = logging.getLogger("Test3DMatPlugin.test_write_3d_mat")
        log.debug("test_write_3d_mat")
        try:
            shutil.rmtree('ttm3d')
        except FileNotFoundError:
            pass
        try:
            si1 = Series(
                'tests/dicom/NYRE_151204_T1/_fl3d1_0005',
                0,
                self.opts)
        except Exception as e:
            logging.debug('test_write_3d_mat: read si1 exception {}'.format(e))
        si1.shape = si1.shape[1:]
        logging.debug('test_write_3d_mat: si1 {} {} {} {}'.format(type(si1), si1.dtype, si1.min(), si1.max()))
        logging.debug('test_write_3d_mat: si1.shape {}, si1.slices {}'.format(si1.shape, si1.slices))

        logging.debug('test_write_3d_mat: si1.tags {}'.format(si1.tags))
        si1.write('ttm3d/mat', 'Image_%05d', formats=['mat'], opts=self.opts)
        logging.debug('test_write_3d_mat: si1 {} {} {}'.format(si1.dtype, si1.min(), si1.max()))
        logging.debug('test_write_3d_mat: si1.shape {}, si1.slices {}'.format(si1.shape, si1.slices))

        si2 = Series(
                'ttm3d/mat/Image_00000.mat',
                0,
                self.opts)
        logging.debug('test_write_3d_mat: si2 {} {} {}'.format(si2.dtype, si2.min(), si2.max()))

        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

        logging.debug('test_write_3d_mat: Get si1.slices {}'.format(si1.slices))
        logging.debug('test_write_3d_mat: Set s3')
        s3 = si1.astype(np.float64)
        logging.debug('test_write_3d_mat: s3 {} {} {} {}'.format(type(s3),
            issubclass(type(s3), Series), s3.dtype, s3.shape))
        logging.debug('test_write_3d_mat: s3 {} {} {}'.format(s3.dtype,
            s3.min(), s3.max()))
        logging.debug('test_write_3d_mat: s3.slices {}'.format(s3.slices))
        si3 = Series(s3)
        np.testing.assert_array_almost_equal(si1, si3, decimal=4)
        logging.debug('test_write_3d_mat: si3.slices {}'.format(si3.slices))
        logging.debug('test_write_3d_mat: si3 {} {} {}'.format(type(si3), si3.dtype, si3.shape))
        #si3.write('ttm3d/mat', 'Image_%05d.real', formats=['mat'], opts=self.opts)

        s3 = si1-si2
        s3.write('ttm3d/diff', 'Image_%05d.mat', formats=['mat'], opts=self.opts)

class Test4DMatPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #sys.argv[1:] = ['aa', 'bb']
        #self.opts = parser.parse_args(['--order', 'none'])
        #self.opts = parser.parse_args(['--of', 'nifti', '--sort', 'tag'])
        #self.opts = parser.parse_args(['--of', 'nifti'])
        self.opts = parser.parse_args(['--of', 'mat'])
        #self.opts = parser.parse_args([])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    #def tearDown(self):
    #    try:
    #        shutil.rmtree('ttb3d')
    #    except FileNotFoundError:
    #        pass
    #    try:
    #        shutil.rmtree('ttb4d')
    #    except FileNotFoundError:
    #        pass

    #@unittest.skip("skipping test_write_4d_mat")
    def test_write_4d_mat(self):
        log = logging.getLogger("Test4DMatPlugin.test_write_4d_mat")
        log.debug("test_write_4d_mat")
        si1 = Series(
                'tests/dicom/NYRE_151204_T1',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (8, 30, 192, 192))
        #np.testing.assert_array_almost_equal(np.arange(0, 10*2.256, 2.256), hdr.getTimeline(), decimal=2)

        #log.debug("test_write_4d_itk: sliceLocations {}".format(
        #    hdr.sliceLocations))
        #log.debug("test_write_4d_itk: tags {}".format(hdr.tags))
        #log.debug("test_write_4d_itk: spacing {}".format(hdr.spacing))
        #log.debug("test_write_4d_itk: imagePositions {}".format(
        #    hdr.imagePositions))
        #log.debug("test_write_4d_itk: orientation {}".format(hdr.orientation))

        #si1.sort_on = imagedata.formats.SORT_ON_SLICE
        #si1.sort_on = imagedata.formats.SORT_ON_TAG
        #log.debug("test_write_4d_mat: si1.sort_on {}".format(
        #    imagedata.formats.sort_on_to_str(si1.sort_on)))
        #si1.output_dir = 'single'
        #si1.output_dir = 'multi'
        try:
            shutil.rmtree('ttm4d')
        except FileNotFoundError:
            pass
        si1.write('ttm4d/mat', 'Image_%05d', formats=['mat'], opts=self.opts)

        # Read back the MAT data and compare to original si1
        si2 = Series(
                'ttm4d/mat/Image_00000.mat',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    #@unittest.skip("skipping test_read_4d_mat")
    def test_read_4d_mat(self):
        log = logging.getLogger("Test4DMatPlugin.test_read_4d_mat")
        log.debug("test_write_4d_mat")
        si1 = Series(
                'tests/mat/A.mat',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (60,16,160,160))

if __name__ == '__main__':
    unittest.main()
