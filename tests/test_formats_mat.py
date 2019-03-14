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

def list_files(startpath):
    import os
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

class Test3DMatPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'mat', '--serdes', '1'])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        shutil.rmtree('ttm3d', ignore_errors=True)
        shutil.rmtree('ttm4d', ignore_errors=True)

    #@unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            'data/mat/time/Image_00000.mat',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    #@unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            ['data/mat/time/Image_00000.mat',
             'data/mat/time/Image_00000.mat'],
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 10, 40, 192, 152))

    #@unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'data/mat/time',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    #@unittest.skip("skipping test_zipread_single_file")
    def test_zipread_single_file(self):
        si1 = Series(
            'data/mat/time.zip?time/Image_00000.mat',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    #@unittest.skip("skipping test_zipread_single_file_wildcard")
    def test_zipread_single_file_wildcard(self):
        si1 = Series(
            'data/mat/time.zip?.*Image_00000.mat',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    #@unittest.skip("skipping test_zipread_single_file_relative")
    def test_zipread_single_file_relative(self):
        si1 = Series(
            'data/mat/time.zip?Image_00000.mat',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    #@unittest.skip("skipping test_zipread_single_directory")
    def test_zipread_single_directory(self):
        si1 = Series(
            'data/mat/time.zip?time',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    #@unittest.skip("skipping test_zipread_all_files")
    def test_zipread_all_files(self):
        si1 = Series(
            'data/mat/time.zip',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    #@unittest.skip("skipping test_write_single_file")
    def test_write_single_file(self):
        si1 = Series(
            'data/mat/time/Image_00000.mat',
            0,
            self.opts)
        si1.write('ttm4d?Image%05d.mat', formats=['mat'])
        list_files('ttm4d')
        si2 = Series('ttm4d/Image00000.mat')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    #@unittest.skip("skipping test_write_single_directory")
    def test_write_single_directory(self):
        si1 = Series(
            'data/mat/time/',
            0,
            self.opts)
        si1.write('ttm4d?Image%05d.mat', formats=['mat'])
        si2 = Series('ttm4d/')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    #@unittest.skip("skipping test_read_3d_mat_no_opt")
    def test_read_3d_mat_no_opt(self):
        log = logging.getLogger("Test3DMatPlugin.test_read_3d_mat_no_opt")
        log.debug("test_read_3d_mat_no_opt")
        shutil.rmtree('ttm3d', ignore_errors=True)
        si1 = Series(
            'data/mat/time/Image_00000.mat',
            0,
            self.opts)
        si1.write('ttm3d/mat?Image_%05d', formats=['mat'], opts=self.opts)
        si2 = Series('ttm3d/mat/Image_00000.mat')

    #@unittest.skip("skipping test_write_3d_mat_no_opt")
    def test_write_3d_mat_no_opt(self):
        log = logging.getLogger("Test3DMatPlugin.test_write_3d_mat_no_opt")
        log.debug("test_write_3d_mat_no_opt")
        shutil.rmtree('ttm3d', ignore_errors=True)
        si1 = Series(
                'data/mat/time/Image_00000.mat',
                0,
                self.opts)
        si1.write('ttm3d/mat?Image_%05d', formats=['mat'])

    #@unittest.skip("skipping test_read_3d_mat")
    def test_write_3d_mat(self):
        log = logging.getLogger("Test3DMatPlugin.test_write_3d_mat")
        log.debug("test_write_3d_mat")
        shutil.rmtree('ttm3d', ignore_errors=True)
        si1 = Series(
                'data/mat/time/Image_00000.mat',
                0,
                self.opts)
        #si1.shape = si1.shape[1:]
        logging.debug('test_write_3d_mat: si1 {} {} {} {}'.format(type(si1), si1.dtype, si1.min(), si1.max()))
        logging.debug('test_write_3d_mat: si1.shape {}, si1.slices {}'.format(si1.shape, si1.slices))

        logging.debug('test_write_3d_mat: si1.tags {}'.format(si1.tags))
        si1.write('ttm3d/mat?Image_%05d', formats=['mat'], opts=self.opts)
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

        s3 = si1-si2
        s3.write('ttm3d/diff?Image_%05d.mat', formats=['mat'], opts=self.opts)

class Test4DMatPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'mat'])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        shutil.rmtree('ttm3d', ignore_errors=True)
        shutil.rmtree('ttm4d', ignore_errors=True)

    #@unittest.skip("skipping test_write_4d_mat")
    def test_write_4d_mat(self):
        log = logging.getLogger("Test4DMatPlugin.test_write_4d_mat")
        log.debug("test_write_4d_mat")
        si1 = Series(
                'data/mat/time/Image_00000.mat',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))
        shutil.rmtree('ttm4d', ignore_errors=True)
        si1.write('ttm4d/mat?Image_%05d', formats=['mat'], opts=self.opts)

        # Read back the MAT data and compare to original si1
        si2 = Series(
                'ttm4d/mat/Image_00000.mat',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

if __name__ == '__main__':
    unittest.main()
