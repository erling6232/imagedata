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

class TestDicomPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #sys.argv[1:] = ['aa', 'bb']
        #self.opts = parser.parse_args(['--order', 'none'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['dicom']

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        shutil.rmtree('ttd3d', ignore_errors=True)
        shutil.rmtree('ttd4d', ignore_errors=True)

    #@unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            'data/dicom/time/time00/Image_00000.dcm',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    #@unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [
            'data/dicom/time/time00/Image_00000.dcm',
            'data/dicom/time/time00/Image_00001.dcm'
            ],
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 192, 152))

    #@unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'data/dicom/time/time00',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (40, 192, 152))

    #@unittest.skip("skipping test_read_dicom_3D_no_opt")
    def test_read_dicom_3D_no_opt(self):
        d=Series(
                'data/dicom/time/time00/Image_00000.dcm')
        #logging.debug("test_series_dicom: type(d) {}".format(type(d)))
        #logging.debug("Series.imageType {}".format(d.imageType))
        #logging.debug("Series.dir() {}".format(dir(d)))
        self.assertEqual(d.dtype, np.uint16)
        self.assertEqual(d.shape, (192, 152))

    #@unittest.skip("skipping test_read_dicom_4D")
    def test_read_dicom_4D(self):
        si1 = Series(
                'data/dicom/time',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))
        t = np.array([  0.  ,   2.99,   5.97,   8.96,  11.95,  14.94,  17.93,  20.92, 23.9 ,  26.89])
        np.testing.assert_array_almost_equal(t, si1.timeline, decimal=2)
        #for axis in si1.axes:
        #    logging.debug('test_read_dicom_4D: axis {}'.format(axis))

    #@unittest.skip("skipping test_copy_dicom_4D")
    def test_copy_dicom_4D(self):
        si = Series(
                'data/dicom/time',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        #si.sort_on = imagedata.formats.SORT_ON_SLICE
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        newsi  = Series(si,
                imagedata.formats.INPUT_ORDER_TIME)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (10, 40, 192, 152))

    #@unittest.skip("skipping test_write_single_file")
    def test_write_single_file(self):
        si1 = Series(
            'data/dicom/time/time00/Image_00000.dcm'
            )
        si1.write('ttd3d?Image.dcm', formats=['dicom'])
        si2 = Series('ttd3d/')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    #@unittest.skip("skipping test_write_single_directory")
    def test_write_single_directory(self):
        si1 = Series('data/dicom/time/time00/')
        si1.write('ttd3d?Image%05d.dcm', formats=['dicom'])
        si2 = Series('ttd3d/')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    #@unittest.skip("skipping test_write_dicom_4D")
    def test_write_dicom_4D(self):
        si = Series(
                'data/dicom/time_all/',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        si.write('ttd4d/Image_%05d', formats=['dicom'], opts=self.opts)
        newsi  = Series('ttd4d',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (10, 40, 192, 152))

    #@unittest.skip("skipping test_write_dicom_4D_no_opt")
    def test_write_dicom_4D_no_opt(self):
        si = Series(
                'data/dicom/time_all/',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        si.write('ttd4d/Image_%05d', formats=['dicom'])
        newsi  = Series('ttd4d',
                imagedata.formats.INPUT_ORDER_TIME)
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (10, 40, 192, 152))

class TestDicomZipPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #sys.argv[1:] = ['aa', 'bb']
        #self.opts = parser.parse_args(['--order', 'none'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['dicom']

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        #shutil.rmtree('ttd3z', ignore_errors=True)
        shutil.rmtree('ttd4z', ignore_errors=True)

    #@unittest.skip("skipping test_write_zip")
    def test_write_zip(self):
        si1 = Series('data/dicom/time/time00/')
        si1.write('ttd3z/dicom.zip?Image_%05d.dcm', formats=['dicom'])
        si2 = Series('ttd3z/dicom.zip')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

class test_zip_archive_dicom(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #self.opts = parser.parse_args(['--template', 'tests/dicom/time/',
        #        '--geometry', 'tests/dicom/time'])
        #self.opts = parser.parse_args(['--of', 'itk', '--sort', 'tag'])
        #self.opts = parser.parse_args(['--of', 'itk'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['dicom']

        #self.hdr,self.si = imagedata.readdata.read(
        #        ('tests/dicom/time/',),
        #        imagedata.formats.INPUT_ORDER_TIME,
        #        self.opts)

    def tearDown(self):
        shutil.rmtree('ttd3', ignore_errors=True)
        shutil.rmtree('ttd4', ignore_errors=True)

    #@unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            'data/dicom/time.zip?time/time00/Image_00000.dcm',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    #@unittest.skip("skipping test_read_single_file_relative")
    def test_read_single_file_relative(self):
        si1 = Series(
            'data/dicom/time.zip?time00/Image_00000.dcm',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    #@unittest.skip("skipping test_read_single_file_wildcard")
    def test_read_single_file_wildcard(self):
        si1 = Series(
            'data/dicom/time.zip?.*time00/Image_00000.dcm',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    #@unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            'data/dicom/time.zip?time/time00/Image_0000[01].dcm',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 192, 152))

    #@unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'data/dicom/time.zip?time/time00/',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (40, 192, 152))

    #@unittest.skip("skipping test_read_two_directories")
    def test_read_two_directories(self):
        si1 = Series(
            'data/dicom/time.zip?time/time0[02]/',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 40, 192, 152))

    #@unittest.skip("skipping test_read_all_files")
    def test_read_all_files(self):
        si1 = Series(
            'data/dicom/time.zip?time/',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

class write_test_zip_archive_dicom(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['dicom']

    def tearDown(self):
        shutil.rmtree('ttd3z', ignore_errors=True)
        shutil.rmtree('ttd4z', ignore_errors=True)

    #@unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series('data/dicom/time.zip?time/time00/Image_00000.dcm')
        si1.write('ttd3z/dicom.zip', formats=['dicom'])
        si2 = Series('ttd3z/dicom.zip?Image_00000.dcm')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    @unittest.skip("skipping test_read_single_file_relative")
    def test_read_single_file_relative(self):
        si1 = Series(
            'data/dicom/time.zip?time00/Image_00000.dcm',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    @unittest.skip("skipping test_read_single_file_wildcard")
    def test_read_single_file_wildcard(self):
        si1 = Series(
            'data/dicom/time.zip?.*time00/Image_00000.dcm',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            'data/dicom/time.zip?time/time00/Image_0000[01].dcm',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 192, 152))

    @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'data/dicom/time.zip?time/time00/',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (40, 192, 152))

    @unittest.skip("skipping test_read_two_directories")
    def test_read_two_directories(self):
        si1 = Series(
            'data/dicom/time.zip?time/time0[02]/',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 40, 192, 152))

    @unittest.skip("skipping test_read_all_files")
    def test_read_all_files(self):
        si1 = Series(
            'data/dicom/time.zip?time/',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

class TestDicomSlicing(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['dicom']

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        shutil.rmtree('ttd3s', ignore_errors=True)
        shutil.rmtree('ttd4s', ignore_errors=True)

    #@unittest.skip("skipping test_slice_inplane")
    def test_slice_inplane(self):
        si1 = Series(
            'data/dicom/time/time00/Image_00000.dcm',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))
        a2 = np.array(si1)[80:120,40:60]
        si2 = si1[80:120,40:60]
        np.testing.assert_array_equal(a2, si2)

    #@unittest.skip("skipping test_slice_z")
    def test_slice_z(self):
        si1 = Series(
            'data/dicom/time/time00/',
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (40, 192, 152))
        a2 = np.array(si1)[1:3]
        si2 = si1[1:3]
        np.testing.assert_array_equal(a2, si2)
        self.assertEqual(len(si2.imagePositions), 2)
        np.testing.assert_array_equal(si2.imagePositions[0], si1.imagePositions[1])
        np.testing.assert_array_equal(si2.imagePositions[1], si1.imagePositions[2])

    #@unittest.skip("skipping test_slice_time_z")
    def test_slice_time_z(self):
        si1 = Series(
            'data/dicom/time/',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))
        a2 = np.array(si1)[1:3,1:4]
        si2 = si1[1:3,1:4]
        np.testing.assert_array_equal(a2, si2)
        self.assertEqual(len(si2.imagePositions), 3)
        np.testing.assert_array_equal(si2.imagePositions[0], si1.imagePositions[1])
        np.testing.assert_array_equal(si2.imagePositions[1], si1.imagePositions[2])
        np.testing.assert_array_equal(si2.imagePositions[2], si1.imagePositions[3])
        self.assertEqual(len(si2.tags[0]), 2)

if __name__ == '__main__':
    unittest.main()
