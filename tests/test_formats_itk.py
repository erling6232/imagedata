#!/usr/bin/env python3

"""Test writing data
"""

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

class TestWritePluginITK_slice(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #self.opts = parser.parse_args(['--template', 'tests/dicom/time/',
        #        '--geometry', 'tests/dicom/time'])
        #self.opts = parser.parse_args(['--of', 'itk', '--sort', 'tag'])
        #self.opts = parser.parse_args(['--of', 'itk'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['itk']

        #self.hdr,self.si = imagedata.readdata.read(
        #        ('tests/dicom/time/',),
        #        imagedata.formats.INPUT_ORDER_TIME,
        #        self.opts)

    def tearDown(self):
        try:
            shutil.rmtree('tti3')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('tti4')
        except FileNotFoundError:
            pass

    @unittest.skip("skipping test_read_3d_itk")
    def test_read_3d_itk(self):
        si1 = Series(
            #'tests/dicom/NYRE_151204_T1/_fl3d1_0005',
            'tests/itk/time/Image_00000.mhd',
            0,
            self.opts)
        #self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (30, 142, 115))

    @unittest.skip("skipping test_read_3d_itk_no_opt")
    def test_read_3d_itk_no_opt(self):
        si1 = Series(
            #'tests/dicom/NYRE_151204_T1/_fl3d1_0005',
            'tests/itk/time/Image_00000.mhd')
        #self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (30, 142, 115))

    @unittest.skip("skipping test_write_3d_itk_no_opt")
    def test_write_3d_itk_no_opt(self):
        si1 = Series(
            #'tests/dicom/NYRE_151204_T1/_fl3d1_0005',
            'tests/itk/time/Image_00000.mhd')
        #self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (30, 142, 115))
        si1.write('tti3', 'Image.mhd', formats=['itk'])

    @unittest.skip("skipping test_write_3d_itk")
    def test_write_3d_itk(self):
        #si1 = Series(
        #        'tests/dicom/NYRE_151204_T1',
        #        imagedata.formats.INPUT_ORDER_TIME,
        #        self.opts)
        si1 = Series(
            #'tests/dicom/NYRE_151204_T1/_fl3d1_0005',
            'tests/itk/time/Image_00000.mhd',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (30, 142, 115))

        #logging.debug("test_write_3d_itk: sliceLocations {}".format(
        #    hdr.sliceLocations))
        logging.debug("test_write_3d_itk: tags {}".format(si1.tags))
        logging.debug("test_write_3d_itk: spacing {}".format(si1.spacing))
        logging.debug("test_write_3d_itk: imagePositions) {}".format(
            si1.imagePositions))
        logging.debug("test_write_3d_itk: orientation {}".format(si1.orientation))

        """
        # Modify header
        si1.sliceLocations = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30)
        #for slice in range(len(hdr.sliceLocations)):
        #   hdr.tags=
        #si1.spacing = (3, 1, 1)
        for slice in range(len(si1.sliceLocations)):
            si1.imagePositions = {
                slice:
                        np.array([slice,1,0])
            }
        #si1.orientation=np.array([1, 0, 0, 0, 0, -1])
        for slice in range(si1.slices):
            si1.imagePositions = {
                    slice: si1.getPositionForVoxel(np.array([slice,0,0]))
            }
        si1.seriesNumber=1001
        si1.seriesDescription="Test 1"
        si1.imageType = ['AB', 'CD', 'EF']
        """

        # Store image with modified header 'hdr'
        si1.write('tti3', 'Image.mhd', formats=['itk'], opts=self.opts)

        # Read back the ITK data and verify that the header was modified
        si2 = Series(
                'tti3/Image.mhd',
                0,
                self.opts)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)
        compare_headers(self, si1, si2)

    @unittest.skip("skipping test_write_4d_itk")
    def test_write_4d_itk(self):
        log = logging.getLogger("TestWritePlugin.test_write_4d_itk")
        log.debug("test_write_4d_itk")
        si = Series(
                #('tests/dicom/time/',),
                ['tests/itk/time/Image_00000.mhd',
                 'tests/itk/time/Image_00001.mhd',
                 'tests/itk/time/Image_00002.mhd',
                 'tests/itk/time/Image_00003.mhd',
                 'tests/itk/time/Image_00004.mhd',
                 'tests/itk/time/Image_00005.mhd',
                 'tests/itk/time/Image_00006.mhd',
                 'tests/itk/time/Image_00007.mhd',
                 'tests/itk/time/Image_00008.mhd',
                 'tests/itk/time/Image_00009.mhd'],
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si.dtype, np.float32)
        self.assertEqual(si.shape, (10,30,142,115))

        import copy
        deep_si = copy.deepcopy(si)
        np.testing.assert_array_equal(si, deep_si)
        #np.testing.assert_array_almost_equal(np.arange(0, 10*2.256, 2.256), hdr.getTimeline(), decimal=2)

        #log.debug("test_write_4d_itk: sliceLocations {}".format(
        #    hdr.sliceLocations))
        #log.debug("test_write_4d_itk: tags {}".format(hdr.tags))
        #log.debug("test_write_4d_itk: spacing {}".format(hdr.spacing))
        #log.debug("test_write_4d_itk: imagePositions {}".format(
        #    hdr.imagePositions))
        #log.debug("test_write_4d_itk: orientation {}".format(hdr.orientation))

        # Modify header
        """
        si.sliceLocations = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
        #slices = len(hdr.sliceLocations)
        #hdr.sliceLocations = np.arange(1, slices*1+0.5, 1).tolist()
        #for slice in range(slices):
        #   hdr.tags=
        hdr.spacing = (3, 2, 1)
        for slice in range(si.shape[1]):
            hdr.imagePositions = {
                slice:
                        np.array([slice,1,0])
            }
        hdr.orientation=np.array([1, 0, 0, 0, 0, -1])
        for slice in range(si.shape[1]):
            hdr.imagePositions = {
                slice: hdr.getPositionForVoxel(np.array([slice,0,0]))
            }
        hdr.seriesNumber=1001
        hdr.seriesDescription="Test 1"
        hdr.imageType = ['AB', 'CD', 'EF']
        """

        si.sort_on = imagedata.formats.SORT_ON_SLICE
        #si.sort_on = imagedata.formats.SORT_ON_TAG
        log.debug("test_write_4d_itk: si.sort_on {}".format(
            imagedata.formats.sort_on_to_str(si.sort_on)))
        si.output_dir = 'single'
        #si.output_dir = 'multi'
        si.write('tti4', 'Image_%05d.mha', opts=self.opts)
        np.testing.assert_array_equal(si, deep_si)

        # Read back the DICOM data and verify that the header was modified
        si2 = Series(
                'tti4/',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si.shape, si2.shape)
        compare_headers(self, si, si2)
        logging.debug('si[0,0,0,0]={}, si2[0,0,0,0]={}'.format(
            si[0,0,0,0], si2[0,0,0,0]))
        logging.debug('si.dtype {}, si2.dtype {}'.format(
            si.dtype, si2.dtype))
        np.testing.assert_array_almost_equal(si, si2, decimal=2)

class TestWritePluginITK_tag(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)
        #self.opts = parser.parse_args(['--template', 'tests/dicom/time/',
        #        '--geometry', 'tests/dicom/time'])
        self.opts = parser.parse_args(['--of', 'mhd', '--sort', 'tag'])
        #self.opts = parser.parse_args(['--of', 'mhd'])
        #self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['itk']

        #self.hdr,self.si = imagedata.readdata.read(
        #        ('tests/dicom/time/',),
        #        imagedata.formats.INPUT_ORDER_TIME,
        #        self.opts)
    def tearDown(self):
        try:
            shutil.rmtree('tti3')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('tti4')
        except FileNotFoundError:
            pass

    @unittest.skip("skipping test_write_3d_itk")
    def test_write_3d_itk(self):
        log = logging.getLogger("TestWritePluginITK.test_write_3d_itk")
        log.debug("test_write_3d_itk")
        hdr,si = imagedata.readdata.read(
                'tests/itk/time/Image_00000.mhd',
                0,
                self.opts)
        self.assertEqual(si.dtype, np.float64)
        #self.assertEqual(si.shape, (1,30,192,192))
        self.assertEqual(si.shape, (1,30, 142, 115))

        #log.debug("test_write_3d_itk: sliceLocations {}".format(
        #    hdr.sliceLocations))
        log.debug("test_write_3d_itk: tags {}".format(hdr.tags))
        log.debug("test_write_3d_itk: spacing {}".format(hdr.spacing))
        log.debug("test_write_3d_itk: imagePositions {}".format(
            hdr.imagePositions))
        log.debug("test_write_3d_itk: orientation {}".format(hdr.orientation))

        # Modify header
        hdr.sliceLocations = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
        #for slice in range(len(hdr.sliceLocations)):
        #   hdr.tags=
        hdr.spacing = (3, 2, 1)
        for slice in range(len(hdr.sliceLocations)):
            hdr.imagePositions = {
                slice:
                        np.array([slice,1,0])
            }
        hdr.orientation=np.array([1, 0, 0, 0, 0, -1])
        for slice in range(si.shape[1]):
            hdr.imagePositions = {
                    slice: hdr.getPositionForVoxel(np.array([slice,0,0]))
            }
        hdr.seriesNumber=1001
        hdr.seriesDescription="Test 1"
        hdr.imageType = ['AB', 'CD', 'EF']

        # Store image with modified header 'hdr'
        try:
            shutil.rmtree('tt')
        except FileNotFoundError:
            pass
        hdr.write_3d_numpy(si, 'tt', 'Image.mhd', self.opts)

        # Read back the ITK data and verify that the header was modified
        hdr2,si2 = imagedata.readdata.read(
                ('tt/Image.mhd',),
                0,
                self.opts)
        self.assertEqual(si.shape, si2.shape)
        np.testing.assert_array_equal(si, si2)
        #np.testing.assert_array_equal(hdr.sliceLocations, hdr2.sliceLocations)
        self.assertEqual(hdr.tags.keys(), hdr2.tags.keys())
        for k in hdr.tags.keys():
            np.testing.assert_array_equal(hdr.tags[k], hdr2.tags[k])
        np.testing.assert_array_equal(hdr.spacing, hdr2.spacing)
        self.assertEqual(hdr.imagePositions.keys(),
                hdr2.imagePositions.keys())
        for k in hdr.imagePositions.keys():
            np.testing.assert_array_equal(hdr.imagePositions[k],
                hdr2.imagePositions[k])
        np.testing.assert_array_equal(hdr.orientation, hdr2.orientation)

    @unittest.skip("skipping test_write_4d_itk")
    def test_write_4d_itk(self):
        log = logging.getLogger("TestWritePlugin.test_write_4d_itk")
        log.debug("test_write_4d_itk")
        hdr,si = imagedata.readdata.read(
                #('tests/dicom/time/',),
                ('tests/mhd/time/Image_00000.mhd',
                'tests/mhd/time/Image_00001.mhd',
                'tests/mhd/time/Image_00002.mhd',
                'tests/mhd/time/Image_00003.mhd',
                'tests/mhd/time/Image_00004.mhd',
                'tests/mhd/time/Image_00005.mhd',
                'tests/mhd/time/Image_00006.mhd',
                'tests/mhd/time/Image_00007.mhd',
                'tests/mhd/time/Image_00008.mhd',
                'tests/mhd/time/Image_00009.mhd'
                ),
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si.dtype, np.float64)
        self.assertEqual(si.shape, (10,30,142,115))
        #np.testing.assert_array_almost_equal(np.arange(0, 10*2.256, 2.256), hdr.getTimeline(), decimal=2)

        #log.debug("test_write_4d_itk: sliceLocations", hdr.sliceLocations)
        #log.debug("test_write_4d_itk: tags {}".format(hdr.tags))
        #log.debug("test_write_4d_itk: spacing", hdr.spacing)
        #log.debug("test_write_4d_itk: imagePositions", hdr.imagePositions)
        #log.debug("test_write_4d_itk: orientation", hdr.orientation)

        # Modify header
        hdr.sliceLocations = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16)
        #slices = len(hdr.sliceLocations)
        #hdr.sliceLocations = np.arange(1, slices*1+0.5, 1).tolist()
        #for slice in range(slices):
        #   hdr.tags=
        hdr.spacing = (3, 2, 1)
        for slice in range(si.shape[1]):
            hdr.imagePositions = {
                slice:
                        np.array([slice,1,0])
            }
        hdr.orientation=np.array([1, 0, 0, 0, 0, -1])
        for slice in range(si.shape[1]):
            hdr.imagePositions = {
                slice: hdr.getPositionForVoxel(np.array([slice,0,0]))
            }
        hdr.seriesNumber=1001
        hdr.seriesDescription="Test 1"
        hdr.imageType = ['AB', 'CD', 'EF']

        #hdr.sort_on = imagedata.formats.SORT_ON_SLICE
        hdr.sort_on = imagedata.formats.SORT_ON_TAG
        log.debug("test_write_4d_itk: hdr.sort_on {}".format(
            imagedata.formats.sort_on_to_str(hdr.sort_on)))
        hdr.output_dir = 'single'
        #hdr.output_dir = 'multi'
        try:
            shutil.rmtree('tt4ds')
        except FileNotFoundError:
            pass
        hdr.write_4d_numpy(si, 'tt4ds', 'Image_%05d.mhd', self.opts)

        # Read back the ITK data and verify that the header was modified
        hdr2,si2 = imagedata.readdata.read(
                #('tt4d/',),
                ('tt4ds/Image_00000.mhd',
                 'tt4ds/Image_00001.mhd',
                 'tt4ds/Image_00002.mhd',
                 'tt4ds/Image_00003.mhd',
                 'tt4ds/Image_00004.mhd',
                 'tt4ds/Image_00005.mhd',
                 'tt4ds/Image_00006.mhd',
                 'tt4ds/Image_00007.mhd',
                 'tt4ds/Image_00008.mhd',
                 'tt4ds/Image_00009.mhd',
                 'tt4ds/Image_00010.mhd',
                 'tt4ds/Image_00011.mhd',
                 'tt4ds/Image_00012.mhd',
                 'tt4ds/Image_00013.mhd',
                 'tt4ds/Image_00014.mhd',
                 'tt4ds/Image_00015.mhd',
                 'tt4ds/Image_00016.mhd',
                 'tt4ds/Image_00017.mhd',
                 'tt4ds/Image_00018.mhd',
                 'tt4ds/Image_00019.mhd',
                 'tt4ds/Image_00020.mhd',
                 'tt4ds/Image_00021.mhd',
                 'tt4ds/Image_00022.mhd',
                 'tt4ds/Image_00023.mhd',
                 'tt4ds/Image_00024.mhd',
                 'tt4ds/Image_00025.mhd',
                 'tt4ds/Image_00026.mhd',
                 'tt4ds/Image_00027.mhd',
                 'tt4ds/Image_00028.mhd',
                 'tt4ds/Image_00029.mhd'
                ),
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual((30, 10, 142, 115), si2.shape)
        #np.testing.assert_array_equal(si, si2)
        #np.testing.assert_array_equal(hdr.sliceLocations, hdr2.sliceLocations)
        #log.debug("hdr2.tags.keys(): {}".format(hdr2.tags.keys()))
        tags = {}
        for slice in range(10):
            tags[slice] = np.arange(30)
        self.assertEqual(tags.keys(), hdr2.tags.keys())
        for k in hdr2.tags.keys():
            np.testing.assert_array_equal(tags[k], hdr2.tags[k])
        np.testing.assert_array_equal(hdr.spacing, hdr2.spacing)
        imagePositions = {}
        for slice in range(10):
            imagePositions[slice] = np.array([0,1+3*slice,0])
        self.assertEqual(imagePositions.keys(),
                hdr2.imagePositions.keys())
        for k in imagePositions.keys():
            log.debug("hdr2.imagePositions: {} {}".format(
                k, hdr2.imagePositions[k]))
            np.testing.assert_array_equal(imagePositions[k],
                hdr2.imagePositions[k])
        np.testing.assert_array_equal(hdr.orientation, hdr2.orientation)
        #self.assertEqual(hdr.seriesNumber, hdr2.seriesNumber)
        #self.assertEqual(hdr.seriesDescription, hdr2.seriesDescription)
        #self.assertEqual(hdr.imageType, hdr2.imageType)

if __name__ == '__main__':
    unittest.main()
