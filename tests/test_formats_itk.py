#!/usr/bin/env python3

"""Test writing data
"""

import unittest
import os.path
import tempfile
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
        indent = ' ' * 4 * level
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


class TestFileArchiveItk(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['itk']

    # @unittest.skip("skipping test_file_not_found")
    def test_file_not_found(self):
        try:
            _ = Series('file_not_found')
        except FileNotFoundError:
            pass

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            'none',
            self.opts)
        self.assertEqual(si1.input_format, 'itk')
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))

    # @unittest.skip("skipping test_read_2D")
    def test_read_2D(self):
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))
        si2 = si1[0, ...]
        with tempfile.TemporaryDirectory() as d:
            si2.write(os.path.join(d, '?Image.mha'), formats=['itk'])
            si3 = Series(d)
        self.assertEqual(si2.dtype, si3.dtype)
        self.assertEqual(si2.shape, si3.shape)
        np.testing.assert_array_equal(si2, si3)

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
             os.path.join('data', 'itk', 'time', 'Image_00001.mha')],
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 3, 192, 152))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'itk', 'time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_write_3d_single_file")
    def test_write_3d_single_file(self):
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            'none',
            self.opts)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image.mha'), formats=['itk'])
            si2 = Series(d)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_write_4d_single_directory")
    def test_write_4d_single_directory(self):
        si1 = Series(
            os.path.join('data', 'itk', 'time'),
            'none',
            self.opts)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image%05d.mha'), formats=['itk'])
            si2 = Series(d)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_write_4d_single_directory_explicit")
    def test_write_4d_single_directory_explicit(self):
        si1 = Series(
            os.path.join('data', 'itk', 'time'),
            'none',
            self.opts)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image%05d.mha'), formats=['itk'])
            si2 = Series(d)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)


class TestWritePluginITKSlice(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['itk']

    # @unittest.skip("skipping test_write_3d_itk_no_opt")
    def test_write_3d_itk_no_opt(self):
        si1 = Series(os.path.join('data', 'itk', 'time', 'Image_00000.mha'))
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image.mha'), formats=['itk'])
            si2 = Series(d)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_almost_equal(si1, si2, decimal=4)

    # @unittest.skip("skipping test_write_3d_itk")
    def test_write_3d_itk(self):
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))

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

        with tempfile.TemporaryDirectory() as d:
            # Store image with modified header 'hdr'
            si1.write(d, formats=['itk'], opts=self.opts)

            # Read back the ITK data and verify that the header was modified
            si2 = Series(
                d,
                'none',
                self.opts)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)
        compare_headers(self, si1, si2, uid=False)

    # @unittest.skip("skipping test_write_4d_itk")
    def test_write_4d_itk(self):
        log = logging.getLogger("TestWritePlugin.test_write_4d_itk")
        log.debug("test_write_4d_itk")
        si = Series(
            [os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
             os.path.join('data', 'itk', 'time', 'Image_00001.mha'),
             os.path.join('data', 'itk', 'time', 'Image_00002.mha')],
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si.dtype, np.uint16)
        self.assertEqual(si.shape, (3, 3, 192, 152))

        import copy
        deep_si = copy.deepcopy(si)
        np.testing.assert_array_equal(si, deep_si)
        # np.testing.assert_array_almost_equal(np.arange(0, 10*2.256, 2.256), hdr.getTimeline(), decimal=2)

        # log.debug("test_write_4d_itk: sliceLocations {}".format(
        #    hdr.sliceLocations))
        # log.debug("test_write_4d_itk: tags {}".format(hdr.tags))
        # log.debug("test_write_4d_itk: spacing {}".format(hdr.spacing))
        # log.debug("test_write_4d_itk: imagePositions {}".format(
        #    hdr.imagePositions))
        # log.debug("test_write_4d_itk: orientation {}".format(hdr.orientation))

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
        # si.sort_on = imagedata.formats.SORT_ON_TAG
        log.debug("test_write_4d_itk: si.sort_on {}".format(
            imagedata.formats.sort_on_to_str(si.sort_on)))
        si.output_dir = 'single'
        # si.output_dir = 'multi'
        with tempfile.TemporaryDirectory() as d:
            si.write(d, opts=self.opts)
            np.testing.assert_array_equal(si, deep_si)

            # Read back the DICOM data and verify that the header was modified
            si2 = Series(
                d,
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si.shape, si2.shape)
        compare_headers(self, si, si2, uid=False)
        logging.debug('si[0,0,0,0]={}, si2[0,0,0,0]={}'.format(
            si[0, 0, 0, 0], si2[0, 0, 0, 0]))
        logging.debug('si.dtype {}, si2.dtype {}'.format(
            si.dtype, si2.dtype))
        np.testing.assert_array_almost_equal(si, si2, decimal=4)


class TestWritePluginItkTag(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)
        self.opts = parser.parse_args(['--of', 'itk', '--sort', 'tag'])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['itk']

    @unittest.skip("skipping test_write_3d_itk")
    def test_write_3d_itk(self):
        log = logging.getLogger("TestWritePluginITK.test_write_3d_itk")
        log.debug("test_write_3d_itk")
        si = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            'none',
            self.opts)
        self.assertEqual(si.dtype, np.uint16)
        self.assertEqual(si.shape, (40, 192, 152))

        # log.debug("test_write_3d_itk: sliceLocations {}".format(
        #    si.sliceLocations))
        log.debug("test_write_3d_itk: tags {}".format(si.tags))
        log.debug("test_write_3d_itk: spacing {}".format(si.spacing))
        log.debug("test_write_3d_itk: imagePositions {}".format(
            si.imagePositions))
        log.debug("test_write_3d_itk: orientation {}".format(si.orientation))

        # Modify header
        si.sliceLocations = [i for i in range(40)]
        si.spacing = (3, 2, 1)
        for slice in range(len(si.sliceLocations)):
            si.imagePositions = {
                slice:
                    np.array([slice, 1, 0])
            }
        for slice in range(si.shape[1]):
            si.imagePositions = {
                slice: si.getPositionForVoxel(np.array([slice, 0, 0]))
            }

        # Store image with modified header
        with tempfile.TemporaryDirectory() as d:
            si.write(d, self.opts)

            # Read back the ITK data and verify that the header was modified
            si2 = Series(
                d,
                'none',
                self.opts)
        self.assertEqual(si.shape, si2.shape)
        np.testing.assert_array_equal(si, si2)
        # np.testing.assert_array_equal(hdr.sliceLocations, hdr2.sliceLocations)
        compare_headers(self, si, si2)

    # @unittest.skip("skipping test_write_4d_itk")
    def test_write_4d_itk(self):
        log = logging.getLogger("TestWritePlugin.test_write_4d_itk")
        log.debug("test_write_4d_itk")
        si = Series(
            [os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
             os.path.join('data', 'itk', 'time', 'Image_00001.mha'),
             os.path.join('data', 'itk', 'time', 'Image_00002.mha')],
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si.dtype, np.uint16)
        self.assertEqual(si.shape, (3, 3, 192, 152))
        # np.testing.assert_array_almost_equal(np.arange(0, 10*2.256, 2.256), hdr.getTimeline(), decimal=2)

        # log.debug("test_write_4d_itk: sliceLocations", hdr.sliceLocations)
        # log.debug("test_write_4d_itk: tags {}".format(hdr.tags))
        # log.debug("test_write_4d_itk: spacing", hdr.spacing)
        # log.debug("test_write_4d_itk: imagePositions", hdr.imagePositions)
        # log.debug("test_write_4d_itk: orientation", hdr.orientation)

        # Modify header
        si.sliceLocations = (1, 2, 3)
        si.spacing = (3, 2, 1)
        for slice in range(si.shape[1]):
            si.imagePositions = {
                slice:
                    np.array([slice, 1, 0])
            }
        si.orientation = np.array([0, 0, 1, 0, 1, 0])
        for slice in range(si.shape[1]):
            si.imagePositions = {
                slice: si.getPositionForVoxel(np.array([slice, 0, 0])).reshape((3, 1))
            }

        si.sort_on = imagedata.formats.SORT_ON_TAG
        log.debug("test_write_4d_itk: hdr.sort_on {}".format(
            imagedata.formats.sort_on_to_str(si.sort_on)))
        si.output_dir = 'single'
        with tempfile.TemporaryDirectory() as d:
            si.write(d, self.opts)

            # Read back the ITK data and verify that the header was modified
            si2 = Series(
                d,
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual((3, 3, 192, 152), si2.shape)
        # np.testing.assert_array_equal(si, si2)
        # np.testing.assert_array_equal(hdr.sliceLocations, hdr2.sliceLocations)
        # log.debug("hdr2.tags.keys(): {}".format(hdr2.tags.keys()))
        tags = {}
        for slice in range(3):
            tags[slice] = np.arange(3)
        self.assertEqual(tags.keys(), si2.tags.keys())
        for k in si2.tags.keys():
            np.testing.assert_array_equal(tags[k], si2.tags[k])
        np.testing.assert_array_equal(si.spacing, si2.spacing)
        image_positions = {}
        for slice in range(3):
            image_positions[slice] = np.array([0, 1 + 3 * slice, 0])
        self.assertEqual(image_positions.keys(),
                         si2.imagePositions.keys())
        np.testing.assert_array_equal(si.orientation, si2.orientation)


if __name__ == '__main__':
    unittest.main()
