#!/usr/bin/env python3

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


class TestDicomPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.input_format, 'dicom')
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [
                os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
                os.path.join('data', 'dicom', 'time', 'time00', 'Image_00021.dcm')
            ],
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 192, 152))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))

    # @unittest.skip("skipping test_read_dicom_3D_no_opt")
    def test_read_dicom_3D_no_opt(self):
        d = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'))
        self.assertEqual(d.dtype, np.uint16)
        self.assertEqual(d.shape, (192, 152))

    # @unittest.skip("skipping test_read_dicom_4D")
    def test_read_dicom_4D(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        t = np.array([0., 2.99, 5.97])
        np.testing.assert_array_almost_equal(t, si1.timeline, decimal=2)
        # for axis in si1.axes:
        #    logging.debug('test_read_dicom_4D: axis {}'.format(axis))

    # @unittest.skip("skipping test_read_dicom_user_defined_TI")
    def test_read_dicom_user_defined_TI(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'TI'),
            input_order='ti',
            opts={'ti': 'InversionTime'})
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (5, 1, 384, 384))
        with tempfile.TemporaryDirectory() as d:
            si1.write(d,
                      formats=['dicom'],
                      opts={'ti': 'InversionTime'})
            si2 = Series(d,
                         input_order='ti',
                         opts={'ti': 'InversionTime'})
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    # @unittest.skip("skipping test_copy_dicom_4D")
    def test_copy_dicom_4D(self):
        si = Series(
            os.path.join('data', 'dicom', 'time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        newsi = Series(si,
                       imagedata.formats.INPUT_ORDER_TIME)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_write_single_file")
    def test_write_single_file(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm')
        )
        with tempfile.TemporaryDirectory() as d:
            si1.write('{}?Image.dcm'.format(d), formats=['dicom'])
            si2 = Series(d)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_write_single_directory")
    def test_write_single_directory(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        with tempfile.TemporaryDirectory() as d:
            si1.write('{}?Image%05d.dcm'.format(d), formats=['dicom'])
            si2 = Series(d)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_write_dicom_4D")
    def test_write_dicom_4D(self):
        si = Series(
            os.path.join('data', 'dicom', 'time_all'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        with tempfile.TemporaryDirectory() as d:
            si.write(os.path.join(d, 'Image_%05d'),
                     formats=['dicom'], opts=self.opts)
            newsi = Series(d,
                           imagedata.formats.INPUT_ORDER_TIME,
                           self.opts)
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_write_dicom_4D_no_opt")
    def test_write_dicom_4D_no_opt(self):
        si = Series(
            os.path.join('data', 'dicom', 'time_all'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        with tempfile.TemporaryDirectory() as d:
            si.write(os.path.join(d, 'Image_%05d'),
                     formats=['dicom'])
            newsi = Series(d,
                           imagedata.formats.INPUT_ORDER_TIME)
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (3, 3, 192, 152))


class TestDicomZipPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

    # @unittest.skip("skipping test_write_zip")
    def test_write_zip(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'dicom.zip?Image_%05d.dcm'),
                      formats=['dicom'])
            si2 = Series(os.path.join(d, 'dicom.zip'))
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)


class TestZipArchiveDicom(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_single_file_relative")
    def test_read_single_file_relative(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_single_file_wildcard")
    def test_read_single_file_wildcard(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?*time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_0002[01].dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 192, 152))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))

    # @unittest.skip("skipping test_read_two_directories")
    def test_read_two_directories(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time0[02]/'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 3, 192, 152))

    # @unittest.skip("skipping test_read_all_files")
    def test_read_all_files(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))


class TestWriteZipArchiveDicom(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(os.path.join('data', 'dicom', 'time.zip?time/time00/Image_00020.dcm'))
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'dicom.zip'), formats=['dicom'])
            si2 = Series(os.path.join(d, 'dicom.zip?Image_00000.dcm'))
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_read_single_file_relative")
    def test_read_single_file_relative(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_single_file_wildcard")
    def test_read_single_file_wildcard(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?*time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_0002[01].dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 192, 152))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))

    # @unittest.skip("skipping test_read_two_directories")
    def test_read_two_directories(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time0[02]/'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 3, 192, 152))

    # @unittest.skip("skipping test_read_all_files")
    def test_read_all_files(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))


class TestDicomSlicing(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

    # @unittest.skip("skipping test_slice_inplane")
    def test_slice_inplane(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))
        a2 = np.array(si1)[80:120, 40:60]
        si2 = si1[80:120, 40:60]
        np.testing.assert_array_equal(a2, si2)

    # @unittest.skip("skipping test_slice_z")
    def test_slice_z(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))
        a2 = np.array(si1)[1:3]
        si2 = si1[1:3]
        np.testing.assert_array_equal(a2, si2)
        self.assertEqual(len(si2.imagePositions), 2)
        np.testing.assert_array_equal(si2.imagePositions[0], si1.imagePositions[1])
        np.testing.assert_array_equal(si2.imagePositions[1], si1.imagePositions[2])

    # @unittest.skip("skipping test_slice_time_z")
    def test_slice_time_z(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        a2 = np.array(si1)[1:3, 1:3]
        si2 = si1[1:3, 1:3]
        np.testing.assert_array_equal(a2, si2)
        self.assertEqual(len(si2.imagePositions), 2)
        np.testing.assert_array_equal(si2.imagePositions[0], si1.imagePositions[1])
        np.testing.assert_array_equal(si2.imagePositions[1], si1.imagePositions[2])
        self.assertEqual(len(si2.tags[0]), 2)

    # @unittest.skip("skipping test_slice_ellipsis_first")
    def test_slice_ellipsis_first(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        a2 = np.array(si1)[..., 10:40]
        si2 = si1[..., 10:40]
        np.testing.assert_array_equal(a2, si2)
        self.assertEqual(len(si2.imagePositions), len(si1.imagePositions))
        for i in range(len(si2.imagePositions)):
            np.testing.assert_array_equal(si2.imagePositions[i], si1.imagePositions[i])
        self.assertEqual(len(si2.tags[0]), len(si1.tags[0]))
        np.testing.assert_array_equal(si2.tags[0], si1.tags[0])

    # @unittest.skip("skipping test_slice_ellipsis_last")
    def test_slice_ellipsis_last(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        a2 = np.array(si1)[1:3, ...]
        si2 = si1[1:3, ...]
        np.testing.assert_array_equal(a2, si2)
        self.assertEqual(len(si2.imagePositions), len(si1.imagePositions))
        for i in range(len(si2.imagePositions)):
            np.testing.assert_array_equal(si2.imagePositions[i], si1.imagePositions[i])
        self.assertEqual(len(si2.tags[0]), 2)
        np.testing.assert_array_equal(si2.tags[0], si1.tags[0][1:3])

    # @unittest.skip("skipping test_slice_ellipsis_middle")
    def test_slice_ellipsis_middle(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        a2 = np.array(si1)[1:3, ..., 10:40]
        si2 = si1[1:3, ..., 10:40]
        np.testing.assert_array_equal(a2, si2)
        self.assertEqual(len(si2.imagePositions), len(si1.imagePositions))
        for i in range(len(si2.imagePositions)):
            np.testing.assert_array_equal(si2.imagePositions[i], si1.imagePositions[i])
        self.assertEqual(len(si2.tags[0]), 2)
        np.testing.assert_array_equal(si2.tags[0], si1.tags[0][1:3])


if __name__ == '__main__':
    unittest.main()
