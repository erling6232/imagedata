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


class Test3DNIfTIPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'nifti', '--serdes', '1'])

        plugins = imagedata.formats.get_plugins_list()
        self.nifti_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'nifti':
                self.nifti_plugin = pclass
        self.assertIsNotNone(self.nifti_plugin)

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all', 'time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            'none',
            self.opts)
        self.assertEqual(si1.input_format, 'nifti')
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    # @unittest.skip("skipping test_read_2D")
    def test_read_2D(self):
        dcm = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        si2 = dcm[0, ...]
        with tempfile.TemporaryDirectory() as d:
            si2.write(d, formats=['nifti'])
            si3 = Series(d, template=si2)
        with tempfile.TemporaryDirectory() as d:
            si3.write(d, formats=['dicom'])
        self.assertEqual(si2.dtype, si3.dtype)
        self.assertEqual(si2.shape, si3.shape)
        np.testing.assert_array_equal(si2, si3)

    # @unittest.skip("skipping test_read_3D")
    def test_read_3D(self):
        dcm = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            si3 = Series(d, template=dcm)
        with tempfile.TemporaryDirectory() as d:
            si3.write(d, formats=['dicom'])
        self.assertEqual(dcm.dtype, si3.dtype)
        self.assertEqual(dcm.shape, si3.shape)
        np.testing.assert_array_equal(dcm, si3)

    # @unittest.skip("skipping test_qform_3D")
    def test_qform_3D(self):
        dcm = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            n = Series(d)
        _ = n.transformationMatrix
        self.assertEqual(dcm.shape, n.shape)
        self.assertEqual(dcm.dtype, n.dtype)
        np.testing.assert_array_almost_equal(
            dcm.transformationMatrix,
            n.transformationMatrix,
            decimal=2)

    @unittest.skip("skipping test_compare_qform_to_dicom")
    def test_compare_qform_to_dicom(self):
        dcm = Series(
            os.path.join('data', 'dicom', 'time'),
            'time')
        n = Series(
            os.path.join('data', 'nifti', 'time_all', 'time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            'time')
        _ = n.transformationMatrix
        self.assertEqual(dcm.shape, n.shape)
        # self.assertEqual(dcm.dtype, n.dtype)
        np.testing.assert_array_almost_equal(dcm.transformationMatrix, n.transformationMatrix)

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [
                os.path.join('data',
                             'nifti',
                             'time_all',
                             'time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
                os.path.join('data',
                             'nifti',
                             'time_all',
                             'time_all_fl3d_dynamic_20190207140517_14.nii.gz')
            ],
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (2, 10, 40, 192, 152))

    # @unittest.skip("skipping test_zipread_single_file")
    def test_zipread_single_file(self):
        si1 = Series(
            os.path.join(
                'data',
                'nifti',
                'time_all.zip?time/time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    # @unittest.skip("skipping test_zipread_single_directory")
    def test_zipread_single_directory(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip?time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    # @unittest.skip("skipping test_zipread_all_files")
    def test_zipread_all_files(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

    # @unittest.skip("skipping test_write_single_file")
    def test_write_single_file(self):
        si1 = Series(
            os.path.join(
                'data',
                'nifti',
                'time_all',
                'time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            'none',
            self.opts)
        with tempfile.TemporaryDirectory() as d:
            si1.write(d + '?Image%1d.nii.gz', formats=['nifti'])
            si2 = Series(os.path.join(d, 'Image0.nii.gz'))
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_read_3d_nifti")
    def test_read_3d_nifti(self):
        si1 = Series(
            os.path.join(
                'data',
                'nifti',
                'time_all',
                'time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        si1 = si1[0]
        logging.debug('test_read_3d_nifti: si1 {} {} {} {}'.format(type(si1), si1.dtype, si1.min(), si1.max()))
        logging.debug('test_read_3d_nifti: si1.slices {}'.format(si1.slices))

        si1.spacing = (5, 0.41015625, 0.41015625)
        for slice in range(si1.shape[0]):
            si1.imagePositions = {
                slice:
                    np.array([slice, 1, 0])
            }
        si1.orientation = np.array([1, 0, 0, 0, 1, 0])
        logging.debug('test_read_3d_nifti: si1.tags {}'.format(si1.tags))
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'nifti?Image_%05d'), formats=['nifti'], opts=self.opts)
            logging.debug('test_read_3d_nifti: si1 {} {} {}'.format(si1.dtype, si1.min(), si1.max()))

            si2 = Series(
                os.path.join(d, 'nifti', 'Image_00000.nii.gz'),
                'none',
                self.opts)
        # noinspection PyArgumentList
        logging.debug('test_read_3d_nifti: si2 {} {} {}'.format(si2.dtype, si2.min(), si2.max()))

        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

        logging.debug('test_read_3d_nifti: Get si1.slices {}'.format(si1.slices))
        logging.debug('test_read_3d_nifti: Set s3')
        s3 = si1.astype(np.float64)
        logging.debug('test_read_3d_nifti: s3 {} {} {} {}'.format(type(s3),
                                                                  issubclass(type(s3), Series), s3.dtype, s3.shape))
        logging.debug('test_read_3d_nifti: s3 {} {} {}'.format(s3.dtype,
                                                               s3.min(), s3.max()))
        logging.debug('test_read_3d_nifti: s3.slices {}'.format(s3.slices))
        si3 = Series(s3)
        np.testing.assert_array_almost_equal(si1, si3, decimal=4)
        logging.debug('test_read_3d_nifti: si3.slices {}'.format(si3.slices))
        logging.debug('test_read_3d_nifti: si3 {} {} {}'.format(type(si3), si3.dtype, si3.shape))
        with tempfile.TemporaryDirectory() as d:
            si3.write(os.path.join(d, 'n?Image_%05d'), formats=['nifti'], opts=self.opts)

        s3 = si1 - si2
        with tempfile.TemporaryDirectory() as d:
            s3.write(os.path.join(d, 'diff?Image_%05d'), formats=['nifti'], opts=self.opts)

    # @unittest.skip("skipping test_read_3d_nifti_no_opt")
    # noinspection PyArgumentList
    def test_read_3d_nifti_no_opt(self):
        si1 = Series(os.path.join(
            'data', 'nifti', 'time_all', 'time_all_fl3d_dynamic_20190207140517_14.nii.gz'))
        logging.debug('test_read_3d_nifti_no_opt: si1 {} {} {} {}'.format(type(si1), si1.dtype, si1.min(), si1.max()))
        logging.debug('test_read_3d_nifti_no_opt: si1.slices {}'.format(si1.slices))

    # @unittest.skip("skipping test_write_3d_nifti_no_opt")
    # noinspection PyArgumentList
    def test_write_3d_nifti_no_opt(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        logging.debug('test_write_3d_nifti_no_opt: si1 {} {} {} {}'.format(type(si1), si1.dtype, si1.min(), si1.max()))
        logging.debug('test_write_3d_nifti_no_opt: si1.slices {}'.format(si1.slices))
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['nifti'])


class Test4DNIfTIPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'nifti', '--input_shape', '8x30'])

        plugins = imagedata.formats.get_plugins_list()
        self.nifti_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'nifti':
                self.nifti_plugin = pclass
        self.assertIsNotNone(self.nifti_plugin)

    # @unittest.skip("skipping test_write_4d_nifti")
    def test_write_4d_nifti(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (10, 40, 192, 152))

        si1.sort_on = imagedata.formats.SORT_ON_SLICE
        logging.debug("test_write_4d_nifti: si1.sort_on {}".format(
            imagedata.formats.sort_on_to_str(si1.sort_on)))
        si1.output_dir = 'single'
        # si1.output_dir = 'multi'
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['nifti'], opts=self.opts)

            # Read back the NIfTI data and verify that the header was modified
            si2 = Series(
                d,
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)


if __name__ == '__main__':
    unittest.main()
