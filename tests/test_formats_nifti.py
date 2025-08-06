#!/usr/bin/env python3

import unittest
import os.path
import tempfile
import numpy as np
import logging
import argparse
import nibabel

# from .context import imagedata
import src.imagedata.cmdline as cmdline
import src.imagedata.formats as formats
from src.imagedata.series import Series
from src.imagedata.collection import Cohort


class TestWriteNIfTIPlugin(unittest.TestCase):

    def _compare_nifti_data(self, img1, img2, descr1='img1', descr2='img2', verify_zooms=True):
        hdr1, hdr2 = img1.header, img2.header
        self.assertEqual(hdr1.get_data_shape(), hdr2.get_data_shape(), "get_data_shape")
        self.assertEqual(hdr1.get_dim_info(), hdr2.get_dim_info(), "get_dim_info")
        self.assertEqual(hdr1.get_xyzt_units(), hdr2.get_xyzt_units(), "get_xyzt_units")
        if verify_zooms:
            np.testing.assert_array_almost_equal(hdr1.get_zooms(), hdr2.get_zooms(), decimal=4, err_msg="get_zooms")
        sform1, sform2 = hdr1.get_sform(coded=True)[0], hdr2.get_sform(coded=True)[0]
        np.testing.assert_array_almost_equal(sform1, sform2, decimal=4)
        qform1, qform2 = hdr1.get_qform(coded=True)[0], hdr2.get_qform(coded=True)[0]
        if qform1 is not None:
            self.assertIsNotNone(qform2)
        np.testing.assert_array_almost_equal(qform1, qform2, decimal=4)

        si1, si2 = np.asarray(img1.dataobj), np.asarray(img2.dataobj)
        np.testing.assert_array_equal(si1, si2)

    def test_tra_rl(self):
        dcm = Series(os.path.join('data', 'dicom', 'tra_rl.zip'))
        nii = nibabel.load(
            os.path.join('data', 'nifti', 'tra_rl.nii.gz')
        )
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            for entry in os.scandir(path=d):
                filename = entry.path
            check = nibabel.load(filename)
            diff = Series(nii.dataobj) - Series(check.dataobj)
            # diff.write(d, formats=['dicom'])
            self._compare_nifti_data(nii, check, 'dcm2niix', 'niftiplugin')

    def test_tra_oblique(self):
        dcm = Series(os.path.join('data', 'dicom', 'tra_oblique.zip'))
        nii = nibabel.load(
            os.path.join('data', 'nifti', 'tra_oblique.nii.gz')
        )
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            for entry in os.scandir(path=d):
                filename = entry.path
            check = nibabel.load(filename)
            self._compare_nifti_data(nii, check, 'dcm2niix', 'niftiplugin')

    def test_cor_hf(self):
        dcm = Series(os.path.join('data', 'dicom', 'cor_hf.zip'))
        nii = nibabel.load(
            os.path.join('data', 'nifti', 'cor_hf.nii.gz')
        )
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            for entry in os.scandir(path=d):
                filename = entry.path
            check = nibabel.load(filename)
            self._compare_nifti_data(nii, check, 'dcm2niix', 'niftiplugin')

    def test_cor_oblique(self):
        dcm = Series(os.path.join('data', 'dicom', 'cor_oblique.zip'))
        nii = nibabel.load(
            os.path.join('data', 'nifti', 'cor_oblique.nii.gz')
        )
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            for entry in os.scandir(path=d):
                filename = entry.path
            check = nibabel.load(filename)
            self._compare_nifti_data(nii, check, 'dcm2niix', 'niftiplugin')

    def test_cor_rl(self):
        dcm = Series(os.path.join('data', 'dicom', 'cor_rl.zip'))
        nii = nibabel.load(
            os.path.join('data', 'nifti', 'cor_rl.nii.gz')
        )
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            for entry in os.scandir(path=d):
                filename = entry.path
            check = nibabel.load(filename)
            self._compare_nifti_data(nii, check, 'dcm2niix', 'niftiplugin')

    def test_sag_ap(self):
        dcm = Series(os.path.join('data', 'dicom', 'sag_ap.zip'))
        nii = nibabel.load(
            os.path.join('data', 'nifti', 'sag_ap.nii.gz')
        )
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            for entry in os.scandir(path=d):
                filename = entry.path
            check = nibabel.load(filename)
            self._compare_nifti_data(nii, check, 'dcm2niix', 'niftiplugin')

    def test_sag_hf(self):
        dcm = Series(os.path.join('data', 'dicom', 'sag_hf.zip'))
        nii = nibabel.load(
            os.path.join('data', 'nifti', 'sag_hf.nii.gz')
        )
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            for entry in os.scandir(path=d):
                filename = entry.path
            check = nibabel.load(filename)
            self._compare_nifti_data(nii, check, 'dcm2niix', 'niftiplugin')

    def test_sag_oblique(self):
        dcm = Series(os.path.join('data', 'dicom', 'sag_oblique.zip'))
        nii = nibabel.load(
            os.path.join('data', 'nifti', 'sag_oblique.nii.gz')
        )
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            for entry in os.scandir(path=d):
                filename = entry.path
            check = nibabel.load(filename)
            self._compare_nifti_data(nii, check, 'dcm2niix', 'niftiplugin')


class TestReadWriteNIfTIPlugin(unittest.TestCase):


    def _compare_dicom_data(self, dcm, nifti, verify_spacing=True):
        self.assertEqual('dicom', dcm.input_format, "dicom input_format")
        self.assertEqual(dcm.shape, nifti.shape, "shape")
        self.assertEqual(dcm.slices, nifti.slices, "slices")
        if verify_spacing:
            np.testing.assert_allclose(nifti.spacing, dcm.spacing,
                                       atol=1e-4, err_msg="nifti vs dicom spacing")

        for s in range(dcm.slices):
            np.testing.assert_allclose(nifti.imagePositions[s].reshape(3), dcm.imagePositions[s].reshape(3),
                                       atol=1e-3, err_msg="imagePositions[{}]".format(s))
        np.testing.assert_allclose(nifti.orientation, dcm.orientation,
                                   atol=1e-6, err_msg="orientation")
        np.testing.assert_allclose(nifti.transformationMatrix, dcm.transformationMatrix,
                                   atol=1e-2, err_msg="transformationMatrix")
        np.testing.assert_array_equal(nifti, dcm, err_msg="voxel values")

    def test_compare_sag_ap(self):
        dcm = Series(os.path.join('data', 'dicom', 'sag_ap.zip'))
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            nifti = Series(d, input_format='nifti')
        self._compare_dicom_data(dcm, nifti)


class TestReadNIfTIPlugin(unittest.TestCase):

    def _compare_dicom_data(self, dcm, nifti, verify_spacing=True):
        self.assertEqual('dicom', dcm.input_format, "dicom input_format")
        self.assertEqual(dcm.shape, nifti.shape, "shape")
        self.assertEqual(dcm.slices, nifti.slices, "slices")
        if verify_spacing:
            np.testing.assert_allclose(nifti.spacing, dcm.spacing,
                                       atol=1e-4, err_msg="nifti vs dicom spacing")

        for s in range(dcm.slices):
            np.testing.assert_allclose(
                np.sort(nifti.imagePositions[s]),
                np.sort(dcm.imagePositions[s]),
                atol=1e-3, err_msg="imagePositions[{}]".format(s))
        np.testing.assert_allclose(nifti.orientation, dcm.orientation,
                                   atol=1e-6, err_msg="orientation")
        np.testing.assert_allclose(nifti.transformationMatrix, dcm.transformationMatrix,
                                   atol=1e-2, err_msg="transformationMatrix")
        np.testing.assert_array_equal(nifti, dcm, err_msg="voxel values")

    def test_compare_sag_ap(self):
        dcm = Series(os.path.join('data', 'dicom', 'sag_ap.zip'), input_format='dicom')
        nifti = Series(os.path.join('data', 'nifti', 'sag_ap.nii.gz'), input_format='nifti')
        self._compare_dicom_data(dcm, nifti)

    def test_compare_sag_hf(self):
        dcm = Series(os.path.join('data', 'dicom', 'sag_hf.zip'), input_format='dicom')
        nifti = Series(os.path.join('data', 'nifti', 'sag_hf.nii.gz'), input_format='nifti')
        self._compare_dicom_data(dcm, nifti)

    def test_compare_sag_oblique(self):
        dcm = Series(os.path.join('data', 'dicom', 'sag_oblique.zip'), input_format='dicom')
        nifti = Series(os.path.join('data', 'nifti', 'sag_oblique.nii.gz'), input_format='nifti')
        self._compare_dicom_data(dcm, nifti, verify_spacing=False)

    def test_compare_cor_hf(self):
        dcm = Series(os.path.join('data', 'dicom', 'cor_hf.zip'), input_format='dicom')
        nifti = Series(os.path.join('data', 'nifti', 'cor_hf.nii.gz'), input_format='nifti')
        self._compare_dicom_data(dcm, nifti)

    def test_compare_cor_oblique(self):
        dcm = Series(os.path.join('data', 'dicom', 'cor_oblique.zip'), input_format='dicom')
        nifti = Series(os.path.join('data', 'nifti', 'cor_oblique.nii.gz'), input_format='nifti')
        self._compare_dicom_data(dcm, nifti, verify_spacing=False)

    def test_compare_cor_rl(self):
        dcm = Series(os.path.join('data', 'dicom', 'cor_rl.zip'), input_format='dicom')
        nifti = Series(os.path.join('data', 'nifti', 'cor_rl.nii.gz'), input_format='nifti')
        self._compare_dicom_data(dcm, nifti)

    def test_compare_tra_oblique(self):
        dcm = Series(os.path.join('data', 'dicom', 'tra_oblique.zip'), input_format='dicom')
        nifti = Series(os.path.join('data', 'nifti', 'tra_oblique.nii.gz'), input_format='nifti')
        self._compare_dicom_data(dcm, nifti, verify_spacing=False)

    def test_compare_tra_rl(self):
        dcm = Series(os.path.join('data', 'dicom', 'tra_rl.zip'), input_format='dicom')
        nifti = Series(os.path.join('data', 'nifti', 'tra_rl.nii.gz'), input_format='nifti')
        self._compare_dicom_data(dcm, nifti)


class Test2DNIfTIPlugin(unittest.TestCase):
    def test_read_single_slice(self):
        dcm = Series(os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'), input_format='dicom')
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            _ = Series(d, input_format='nifti')


class Test3DNIfTIPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'nifti', '--serdes', '1'])

    def test_nifti_plugin(self):
        plugins = formats.get_plugins_list()
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
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    def test_dtype_int64(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all', 'time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            'none',
            dtype=int,
            input_format='nifti')
        self.assertEqual(si1.dtype, np.int64)

    def test_dtype_float(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all', 'time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            'none',
            dtype=float,
            input_format='nifti')
        self.assertEqual(si1.dtype, np.float64)

    # @unittest.skip("skipping test_qform_3D")
    def test_qform_3D(self):
        dcm = Series(os.path.join('data', 'dicom', 'time', 'time00'), input_format='dicom')
        self.assertEqual('dicom', dcm.input_format)
        with tempfile.TemporaryDirectory() as d:
            dcm.write(d, formats=['nifti'])
            n = Series(d, input_format='nifti')
        self.assertEqual('nifti', n.input_format)
        self.assertEqual(dcm.shape, n.shape)
        self.assertEqual(dcm.dtype, n.dtype)
        np.testing.assert_allclose(n.transformationMatrix, dcm.transformationMatrix, atol=1e-2)

    # @unittest.skip("skipping test_compare_qform_to_dicom")
    def test_compare_qform_to_dicom(self):
        dcm = Series(
            os.path.join('data', 'dicom', 'time'),
            'time',
            input_format='dicom')
        self.assertEqual('dicom', dcm.input_format)
        n = Series(
            os.path.join('data', 'nifti', 'time', 'time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            'time',
            input_format='nifti')
        self.assertEqual('nifti', n.input_format)
        self.assertEqual(dcm.shape, n.shape)
        # obj.assertEqual(dcm.dtype, n.dtype)
        np.testing.assert_allclose(n.transformationMatrix, dcm.transformationMatrix, atol=1e-2)

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
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(np.int16, si1.dtype)
        self.assertEqual((3, 3, 192, 152), si1.shape)

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
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_zipread_single_directory")
    def test_zipread_single_directory(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip?time'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_zipread_all_files")
    def test_zipread_all_files(self):
        si1 = Series(
            os.path.join('data', 'nifti', 'time_all.zip'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    def test_write_ndarray(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128)).write(d, formats=['nifti'])

    def test_write_single_file_not_directory(self):
        a = Series(np.eye(128))
        with tempfile.TemporaryDirectory() as d:
            filename = os.path.join(d, 'test.nii.gz')
            a.write(
                filename,
                formats=['nifti']
            )
            if not os.path.isfile(filename):
                raise AssertionError('File does not exist: {}'.format(filename))

    def test_write_single_file_in_directory(self):
        a = Series(np.eye(128))
        with tempfile.TemporaryDirectory() as d:
            a.write(
                d,
                formats=['nifti']
            )
            expect_filename = os.path.join(d, 'Image.nii.gz')
            if not os.path.isfile(expect_filename):
                raise AssertionError('File does not exist: {}'.format(expect_filename))

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
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'nifti', '--input_shape', '8x30'])

        plugins = formats.get_plugins_list()
        self.nifti_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'nifti':
                self.nifti_plugin = pclass
        self.assertIsNotNone(self.nifti_plugin)

    def test_write_4d_nifti_time(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

        si1.sort_on = formats.SORT_ON_SLICE
        logging.debug("test_write_4d_nifti: si1.sort_on {}".format(
            formats.sort_on_to_str(si1.sort_on)))
        si1.output_dir = 'single'
        # si1.output_dir = 'multi'
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['nifti'], opts=self.opts)

            # Read back the NIfTI data and verify that the header was modified
            si2 = Series(
                d,
                formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    def test_write_4d_nifti_dwi(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'dwi'),
            formats.INPUT_ORDER_B,
            self.opts
        )
        self.assertEqual(si1.shape, (3, 30, 384, 312))
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['nifti'], opts=self.opts)
            # Read back the NIfTI data and verify
            si2 = Series(d,
                         formats.INPUT_ORDER_B,
                         self.opts)
            self.assertEqual(si1.shape, si2.shape)
            np.testing.assert_array_almost_equal(si1.spacing, si2.spacing)
            np.testing.assert_array_equal(si1, si2)


class TestNIfTIPluginWrite(unittest.TestCase):
    def test_write_dicom(self):
        si1 = Series(
            # os.path.join('data', 'nifti', 'time_all', 'time_all_fl3d_dynamic_20190207140517_14.nii.gz'),
            os.path.join('data', 'nifti', 'cor_rl.nii.gz'),
            'time')
        self.assertEqual(si1.input_format, 'nifti')
        self.assertEqual(si1.dtype, np.int16)
        self.assertEqual(si1.shape, (3, 320, 220))
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'])

    def test_write_cohort(self):
        cohort = Cohort('data/dicom/cohort.zip')
        with tempfile.TemporaryDirectory() as d:
            cohort.write(d, formats=['nifti'])


if __name__ == '__main__':
    unittest.main()
