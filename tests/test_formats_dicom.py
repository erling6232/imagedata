import unittest
import math
import os.path
import tempfile
import numpy as np
import logging
import argparse
import pydicom.filereader
from numbers import Number
from pydicom.dataset import Dataset

import src.imagedata.cmdline as cmdline
import src.imagedata.formats as formats
from src.imagedata.series import Series
from .compare_headers import compare_headers, compare_pydicom, compare_tags, compare_tags_in_slice


class TestDicomPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

    def test_dicom_plugin(self):
        plugins = formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

    def test_dicom_plugin_only(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
            'none',
            input_format='dicom'
        )

    def test_cannot_sort_dicom(self):
        si = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
            'none',
            input_format='dicom'
        )
        with tempfile.TemporaryDirectory() as self.d:
            # Duplicate image file
            si.write(os.path.join(self.d, '0'), formats=['dicom'], opts = {'keep_uid': True})
            si1 = si + 1
            si1.seriesInstanceUID = si.seriesInstanceUID
            si1.write(os.path.join(self.d, '1'), formats=['dicom'], opts = {'keep_uid': True})
            with self.assertRaises((formats.UnknownInputError, formats.CannotSort)) as context:
                _ = Series(self.d, input_format='dicom')

    def test_without_dicom_plugin(self):
        def _read_series():
            si1 = Series(
                os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
                'none',
                input_format='mat'
            )
        self.assertRaises(formats.UnknownInputError, _read_series)

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual(si1.input_format, 'dicom')
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))
        self.assertEqual(len(si1.axes), 2)

    def test_dtype_int64(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
            'none',
            dtype=int,
            input_format='dicom')
        self.assertEqual(si1.dtype, np.int64)

    def test_dtype_float(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
            'none',
            dtype=float,
            input_format='dicom')
        self.assertEqual(si1.dtype, np.float64)

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [
                os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
                os.path.join('data', 'dicom', 'time', 'time00', 'Image_00021.dcm')
            ],
            'none',
            self.opts, input_format='dicom')
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 192, 152))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))

    def test_read_single_directory_headers_only(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            opts={'headers_only': True})
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(tuple(), si1.shape)
        self.assertEqual(3, len(si1.axes))
        self.assertEqual(14, si1.seriesNumber)

    def test_read_auto_volume(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'auto',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(formats.INPUT_ORDER_NONE, si1.input_order)
        self.assertEqual((3, 192, 152), si1.shape)
        self.assertEqual(3, len(si1.axes))
        self.assertEqual(14, si1.seriesNumber)

    def test_read_auto_time(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            'auto',
            self.opts,
            input_format='dicom')
        self.assertEqual(formats.INPUT_ORDER_TIME, si1.input_order)
        self.assertEqual((3, 3, 192, 152), si1.shape)
        self.assertEqual(4, len(si1.axes))
        self.assertEqual(14, si1.seriesNumber)

    # @unittest.skip("skipping test_read_dicom_3D_no_opt")
    def test_read_dicom_3D_no_opt(self):
        d = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'))
        self.assertEqual('dicom', d.input_format)
        self.assertEqual(d.dtype, np.uint16)
        self.assertEqual(d.shape, (192, 152))

    # @unittest.skip("skipping test_read_dicom_4D")
    def test_read_dicom_4D(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        t = np.array([0., 2.99, 5.97])
        np.testing.assert_array_almost_equal(t, si1.timeline, decimal=2)
        # for axis in si1.axes:
        #    logging.debug('test_read_dicom_4D: axis {}'.format(axis))

    # @unittest.skip("skipping test_read_dicom_4D_wrong_order")
    def test_read_dicom_4D_wrong_order(self):
        with self.assertRaises((formats.UnknownInputError, formats.CannotSort)) as context:
            _ = Series(
                os.path.join('data', 'dicom', 'time'),
                input_format='dicom',
                input_order='b',
                opts=self.opts)

    # @unittest.skip("skipping test_read_dicom_user_defined_TI")
    def test_read_dicom_user_defined_TI(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'TI'),
            input_order='ti',
            input_format='dicom',
            ti='InversionTime', ignore_series_uid=True)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (5, 1, 384, 384))
        with tempfile.TemporaryDirectory() as d:
            si1.write(d,
                      formats=['dicom'],
                      opts={'ti': 'InversionTime'})
            si2 = Series(d,
                         input_order='ti',
                         ti='InversionTime')
        self.assertEqual('dicom', si2.input_format)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    # @unittest.skip("skipping test_read_dicom_user_function_TI")
    def test_read_dicom_user_function_TI(self):

        def _get_TI(im: Dataset) -> float:
            return float(im.data_element('InversionTime').value)

        si1 = Series(
            os.path.join('data', 'dicom', 'TI'),
            input_order='ti',
            input_format='dicom',
            ti=_get_TI, ignore_series_uid=True)
        with tempfile.TemporaryDirectory() as d:
            si1.write(d,
                      formats=['dicom'],
                      opts={'ti': 'InversionTime'})
            si2 = Series(d,
                         input_order='ti',
                         input_format='dicom',
                         ti='InversionTime')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    def test_verify_correct_slice(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
            'none',
            self.opts,
            input_format='dicom')
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image_00000.dcm'),
                      formats=['dicom'], opts={'keep_uid': True})

            # Use pydicom as truth to verify that written copy is identical to original
            orig = pydicom.filereader.dcmread(
                os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm')
            )
            temp = pydicom.filereader.dcmread(
                os.path.join(d, 'Image_00000.dcm')
            )
            compare_pydicom(self, orig, temp)

    def test_verify_correct_volume(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'], opts={'keep_uid': True})

            # Use pydicom as truth to verify that written copy is identical to original
            for i, f in enumerate(['Image_00019.dcm', 'Image_00020.dcm', 'Image_00021.dcm']):
                orig = pydicom.filereader.dcmread(
                    os.path.join('data', 'dicom', 'time', 'time00', f)
                )
                temp = pydicom.filereader.dcmread(
                    os.path.join(d, 'Image_{:05d}.dcm'.format(i))
                )
                compare_pydicom(self, orig, temp)

    def test_verify_correct_volume_no_slicelocation(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        del si1.dicomTemplate.SliceLocation
        si1.axes[0]._values = None
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'], opts={'keep_uid': True})

            # Use pydicom as truth to verify that written copy is identical to original
            for i, f in enumerate(['Image_00019.dcm', 'Image_00020.dcm', 'Image_00021.dcm']):
                orig = pydicom.filereader.dcmread(
                    os.path.join('data', 'dicom', 'time', 'time00', f)
                )
                temp = pydicom.filereader.dcmread(
                    os.path.join(d, 'Image_{:05d}.dcm'.format(i))
                )
                compare_pydicom(self, orig, temp)

    def test_verify_image_positions_and_transformation_matrix(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'sag_ap.zip'),
            'none',
            self.opts,
            input_format='dicom')
        M = si1.transformationMatrix
        T0 = M[:3, 3]
        ipp = np.array(T0)
        for s in range(si1.slices):
            np.testing.assert_allclose(si1.imagePositions[s], ipp,
                                       atol=1e-3, err_msg='imagePositions[{}]'.format(s))
            ipp += M[:3, 0]

    def test_verify_image_positions_and_transformation_matrix2(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time_all'),
            'time',
            self.opts,
            input_format='dicom')
        M = si1.transformationMatrix
        T0 = M[:3, 3]
        ipp = np.array(T0)
        for s in range(si1.slices):
            np.testing.assert_allclose(si1.imagePositions[s], ipp,
                                       atol=1e-3, err_msg='imagePositions[{}]'.format(s))
            ipp += M[:3, 0]

    def test_verify_correct_4D(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            'time',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'], opts={'keep_uid': True})

            # Use pydicom as truth to verify that written copy is identical to original
            i = 0
            for t in ['time00', 'time01', 'time02']:
                for f in ['Image_00019.dcm', 'Image_00020.dcm', 'Image_00021.dcm']:
                    orig = pydicom.filereader.dcmread(
                        os.path.join('data', 'dicom', 'time', t, f)
                    )
                    temp = pydicom.filereader.dcmread(
                        os.path.join(d, 'Image_{:05d}.dcm'.format(i))
                    )
                    compare_pydicom(self, orig, temp, uid=False)
                    i += 1

    # @unittest.skip("skipping test_copy_dicom_4D")
    def test_copy_dicom_4D(self):
        si = Series(
            os.path.join('data', 'dicom', 'time'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si.input_format)
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        newsi = Series(si,
                       formats.INPUT_ORDER_TIME)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (3, 3, 192, 152))

    def test_write_ndarray(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128)).write(d, formats=['dicom'])

    # @unittest.skip("skipping test_write_single_file")
    def test_write_single_file(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm')
        )
        self.assertEqual('dicom', si1.input_format)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image.dcm'),
                      formats=['dicom'])
            si2 = Series(d)
        self.assertEqual('dicom', si2.input_format)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_write_single_directory")
    def test_write_single_directory(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si1.input_format)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image{:05d}.dcm'),
                      formats=['dicom'])
            si2 = Series(d)
        self.assertEqual('dicom', si2.input_format)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_write_dicom_4D")
    def test_write_dicom_4D(self):
        si = Series(
            os.path.join('data', 'dicom', 'time_all'),
            'time',
            self.opts,
            input_format='dicom')
        with tempfile.TemporaryDirectory() as d:
            si.write(os.path.join(d, 'Image_{:05d}.dcm'),
                     formats=['dicom'], opts=self.opts)
            newsi = Series(d,
                           'time',
                           self.opts,
                           input_format='dicom')
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_write_dicom_4D_no_opt")
    def test_write_dicom_4D_no_opt(self):
        si = Series(
            os.path.join('data', 'dicom', 'time_all'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si.input_format)
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        with tempfile.TemporaryDirectory() as d:
            si.write(os.path.join(d, 'Image_{:05d}.dcm'),
                     formats=['dicom'])
            newsi = Series(d,
                           formats.INPUT_ORDER_TIME)
        self.assertEqual('dicom', newsi.input_format)
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (3, 3, 192, 152))

    def test_write_dicom_4D_multi_slice(self):
        si = Series(
            os.path.join('data', 'dicom', 'time_all'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si.input_format)
        with tempfile.TemporaryDirectory() as d:
            si.write(d, opts={
                'output_dir': 'multi',
                'output_sort': formats.SORT_ON_SLICE
            })
            newsi = Series(d, formats.INPUT_ORDER_TIME)
        self.assertEqual('dicom', newsi.input_format)
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)

    def test_write_dicom_4D_multi_tag(self):
        si = Series(
            os.path.join('data', 'dicom', 'time_all'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si.input_format)
        with tempfile.TemporaryDirectory() as d:
            si.write(d, opts={
                'output_dir': 'multi',
                'output_sort': formats.SORT_ON_TAG
            })
            newsi = Series(d, formats.INPUT_ORDER_TIME)
        self.assertEqual('dicom', newsi.input_format)
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)

    def test_write_float(self):
        si = Series(np.arange(8*8*8).reshape((8, 8, 8)))
        si.seriesNumber = 100
        si.seriesDescription = 'float'
        si.imageType = ['DERIVED', 'SECONDARY']
        si.header.photometricInterpretation = 'MONOCHROME2'
        fsi = si / math.sqrt(2)
        fsi_center = fsi.windowCenter
        fsi_width = fsi.windowWidth
        with tempfile.TemporaryDirectory() as d:
            fsi.write(d, formats=['dicom'])
            fsi_read = Series(d)
            self.assertEqual(fsi_read.input_format, 'dicom')
        compare_headers(self, fsi, fsi_read, uid=False)
        self.assertAlmostEqual(fsi_read.windowCenter, fsi_center, places=5)
        self.assertAlmostEqual(fsi_read.windowWidth, fsi_width, places=4)
        self.assertAlmostEqual(fsi.windowCenter, fsi_read.windowCenter, places=4)

    def test_changed_uid(self):
        eye = Series(np.eye(128, dtype=np.uint16))
        eye_seriesInstanceUID = eye.seriesInstanceUID
        with tempfile.TemporaryDirectory() as d:
            eye.write(os.path.join(d, 'Image.dcm'), formats=['dicom'])
            eye_read = Series(d, input_format='dicom')
        self.assertEqual(eye_seriesInstanceUID, eye.seriesInstanceUID)

    def test_changed_uid_on_copy(self):
        eye = Series(np.eye(128, dtype=np.uint16))
        eye_seriesInstanceUID = eye.seriesInstanceUID
        eye_copy = Series(eye)
        self.assertNotEqual(eye_seriesInstanceUID, eye_copy.seriesInstanceUID)

    def test_unchanged_uid_on_slicing(self):
        eye = Series(np.eye(128, dtype=np.uint16))
        eye_seriesInstanceUID = eye.seriesInstanceUID
        eye_copy = eye[0]
        self.assertEqual(eye_seriesInstanceUID, eye_copy.seriesInstanceUID)

    def test_write_keep_uid(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si1.input_format)
        # Make a copy of SOPInstanceUIDs before they are possibly modified in write()
        si1_seriesInstanceUID = si1.seriesInstanceUID
        si1_sopinsuid = {}
        for _slice in range(si1.slices):
            si1_sopinsuid[_slice] = {}
            for _tag in si1.tags[0]:
                si1_sopinsuid[_slice][_tag] = \
                    si1.SOPInstanceUIDs[_tag + (_slice,)]
                    # si1.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image{:05d}.dcm'),
                      formats=['dicom'],
                      opts={'keep_uid': True})
            si2 = Series(d)
        self.assertEqual('dicom', si2.input_format)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        self.assertEqual(si1_seriesInstanceUID, si1.seriesInstanceUID)
        self.assertEqual(si1.seriesInstanceUID, si2.seriesInstanceUID)
        self.assertEqual('1.2.840.10008.5.1.4.1.1.4', si2.SOPClassUID)
        self.assertEqual(si1.slices, si2.slices)
        self.assertEqual(len(si1.tags[0]), len(si2.tags[0]))
        for _slice in range(si1.slices):
            for _tag in si1.tags[0]:
                # si1 SOPInstanceUIDs should be identical to si2
                self.assertEqual(
                    si1.SOPInstanceUIDs[_tag + (_slice,)],
                    si2.SOPInstanceUIDs[_tag + (_slice,)]
                )
                # si2 SOPInstanceUIDs should also be identical to original si1
                self.assertEqual(
                    si1_sopinsuid[_slice][_tag],
                    si2.SOPInstanceUIDs[_tag + (_slice,)]
                )

    def test_write_no_keep_uid(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'), input_format='dicom')
        # Make a copy of SOPInstanceUIDs before they are modified in write()
        si1_seriesInstanceUID = si1.seriesInstanceUID
        si1_sopinsuid = {}
        for _slice in range(si1.slices):
            si1_sopinsuid[_slice] = {}
            for _tag in si1.tags[0]:
                si1_sopinsuid[_slice][_tag] =\
                    si1.SOPInstanceUIDs[_tag + (_slice,)]
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image{:05d}.dcm'),
                      formats=['dicom'],
                      opts={'keep_uid': False})
            si2 = Series(d, input_format='dicom')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        self.assertEqual(si1_seriesInstanceUID, si1.seriesInstanceUID)
        self.assertNotEqual(si1.seriesInstanceUID, si2.seriesInstanceUID)
        self.assertEqual('1.2.840.10008.5.1.4.1.1.4', si2.SOPClassUID)
        self.assertEqual(si1.slices, si2.slices)
        self.assertEqual(len(si1.tags[0]), len(si2.tags[0]))
        for _slice in range(si1.slices):
            for _tag in si1.tags[0]:
                # si2 SOPInstanceUIDs should differ from original si1
                self.assertNotEqual(
                    si1_sopinsuid[_slice][_tag],
                    si2.SOPInstanceUIDs[_tag + (_slice,)]
                )

    def test_read_dicom_not_DWI(self):
        with self.assertRaises((formats.UnknownInputError, formats.CannotSort)) as context:
            _ = Series(
                os.path.join('data', 'dicom', 'time'),
                input_format='dicom',
                input_order='b'
            )

    def test_read_dicom_not_DWI_no_CSA(self):
        with self.assertRaises((formats.UnknownInputError, formats.CannotSort)) as context:
            _ = Series(
                os.path.join('data', 'dicom', 'lena_color.dcm'),
                input_format='dicom',
                input_order='b'
            )


class TestDicomZipPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

        plugins = formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

    # @unittest.skip("skipping test_write_zip")
    def test_write_zip(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si1.input_format)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'dicom.zip?Image_{:05d}.dcm'),
                      formats=['dicom'])
            si2 = Series(os.path.join(d, 'dicom.zip'))
        self.assertEqual('dicom', si2.input_format)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)


class TestZipArchiveDicom(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_single_file_relative")
    def test_read_single_file_relative(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_single_file_wildcard")
    def test_read_single_file_wildcard(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?*time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_0002[01].dcm'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 192, 152))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))

    # @unittest.skip("skipping test_read_two_directories")
    def test_read_two_directories(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time0[02]/'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 3, 192, 152))

    # @unittest.skip("skipping test_read_all_files")
    def test_read_all_files(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))


class TestWriteZipArchiveDicom(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(os.path.join('data', 'dicom', 'time.zip?time/time00/Image_00020.dcm'))
        self.assertEqual('dicom', si1.input_format)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'dicom.zip'), formats=['dicom'])
            si2 = Series(os.path.join(d, 'dicom.zip?Image_00000.dcm'))
        self.assertEqual('dicom', si2.input_format)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_read_single_file_relative")
    def test_read_single_file_relative(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_single_file_wildcard")
    def test_read_single_file_wildcard(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?*time00/Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (192, 152))

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/Image_0002[01].dcm'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 192, 152))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time00/'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 192, 152))

    # @unittest.skip("skipping test_read_two_directories")
    def test_read_two_directories(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/time0[02]/'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 3, 192, 152))

    # @unittest.skip("skipping test_read_all_files")
    def test_read_all_files(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time.zip?time/'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))


class TestDicomSlicing(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

        plugins = formats.get_plugins_list()
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
        self.assertEqual('dicom', si1.input_format)
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
        self.assertEqual('dicom', si1.input_format)
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
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        a2 = np.array(si1)[1:3, 1:3]
        si2 = si1[1:3, 1:3]
        np.testing.assert_array_equal(a2, si2)
        self.assertEqual(len(si2.imagePositions), 2)
        np.testing.assert_array_equal(si2.imagePositions[0], si1.imagePositions[1])
        np.testing.assert_array_equal(si2.imagePositions[1], si1.imagePositions[2])
        self.assertEqual(len(si2.tags[1]), 2)

    # @unittest.skip("skipping test_slice_ellipsis_first")
    def test_slice_ellipsis_first(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si1.input_format)
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
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si1.input_format)
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
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si1.input_format)
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


class TestDicomNDSort(unittest.TestCase):

    #@unittest.skip("skipping test_t1_de_te")
    def test_5D_time_te(self):
        si = Series(
            os.path.join('data', 'dicom', '5D.zip?t1_fl2d_DE_4TEs'),
            'time,te',
            input_format='dicom'
        )
        with tempfile.TemporaryDirectory() as d:
            si.write(os.path.join(d, 'slice', 'single'), formats=['dicom'],
                     opts={'output_sort': 0, 'output_dir': 'single'}
                     )
            si1 = Series(os.path.join(d, 'slice', 'single'),
                         'time,te',
                         input_format='dicom'
            )
            np.testing.assert_array_equal(si1, si)
            np.testing.assert_array_equal(si1.tags[0], si.tags[0])
            si.write(os.path.join(d, 'slice', 'multi'), formats=['dicom'],
                     opts={'output_sort': 0, 'output_dir': 'multi'}
                     )
            si2 = Series(os.path.join(d, 'slice', 'multi'),
                         'time,te',
                         input_format='dicom'
                         )
            np.testing.assert_array_equal(si2, si)
            np.testing.assert_array_equal(si2.tags[0], si.tags[0])
            si.write(os.path.join(d, 'tag', 'single'), formats=['dicom'],
                     opts={'output_sort': 1, 'output_dir': 'single'}
            )
            si3 = Series(os.path.join(d, 'tag', 'single'),
                         'time,te',
                         input_format='dicom'
                         )
            np.testing.assert_array_equal(si3, si)
            np.testing.assert_array_equal(si3.tags[0], si.tags[0])
            si.write(os.path.join(d, 'tag', 'multi'), formats=['dicom'],
                     opts={'output_sort': 1, 'output_dir': 'multi'}
                     )
            si4 = Series(os.path.join(d, 'tag', 'multi'),
                         'time,te',
                         input_format='dicom'
                         )
            np.testing.assert_array_equal(si4, si)
            np.testing.assert_array_equal(si4.tags[0], si.tags[0])

    def test_slice_5D_time_te(self):
        si = Series(
            os.path.join('data', 'dicom', '5D.zip?t1_fl2d_DE_4TEs'),
            'time,te',
            input_format='dicom'
        )
        # Slice row/column
        si1 = si[..., 10:40, 20:30]
        self.assertAlmostEqual(si.axes.time.values, si1.axes.time.values)
        self.assertAlmostEqual(si.axes.te.values, si1.axes.te.values)
        self.assertAlmostEqual(si.axes.slice.values, si1.axes.slice.values)
        np.testing.assert_array_almost_equal(
            np.array(si.axes.row.values[10:40]), np.array(si1.axes.row.values))
        np.testing.assert_array_almost_equal(
            np.array(si.axes.column.values[20:30]), np.array(si1.axes.column.values))
        # Slice slice direction
        si2 = si[..., 1:, :, :]
        self.assertAlmostEqual(si.axes.time.values, si2.axes.time.values)
        self.assertAlmostEqual(si.axes.te.values, si2.axes.te.values)
        np.testing.assert_array_almost_equal(
            np.array(si.axes.slice.values[1:]), np.array(si2.axes.slice.values))
        np.testing.assert_array_almost_equal(si2.sliceLocations, si.sliceLocations[1:])
        self.assertAlmostEqual(si.axes.row.values, si2.axes.row.values)
        self.assertAlmostEqual(si.axes.column.values, si2.axes.column.values)
        # Slice TE
        si3 = si[:, 1:, ...]
        self.assertAlmostEqual(si.axes.time.values, si3.axes.time.values)
        np.testing.assert_array_almost_equal(
            np.array(si.axes.te.values[1:]), np.array(si3.axes.te.values))
        self.assertAlmostEqual(si.axes.slice.values, si3.axes.slice.values)
        self.assertAlmostEqual(si.axes.row.values, si3.axes.row.values)
        self.assertAlmostEqual(si.axes.column.values, si3.axes.column.values)
        compare_tags_in_slice(self, si.tags, si3.tags, axis=1, slicing=slice(1, None))
        # Slice time
        si4 = si[1:]
        np.testing.assert_array_almost_equal(
            np.array(si.axes.time.values[1:]), np.array(si4.axes.time.values))
        self.assertAlmostEqual(si.axes.te.values, si1.axes.te.values)
        self.assertAlmostEqual(si.axes.slice.values, si4.axes.slice.values)
        self.assertAlmostEqual(si.axes.row.values, si4.axes.row.values)
        self.assertAlmostEqual(si.axes.column.values, si4.axes.column.values)
        compare_tags_in_slice(self, si.tags, si4.tags, axis=0, slicing=slice(1, None))
        # Slice time and TE
        si5 = si[1:, 1:, ...]
        np.testing.assert_array_almost_equal(
            np.array(si.axes.time.values[1:]), np.array(si5.axes.time.values))
        np.testing.assert_array_almost_equal(
            np.array(si.axes.te.values[1:]), np.array(si5.axes.te.values))
        self.assertAlmostEqual(si.axes.slice.values, si5.axes.slice.values)
        self.assertAlmostEqual(si.axes.row.values, si5.axes.row.values)
        self.assertAlmostEqual(si.axes.column.values, si5.axes.column.values)
        compare_tags_in_slice(self, si.tags, si5.tags,
                     axis=(0, 1), slicing=(slice(1, None), slice(1, None))
                     )

    # @unittest.skip("skipping test_6D_te_time_fa")
    def test_6D_te_time_fa(self):
        si = Series(
            os.path.join('data', 'dicom', '6D_TE_TIME_FA.zip'),
            'te,time,fa',
            input_format='dicom',
            opts = {'ignore_series_uid': True}
        )
        with tempfile.TemporaryDirectory() as d:
            si.write(d, formats=['dicom'])

    # @unittest.skip("skipping test_ep2d_1bvec")
    def test_ep2d_1bvec(self):
        si = Series(
            os.path.join('data', 'dicom', 'ep2d_RSI_b0_500_1500_6dir.zip'),
            'b,bvector',
            input_format='dicom'
        )
        with tempfile.TemporaryDirectory() as d:
            si.write(d, formats=['dicom'])
            si1 = Series(d, 'b,bvector', input_format='dicom')
            compare_tags(self, si.tags, si1.tags)
        tags = si.tags[0]
        for idx in np.ndindex(tags.shape):
            try:
                b, bvector = tags[idx]
            except TypeError:
                continue
            rsi = si[idx]
            if b < 1:
                self.assertEqual(len(bvector), 0)
            else:
                self.assertEqual(len(bvector), 3)

    # @unittest.skip("skipping test_ep2d_6D")
    def test_ep2d_6D(self):
        si = Series(
            os.path.join('data', 'dicom', 'RSI_6D.zip?RSI_6D/ep2d_RSI_b0_50_100_200_TE_?5'),
            'b,bvector,te',
            input_format='dicom',
            opts={'ignore_series_uid': True, 'accept_duplicate_tag': True}
        )
        with tempfile.TemporaryDirectory() as d:
            si.write(d, formats=['dicom'])
            si1 = Series(d, 'b,bvector,te', input_format='dicom')
            compare_tags(self, si.tags, si1.tags)


class TestDicomPluginSortCriteria(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

    def test_user_defined_sort(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            't',
            opts={
                't': 'InstanceNumber'
            })
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'])


class TestDicomSR(unittest.TestCase):
    def test_read_single_file(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'srdoc103.zip?srdoc103/report01.dcm'),
            'none')
        self.assertEqual(si1.input_format, 'dicom')
        self.assertEqual(len(si1.header.datasets), 1)


if __name__ == '__main__':
    unittest.main()
    # logging.basicConfig(level=logging.DEBUG)
    # runner = unittest.TextTestRunner(verbosity=2)
    # unittest.main(testRunner=runner)
