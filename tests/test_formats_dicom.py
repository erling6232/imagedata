import unittest
import math
import os.path
import tempfile
import numpy as np
import logging
import argparse
import pydicom.filereader

import src.imagedata.cmdline as cmdline
import src.imagedata.formats as formats
from src.imagedata.series import Series
from .compare_headers import compare_headers, compare_pydicom


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
        with tempfile.TemporaryDirectory(
                prefix='{}.test_cannot_sort_dicom'.format(os.getpid())) as self.d:
            # Duplicate image file
            si.write(os.path.join(self.d, '0'), formats=['dicom'], opts = {'keep_uid': True})
            si.write(os.path.join(self.d, '1'), formats=['dicom'], opts = {'keep_uid': True})
            with self.assertRaises(formats.UnknownInputError) as context:
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

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [
                os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
                os.path.join('data', 'dicom', 'time', 'time00', 'Image_00021.dcm')
            ],
            'none',
            self.opts)
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
            self.opts)
        self.assertEqual('dicom', si1.input_format)
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
        with self.assertRaises(formats.UnknownInputError) as context:
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
            opts={'ti': 'InversionTime', 'ignore_series_uid': True})
        self.assertEqual('dicom', si1.input_format)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (5, 1, 384, 384))
        with tempfile.TemporaryDirectory(prefix='test_read_dicom_user_defined_TI') as d:
            si1.write(d,
                      formats=['dicom'],
                      opts={'ti': 'InversionTime'})
            si2 = Series(d,
                         input_order='ti',
                         opts={'ti': 'InversionTime'})
        self.assertEqual('dicom', si2.input_format)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    def test_verify_correct_slice(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm'),
            'none',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        with tempfile.TemporaryDirectory(prefix='test_verify_correct_slice') as d:
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
        with tempfile.TemporaryDirectory(prefix='test_verify_correct_volume') as d:
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
        si1.axes[0].values = None
        with tempfile.TemporaryDirectory(prefix='test_verify_correct_volume_no_sliceloc') as d:
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

    def test_verify_correct_4D(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            'time',
            self.opts)
        self.assertEqual('dicom', si1.input_format)
        with tempfile.TemporaryDirectory(prefix='test_verify_correct_4D') as d:
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
        with tempfile.TemporaryDirectory(prefix='test_write_ndarray') as d:
            Series(np.eye(128)).write(d, formats=['dicom'])

    # @unittest.skip("skipping test_write_single_file")
    def test_write_single_file(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00', 'Image_00020.dcm')
        )
        self.assertEqual('dicom', si1.input_format)
        with tempfile.TemporaryDirectory(prefix='test_write_single_file') as d:
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
        with tempfile.TemporaryDirectory(prefix='test_write_single_directory') as d:
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
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual('dicom', si.input_format)
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        with tempfile.TemporaryDirectory(prefix='test_write_dicom_4D') as d:
            si.write(os.path.join(d, 'Image_{:05d}.dcm'),
                     formats=['dicom'], opts=self.opts)
            newsi = Series(d,
                           formats.INPUT_ORDER_TIME,
                           self.opts)
        self.assertEqual('dicom', newsi.input_format)
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
        with tempfile.TemporaryDirectory(prefix='test_write_dicom_4D_no_opt') as d:
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
        with tempfile.TemporaryDirectory(prefix='test_write_dicom_4D_multi_slice') as d:
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
        with tempfile.TemporaryDirectory(prefix='test_write_dicom_4D_multi_tag') as d:
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
        with tempfile.TemporaryDirectory(prefix='{}'.format(os.getpid())) as d:
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
        with tempfile.TemporaryDirectory(prefix='test_changed_uid') as d:
            eye.write(d, formats=['dicom'])
            eye_read = Series(d)
            self.assertEqual('dicom', eye_read.input_format)
        self.assertNotEqual(eye_seriesInstanceUID, eye.seriesInstanceUID)

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
                    si1.SOPInstanceUIDs[(_tag, _slice)]
                    # si1.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag)
        with tempfile.TemporaryDirectory(prefix='test_write_keep_uid') as d:
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
                    si1.SOPInstanceUIDs[(_tag, _slice)],
                    si2.SOPInstanceUIDs[(_tag, _slice)]
                )
                # si2 SOPInstanceUIDs should also be identical to original si1
                self.assertEqual(
                    si1_sopinsuid[_slice][_tag],
                    si2.SOPInstanceUIDs[(_tag, _slice)]
                )

    def test_write_no_keep_uid(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si1.input_format)
        # Make a copy of SOPInstanceUIDs before they are modified in write()
        si1_seriesInstanceUID = si1.seriesInstanceUID
        si1_sopinsuid = {}
        for _slice in range(si1.slices):
            si1_sopinsuid[_slice] = {}
            for _tag in si1.tags[0]:
                si1_sopinsuid[_slice][_tag] =\
                    si1.SOPInstanceUIDs[(_tag, _slice)]
        with tempfile.TemporaryDirectory(prefix='test_write_no_keep_uid') as d:
            si1.write(os.path.join(d, 'Image{:05d}.dcm'),
                      formats=['dicom'],
                      opts={'keep_uid': False})
            si2 = Series(d)
        self.assertEqual('dicom', si2.input_format)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        self.assertNotEqual(si1_seriesInstanceUID, si1.seriesInstanceUID)
        self.assertEqual(si1.seriesInstanceUID, si2.seriesInstanceUID)
        self.assertEqual('1.2.840.10008.5.1.4.1.1.4', si2.SOPClassUID)
        self.assertEqual(si1.slices, si2.slices)
        self.assertEqual(len(si1.tags[0]), len(si2.tags[0]))
        for _slice in range(si1.slices):
            for _tag in si1.tags[0]:
                # si2 SOPInstanceUIDs should differ from original si1
                self.assertNotEqual(
                    si1_sopinsuid[_slice][_tag],
                    si2.SOPInstanceUIDs[(_tag, _slice)]
                )

    def test_read_dicom_not_DWI(self):
        with self.assertRaises(formats.UnknownInputError) as context:
            _ = Series(
                os.path.join('data', 'dicom', 'time'),
                input_format='dicom',
                input_order='b'
            )

    def test_read_dicom_not_DWI_no_CSA(self):
        with self.assertRaises(formats.UnknownInputError) as context:
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
        with tempfile.TemporaryDirectory(
                prefix='{}.test_write_zip'.format(os.getpid())) as d:
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
        with tempfile.TemporaryDirectory(
                prefix='{}.test_read_single_file'.format(os.getpid())) as d:
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
        self.assertEqual(len(si2.tags[0]), 2)

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
        with tempfile.TemporaryDirectory(
                prefix='{}.test_user_defined_sort'.format(os.getpid())) as d:
            si1.write(d, formats=['dicom'])


if __name__ == '__main__':
    unittest.main()
    # logging.basicConfig(level=logging.DEBUG)
    # runner = unittest.TextTestRunner(verbosity=2)
    # unittest.main(testRunner=runner)
