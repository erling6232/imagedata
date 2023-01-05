#!/usr/bin/env python3

import unittest
import copy
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

    def test_dicom_plugin(self):
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

    def test_read_single_directory_headers_only(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            opts={'headers_only': True})
        self.assertEqual(tuple(), si1.shape)
        self.assertEqual(3, len(si1.axes))
        self.assertEqual(14, si1.seriesNumber)

    def test_read_auto_volume(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'auto',
            self.opts)
        self.assertEqual(imagedata.formats.INPUT_ORDER_NONE, si1.input_order)
        self.assertEqual((3, 192, 152), si1.shape)
        self.assertEqual(3, len(si1.axes))
        self.assertEqual(14, si1.seriesNumber)

    def test_read_auto_time(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            'auto',
            self.opts)
        self.assertEqual(imagedata.formats.INPUT_ORDER_TIME, si1.input_order)
        self.assertEqual((3, 3, 192, 152), si1.shape)
        self.assertEqual(4, len(si1.axes))
        self.assertEqual(14, si1.seriesNumber)

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

    # @unittest.skip("skipping test_read_dicom_4D_wrong_order")
    def test_read_dicom_4D_wrong_order(self):
        with self.assertRaises(imagedata.formats.CannotSort) as context:
            si1 = Series(
                os.path.join('data', 'dicom', 'time'),
                'none',
                self.opts)

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

    def test_write_ndarray(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128)).write(d, formats=['dicom'])

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

    def test_write_keep_uid(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        # Make a copy of SOPInstanceUIDs before they are possibly modified in write()
        si1_sopinsuid = {}
        for _slice in range(si1.slices):
            si1_sopinsuid[_slice] = {}
            for _tag in si1.tags[0]:
                si1_sopinsuid[_slice][_tag] = \
                    si1.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag)
        with tempfile.TemporaryDirectory() as d:
            si1.write('{}?Image%05d.dcm'.format(d),
                      formats=['dicom'],
                      opts={'keep_uid': True})
            si2 = Series(d)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        self.assertEqual(si1.seriesInstanceUID, si2.seriesInstanceUID)
        self.assertEqual('1.2.840.10008.5.1.4.1.1.4', si2.SOPClassUID)
        self.assertEqual(si1.slices, si2.slices)
        self.assertEqual(len(si1.tags[0]), len(si2.tags[0]))
        for _slice in range(si1.slices):
            for _tag in si1.tags[0]:
                # si1 SOPInstanceUIDs should be identical to si2
                self.assertEqual(
                    si1.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag),
                    si2.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag)
                )
                # si2 SOPInstanceUIDs should also be identical to original si1
                self.assertEqual(
                    si1_sopinsuid[_slice][_tag],
                    si2.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag)
                )

    def test_write_no_keep_uid(self):
        si1 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        # Make a copy of SOPInstanceUIDs before they are modified in write()
        si1_sopinsuid = {}
        for _slice in range(si1.slices):
            si1_sopinsuid[_slice] = {}
            for _tag in si1.tags[0]:
                si1_sopinsuid[_slice][_tag] =\
                    si1.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag)
        with tempfile.TemporaryDirectory() as d:
            si1.write('{}?Image%05d.dcm'.format(d),
                      formats=['dicom'],
                      opts={'keep_uid': False})
            si2 = Series(d)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        self.assertNotEqual(si1.seriesInstanceUID, si2.seriesInstanceUID)
        self.assertEqual('1.2.840.10008.5.1.4.1.1.4', si2.SOPClassUID)
        self.assertEqual(si1.slices, si2.slices)
        self.assertEqual(len(si1.tags[0]), len(si2.tags[0]))
        for _slice in range(si1.slices):
            for _tag in si1.tags[0]:
                # si1 SOPInstanceUIDs where modified at write() and will be identical to si2
                self.assertEqual(
                    si1.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag),
                    si2.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag)
                )
                # si2 SOPInstanceUIDs should differ from original si1
                self.assertNotEqual(
                    si1_sopinsuid[_slice][_tag],
                    si2.getDicomAttribute('SOPInstanceUID', slice=_slice, tag=_tag)
                )


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

    # @unittest.skip("skipping test_read_dicom_not_DWI")
    def test_read_dicom_not_DWI(self):
        with self.assertRaises(imagedata.formats.CannotSort) as context:
            d = Series(
                os.path.join('data', 'dicom', 'time'),
                'b'
            )

    # @unittest.skip("skipping test_read_dicom_not_DWI_no_CSA")
    def test_read_dicom_not_DWI_no_CSA(self):
        with self.assertRaises(imagedata.formats.CannotSort) as context:
            d = Series(
                os.path.join('data', 'dicom', 'lena_color.dcm'),
                'b'
            )


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
