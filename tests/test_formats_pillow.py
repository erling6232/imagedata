import unittest
import os.path
import tempfile
import numpy as np
import logging
import argparse

import imagedata.cmdline as cmdline
import imagedata.formats
import imagedata.formats as formats
from imagedata.series import Series


class Test3DPillowPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'pillow', '--serdes', '1'])

    def test_pillow_plugin(self):
        plugins = formats.get_plugins_list()
        self.pillow_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'pillow':
                self.pillow_plugin = pclass
        self.assertIsNotNone(self.pillow_plugin)

    def test_read_single_file(self):
        si1 = Series(os.path.join('data', 'lena_color.jpg'), input_format='pillow')
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        self.assertEqual(si1.dtype, rgb_dtype)
        self.assertEqual(si1.shape, (512, 512))

    def test_write_pdf(self):
        si1 = Series(os.path.join('data', 'lena_color.jpg'), input_format='pillow')
        with tempfile.TemporaryDirectory() as d:
            filename = os.path.join(d, 'Image.pdf')
            si1.write(filename, formats='pdf')
            if not os.path.isfile(filename):
                raise AssertionError('File does not exist: {}'.format(filename))

    def test_write_pdf_multiple(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            dtype=np.uint8,
            input_format='dicom')
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats='pdf')
            filename = os.path.join(d, 'Image_00000.pdf')
            if not os.path.isfile(filename):
                raise AssertionError('File does not exist: {}'.format(filename))

    def test_write_int64_fails(self):
        with tempfile.TemporaryDirectory() as d:
            si = Series(np.eye(128), dtype=np.int64)
            with self.assertRaises(OSError):
                si.write(os.path.join(d, 'float.tiff'), formats='tiff')

    def test_write_int32_tiff_single_file(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128), dtype=np.int32).write(os.path.join(d, 'int32'), formats='tiff')
            si = Series(d, input_format='pillow')
        self.assertEqual(si.dtype, np.int32)

    def test_write_int32_tiff_directory(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128), dtype=np.int32).write(d, formats='tiff')
            si = Series(d, input_format='pillow')
            filename = os.path.join(d, 'Image_00000.tiff')
            if not os.path.isfile(filename):
                raise AssertionError('File does not exist: {}'.format(filename))
        self.assertEqual(si.dtype, np.int32)

    def test_write_uint16_pillow(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128), dtype=np.uint16).write(os.path.join(d, 'pillow.png'), formats='pillow')
            si = Series(d, input_format='pillow')
        self.assertEqual(si.dtype, np.uint16)

    def test_write_uint16_pillow_png(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128), dtype=np.uint16).write(os.path.join(d, 'pillow'), formats='png')
            si = Series(d, input_format='pillow')
        self.assertEqual(si.dtype, np.uint16)

    def test_write_float32(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128), dtype=np.float32).write(os.path.join(d, 'float.tiff'), formats='tiff')
            si = Series(d, input_format='pillow')
        self.assertEqual(si.dtype, np.float32)

    def test_write_bool(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128), dtype=bool).write(os.path.join(d, 'bool.tiff'), formats='tiff')
            si = Series(d, input_format='pillow')
        self.assertEqual(si.dtype, bool)
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128), dtype=bool).write(os.path.join(d, 'float.png'), formats='png')
            si = Series(d, input_format='pillow')
        self.assertEqual(si.dtype, bool)

    def test_write_single_file(self):
        si1 = Series(os.path.join('data', 'lena_color.jpg'), input_format='pillow')
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image00000.png'), formats='png')
            si2 = Series(os.path.join(d, 'Image00000.png'), input_format='pillow')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    def test_write_single_volume(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            input_format='dicom')
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats='png')
            si2 = Series(d, input_format='pillow')
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)


class Test4DMatPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'pillow', '--serdes', '1'])

        plugins = formats.get_plugins_list()
        self.pillow_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'pillow':
                self.pillow_plugin = pclass
        self.assertIsNotNone(self.pillow_plugin)

    # @unittest.skip("skipping test_write_4d_mat")
    def test_write_4d(self):
        si1 = Series(
            os.path.join('data', 'dicom', 'time'),
            'time',
            dtype=np.uint8,
            input_format='dicom')
        self.assertEqual(si1.dtype, np.uint8)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image_{0:05d}.tiff'),
                      formats='tiff', opts=self.opts)

            # Read back the PNG data and compare to original si1
            si2 = Series(d, 'time', input_format='pillow',
                         input_shape=(3, 3))
            si3 = Series(d, 'time', input_format='pillow',
                         input_shape='3x3')
            si4 = Series(d, 'time', input_format='pillow',
                         input_shape=si1.shape)
            with self.assertRaises(imagedata.formats.UnknownInputError):
                _ = Series(d, 'time', input_format='pillow',
                           input_shape='3xb')
            with self.assertRaises(imagedata.formats.UnknownInputError):
                _ = Series(d, 'time', input_format='pillow',
                             input_shape='3')
            with self.assertRaises(imagedata.formats.UnknownInputError):
                _ = Series(d, 'none', input_format='pillow',
                           input_shape='3x3')
            with self.assertRaises(imagedata.formats.UnknownInputError):
                _ = Series(d, 'time,te', input_format='pillow',
                           input_shape='3x3')
        for trial in (si2, si3, si4):
            self.assertEqual(si1.shape, trial.shape)
            self.assertEqual(si1.dtype, trial.dtype)
            np.testing.assert_array_equal(si1, trial)


if __name__ == '__main__':
    unittest.main()
