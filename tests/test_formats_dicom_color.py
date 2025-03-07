#!/usr/bin/env python3

import unittest
import os.path
import numpy as np
import argparse
import tempfile
# from .context import imagedata
import src.imagedata.cmdline as cmdline
import src.imagedata.formats as formats
from src.imagedata.series import Series


class TestDicomColor(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'dicom', '--serdes', '1'])
        self.opts_gray = parser.parse_args(['--of', 'dicom', '--serdes', '1',
                                            '--psopt', 'pnggray'])

        plugins = formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

    # @unittest.skip("skipping test_read_dicom_color")
    def test_read_dicom_color(self):
        si1 = Series(os.path.join('data', 'lena_color.jpg'))
        si2 = Series(os.path.join('data', 'dicom', 'lena_color.dcm'))
        self.assertEqual('dicom', si2.input_format)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)
        self.assertEqual(si1.color, si2.color)
        np.testing.assert_array_equal(si1, si2)

    # @unittest.skip("skipping test_write_dicom_color")
    def test_write_dicom_color(self):
        si1 = Series(
            os.path.join('data', 'lena_color.jpg'),
            'none',
            self.opts)
        self.assertEqual('itk', si1.input_format)
        self.assertEqual(si1.dtype, np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]))
        self.assertEqual(si1.shape, (512, 512))
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image.dcm'), formats=['dicom'])
            si2 = Series(d)
            self.assertEqual('dicom', si2.input_format)
        d = si1 - si2
        np.testing.assert_array_equal(si1, si2)


if __name__ == '__main__':
    unittest.main()
