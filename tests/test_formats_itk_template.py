#!/usr/bin/env python3

import unittest
import os.path
import tempfile
import numpy as np
import argparse

# from .context import imagedata
import src.imagedata.cmdline as cmdline
import src.imagedata.formats as formats
from src.imagedata.series import Series
from .compare_headers import compare_headers, compare_template_headers, compare_geometry_headers


class ShouldHaveFailed(Exception):
    pass


class TestItkTemplate(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts_template = parser.parse_args(['--of', 'itk',
                                                '--template', 'data/dicom/time/time00/'])
        self.opts_geometry = parser.parse_args(['--of', 'itk',
                                                '--geometry', 'data/dicom/time/time00/'])
        self.opts_tempgeom = parser.parse_args(['--of', 'itk',
                                                '--template', 'data/dicom/time/time00/',
                                                '--geometry', 'data/dicom/time/time00/'])

        plugins = formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'itk':
                self.itk_plugin = pclass
        self.assertIsNotNone(self.itk_plugin)

        self.template = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.geometry = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.geometry.spacing = (3, 2, 1)

    # @unittest.skip("skipping test_itk_template_cmdline")
    def test_itk_template_cmdline(self):
        # Read the ITK series, adding DICOM template
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            'none',
            self.opts_template)
        # Read the original DICOM series
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_template_headers(self, si1, si2)
        # Write constructed series si1 to disk,
        # then re-read and compare to original si2
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'])
            si3 = Series(d)
        # np.testing.assert_array_equal(si2, si3)
        compare_template_headers(self, si2, si3)

    # @unittest.skip("skipping test_itk_template_prog")
    def test_itk_template_prog(self):
        # Read the ITK series, adding DICOM template
        # adding DICOM template in Series constructor
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            template=self.template)
        # Compare constructed series si1 to original series
        self.assertEqual(si1.dtype, self.template.dtype)
        np.testing.assert_array_equal(si1, self.template)
        compare_template_headers(self, si1, self.template)

    # @unittest.skip("skipping test_itk_geometry_cmdline")
    def test_itk_geometry_cmdline(self):
        # Read the ITK series, adding DICOM geometry
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            'none',
            self.opts_geometry)
        # Read the original DICOM series
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_geometry_headers(self, si1, si2)
        try:
            compare_template_headers(self, si1, si2)
        except (ValueError, AssertionError):
            # Expected to fail
            pass
        else:
            raise ShouldHaveFailed('Template header should differ when joining geometry')
        # Write constructed series si1 to disk,
        # then re-read and compare to original si2
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'])
            si3 = Series(d)
        np.testing.assert_array_equal(si2, si3)
        compare_geometry_headers(self, si2, si2)

    # @unittest.skip("skipping test_itk_geometry_prog")
    def test_itk_geometry_prog(self):
        # Read the ITK series,
        # adding DICOM geometry in Series constructor
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            geometry=self.geometry)
        # Compare constructed series si1 to original series
        self.assertEqual(si1.dtype, self.geometry.dtype)
        np.testing.assert_array_equal(si1, self.geometry)
        compare_geometry_headers(self, si1, self.geometry)
        try:
            compare_template_headers(self, si1, self.geometry)
        except (ValueError, AssertionError):
            # Excpected to fail
            pass
        else:
            raise ShouldHaveFailed('Template header should differ when joining geometry')

    # @unittest.skip("skipping test_itk_tempgeom_cmdline")
    def test_itk_tempgeom_cmdline(self):
        # Read the ITK series, adding DICOM template and geometry
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            'none',
            self.opts_tempgeom)
        # Read the original DICOM series
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_headers(self, si1, si2)
        # Write constructed series si1 to disk,
        # then re-read and compare to original si2
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'])
            si3 = Series(d)
        np.testing.assert_array_equal(si2, si3)
        compare_headers(self, si2, si3)

    # @unittest.skip("skipping test_itk_tempgeom_prog")
    def test_itk_tempgeom_prog(self):
        # Read the ITK series,
        # adding DICOM template in Series constructor
        si1 = Series(
            os.path.join('data', 'itk', 'time', 'Image_00000.mha'),
            template=self.template,
            geometry=self.geometry)
        # Compare constructed series si1 to original series
        self.assertEqual(si1.dtype, self.template.dtype)
        np.testing.assert_array_equal(si1, self.template)
        compare_geometry_headers(self, si1, self.geometry)
        compare_template_headers(self, si1, self.template)


if __name__ == '__main__':
    unittest.main()
