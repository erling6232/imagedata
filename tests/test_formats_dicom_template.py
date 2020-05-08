#!/usr/bin/env python3

import unittest
import os.path
import numpy as np
import tempfile
import argparse

from .context import imagedata
import imagedata.cmdline
import imagedata.readdata
import imagedata.formats
from imagedata.series import Series
from .compare_headers import compare_headers, compare_template_headers, compare_geometry_headers


class ShouldHaveFailed(Exception):
    pass


class TestDicomTemplate(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'dicom'])
        self.opts_template = parser.parse_args(['--of', 'dicom',
                                                '--input_options', 'AcceptDuplicateTag=True',
                                                '--template', 'data/dicom/time/time00/'])
        self.opts_geometry = parser.parse_args(['--of', 'dicom',
                                                '--input_options', 'AcceptDuplicateTag=True',
                                                '--geometry', 'data/dicom/time/time00/'])
        self.opts_tempgeom = parser.parse_args(['--of', 'dicom',
                                                '--input_options', 'AcceptDuplicateTag=True',
                                                '--template', 'data/dicom/time/time00/',
                                                '--geometry', 'data/dicom/time/time00/'])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

        # Create a DICOM series with empty header
        si0 = Series(os.path.join('data', 'biff', 'time', 'time00.biff'))
        self.emptydir = tempfile.TemporaryDirectory()
        si0.write(os.path.join(self.emptydir.name, 'empty_header'), formats=['dicom'])

    def tearDown(self):
        self.emptydir.cleanup()
        self.emptydir = None

    # @unittest.skip("skipping test_dicom_template_cmdline")
    def test_dicom_template_cmdline(self):
        # Read the DICOM empty header series,
        # adding DICOM template
        si1 = Series(
            os.path.join(self.emptydir.name, 'empty_header'),
            'none',
            self.opts_template)
        # Read the original DICOM series
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_template_headers(self, si1, si2)
        try:
            compare_geometry_headers(self, si1, si2)
        except AssertionError:
            # Expected to fail
            pass
        else:
            raise ShouldHaveFailed('Template header should differ when joining geometry')
        # Write constructed series si1 to disk,
        # then re-read and compare to original si2
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'])
            si3 = Series(d, 'none', self.opts)
        np.testing.assert_array_equal(si2, si3)
        compare_template_headers(self, si2, si3)

    # @unittest.skip("skipping test_dicom_template_prog")
    def test_dicom_template_prog(self):
        # Read the DICOM empty header series,
        # adding DICOM template in Series constructor
        template = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        si1 = Series(
            os.path.join(self.emptydir.name, 'empty_header'),
            template=template)
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_template_headers(self, si1, si2)

    # @unittest.skip("skipping test_dicom_geometry_cmdline")
    def test_dicom_geometry_cmdline(self):
        # Read the DICOM empty header series,
        # adding DICOM geometry
        si1 = Series(
            os.path.join(self.emptydir.name, 'empty_header'),
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
        except AssertionError:
            # Expected to fail
            pass
        else:
            raise ShouldHaveFailed('Template header should differ when joining geometry')
        # Write constructed series si1 to disk,
        # then re-read and compare to original si2
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'])
            si3 = Series(d, 'none', self.opts)
        np.testing.assert_array_equal(si2, si3)
        compare_geometry_headers(self, si2, si2)

    # @unittest.skip("skipping test_dicom_geometry_prog")
    def test_dicom_geometry_prog(self):
        # Read the DICOM empty header series,
        # adding DICOM geometry in Series constructor
        geometry = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        si1 = Series(
            os.path.join(self.emptydir.name, 'empty_header'),
            geometry=geometry)
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_geometry_headers(self, si1, si2)
        try:
            compare_template_headers(self, si1, si2)
        except AssertionError:
            # Excpected to fail
            pass
        else:
            raise ShouldHaveFailed('Template header should differ when joining geometry')

    # @unittest.skip("skipping test_dicom_tempgeom_cmdline")
    def test_dicom_tempgeom_cmdline(self):
        # Read the DICOM empty header series,
        # adding DICOM template
        si1 = Series(
            os.path.join(self.emptydir.name, 'empty_header'),
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
            si3 = Series(d, 'none', self.opts)
        np.testing.assert_array_equal(si2, si3)
        compare_headers(self, si2, si3)

    # @unittest.skip("skipping test_dicom_tempgeom_prog")
    def test_dicom_tempgeom_prog(self):
        # Read the DICOM empty header series,
        # adding DICOM template in Series constructor
        template = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        geometry = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        si1 = Series(
            os.path.join(self.emptydir.name, 'empty_header'),
            template=template,
            geometry=geometry)
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_headers(self, si1, si2)


if __name__ == '__main__':
    unittest.main()
