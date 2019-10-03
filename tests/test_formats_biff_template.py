#!/usr/bin/env python3

import unittest
import sys
import shutil
import numpy as np
import logging
import argparse

from .context import imagedata
import imagedata.cmdline
import imagedata.readdata
import imagedata.formats
from imagedata.series import Series
from .compare_headers import compare_headers, compare_template_headers, compare_geometry_headers

class ShouldHaveFailed(Exception): pass

class test_biff_template(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts_template = parser.parse_args(['--of', 'biff',
            '--template', 'data/dicom/time/time00/'])
        self.opts_geometry = parser.parse_args(['--of', 'biff',
            '--geometry', 'data/dicom/time/time00/'])
        self.opts_tempgeom = parser.parse_args(['--of', 'biff',
            '--template', 'data/dicom/time/time00/',
            '--geometry', 'data/dicom/time/time00/'])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'biff': self.biff_plugin = pclass
        self.assertIsNotNone(self.biff_plugin)

        self.template=Series('data/dicom/time/time00/')
        self.geometry=Series('data/dicom/time/time00/')
        self.geometry.spacing = (3, 2, 1)

    def tearDown(self):
        shutil.rmtree('ttbt', ignore_errors=True)

    #@unittest.skip("skipping test_biff_template_cmdline")
    def test_biff_template_cmdline(self):
        # Read the BIFF series, adding DICOM template
        si1 = Series(
                'data/biff/time/time00.biff',
                'none',
                self.opts_template)
        # Read the original DICOM series
        si2 = Series('data/dicom/time/time00/')
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_template_headers(self, si1, si2)
        # Write constructed series si1 to disk,
        # then re-read and compare to original si2
        si1.write('ttbt', formats=['dicom'])
        si3 = Series('ttbt')
        np.testing.assert_array_equal(si2, si3)
        compare_template_headers(self, si2, si3)

    #@unittest.skip("skipping test_biff_template_prog")
    def test_biff_template_prog(self):
        # Read the BIFF series, adding DICOM template
        # adding DICOM template in Series constructor
        si1 = Series(
            'data/biff/time/time00.biff',
            template=self.template)
        # Compare constructed series si1 to original series
        self.assertEqual(si1.dtype, self.template.dtype)
        np.testing.assert_array_equal(si1, self.template)
        compare_template_headers(self, si1, self.template)

    #@unittest.skip("skipping test_biff_geometry_cmdline")
    def test_biff_geometry_cmdline(self):
        # Read the BIFF series, adding DICOM geometry
        si1 = Series(
                'data/biff/time/time00.biff',
                'none',
                self.opts_geometry)
        # Read the original DICOM series
        si2 = Series('data/dicom/time/time00/')
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
        si1.write('ttbt', formats=['dicom'])
        si3 = Series('ttbt')
        np.testing.assert_array_equal(si2, si3)
        compare_geometry_headers(self, si2, si2)

    #@unittest.skip("skipping test_biff_geometry_prog")
    def test_biff_geometry_prog(self):
        # Read the BIFF series,
        # adding DICOM geometry in Series constructor
        si1 = Series(
                'data/biff/time/time00.biff',
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

    #@unittest.skip("skipping test_biff_tempgeom_cmdline")
    def test_biff_tempgeom_cmdline(self):
        # Read the BIFF series, adding DICOM template and geometry
        si1 = Series(
            'data/biff/time/time00.biff',
            'none',
            self.opts_tempgeom)
        # Read the original DICOM series
        si2 = Series('data/dicom/time/time00/')
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_headers(self, si1, si2)
        # Write constructed series si1 to disk,
        # then re-read and compare to original si2
        si1.write('ttbt', formats=['dicom'])
        si3 = Series('ttbt')
        np.testing.assert_array_equal(si2, si3)
        compare_headers(self, si2, si3)

    #@unittest.skip("skipping test_biff_tempgeom_prog")
    def test_biff_tempgeom_prog(self):
        # Read the BIFF series,
        # adding DICOM template and geometry in Series constructor
        si1 = Series(
                'data/biff/time/time00.biff',
                template=self.template,
                geometry=self.geometry)
        # Compare constructed series si1 to original series
        self.assertEqual(si1.dtype, self.template.dtype)
        np.testing.assert_array_equal(si1, self.template)
        compare_geometry_headers(self, si1, self.geometry)
        compare_template_headers(self, si1, self.template)

if __name__ == '__main__':
    unittest.main()
