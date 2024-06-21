#!/usr/bin/env python3

import unittest
import os.path
from datetime import datetime, timezone
import numpy as np
import tempfile
import argparse

# from .context import imagedata
import src.imagedata.cmdline as cmdline
import src.imagedata.formats as formats
from src.imagedata.series import Series
from .compare_headers import compare_headers, compare_template_headers, compare_geometry_headers


class ShouldHaveFailed(Exception):
    pass


class TestDicomTemplate(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

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

        plugins = formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

    def getEmpty(self, prefix=None):
        # Create a DICOM series with empty header
        si0 = Series(os.path.join('data', 'mat', 'time', 'Image_00000.mat'), input_order='time')
        emptydir = tempfile.TemporaryDirectory(prefix='{}.{}'.format(os.getpid(), prefix))
        si00 = Series(si0[0], input_order='none')
        si00.write(os.path.join(emptydir.name, 'empty_header'), formats=['dicom'])

        # Provide sensible time tags
        si01 = Series(si0[:2], input_order='time')
        for s in range(3):
            for t in range(2):
                time_str = datetime.fromtimestamp(float(t), timezone.utc).strftime("%H%M%S.%f")
                si01.setDicomAttribute('AcquisitionTime', time_str, slice=s, tag=t)
        si01.write(os.path.join(emptydir.name, 'empty_header_time'), formats=['dicom'])
        return emptydir

    def dropEmpty(self, emptydir):
        emptydir.cleanup()

    # @unittest.skip("skipping test_dicom_template_cmdline")
    def test_dicom_template_cmdline(self):
        # Read the DICOM empty header series,
        # adding DICOM template
        emptydir = self.getEmpty('template_cmdline')
        si1 = Series(
            os.path.join(emptydir.name, 'empty_header'),
            'none',
            self.opts_template)
        self.assertEqual('dicom', si1.input_format)
        # Read the original DICOM series
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si2.input_format)
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
        with tempfile.TemporaryDirectory(
                prefix='{}.{}'.format(os.getpid(),'template_cmdline_si1')) as d:
            si1.write(d, formats=['dicom'])
            si3 = Series(d, 'none', self.opts)
        self.assertEqual('dicom', si3.input_format)
        np.testing.assert_array_equal(si2, si3)
        compare_template_headers(self, si2, si3)
        self.dropEmpty(emptydir)

    # @unittest.skip("skipping test_dicom_template_prog")
    def test_dicom_template_prog(self):
        # Read the DICOM empty header series,
        # adding DICOM template in Series constructor
        emptydir = self.getEmpty('template_prog')
        template = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', template.input_format)
        si1 = Series(
            os.path.join(emptydir.name, 'empty_header'),
            template=template)
        self.assertEqual('dicom', si1.input_format)
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si2.input_format)
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_template_headers(self, si1, si2)
        self.dropEmpty(emptydir)

    # @unittest.skip("skipping test_dicom_geometry_cmdline")
    def test_dicom_geometry_cmdline(self):
        # Read the DICOM empty header series,
        # adding DICOM geometry
        emptydir = self.getEmpty('geometry_cmdline')
        si1 = Series(
            os.path.join(emptydir.name, 'empty_header'),
            'none',
            self.opts_geometry)
        self.assertEqual('dicom', si1.input_format)
        # Read the original DICOM series
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si2.input_format)
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
        with tempfile.TemporaryDirectory(
                prefix='{}.geometry_cmdline_si1'.format(os.getpid())) as d:
            si1.write(d, formats=['dicom'])
            si3 = Series(d, 'none', self.opts)
        self.assertEqual('dicom', si3.input_format)
        np.testing.assert_array_equal(si2, si3)
        compare_geometry_headers(self, si2, si2)
        self.dropEmpty(emptydir)

    # @unittest.skip("skipping test_dicom_geometry_prog")
    def test_dicom_geometry_prog(self):
        # Read the DICOM empty header series,
        # adding DICOM geometry in Series constructor
        emptydir = self.getEmpty('geometry_prog')
        geometry = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', geometry.input_format)
        si1 = Series(
            os.path.join(emptydir.name, 'empty_header'),
            geometry=geometry)
        self.assertEqual('dicom', si1.input_format)
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si2.input_format)
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
        self.dropEmpty(emptydir)

    # @unittest.skip("skipping test_dicom_tempgeom_cmdline")
    def test_dicom_tempgeom_cmdline(self):
        # Read the DICOM empty header series,
        # adding DICOM template
        emptydir = self.getEmpty('tempgeom_cmdline')
        si1 = Series(
            os.path.join(emptydir.name, 'empty_header'),
            'none',
            self.opts_tempgeom)
        self.assertEqual('dicom', si1.input_format)
        # Read the original DICOM series
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si2.input_format)
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_headers(self, si1, si2)
        # Write constructed series si1 to disk,
        # then re-read and compare to original si2
        with tempfile.TemporaryDirectory(
                prefix='{}.tempgeom_cmdline_si1'.format(os.getpid())) as d:
            si1.write(d, formats=['dicom'])
            si3 = Series(d, 'none', self.opts)
        self.assertEqual('dicom', si3.input_format)
        np.testing.assert_array_equal(si2, si3)
        compare_headers(self, si2, si3)
        self.dropEmpty(emptydir)

    # @unittest.skip("skipping test_dicom_tempgeom_prog")
    def test_dicom_tempgeom_prog(self):
        # Read the DICOM empty header series,
        # adding DICOM template in Series constructor
        emptydir = self.getEmpty('tempgeom_prog1')
        template = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', template.input_format)
        geometry = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', geometry.input_format)
        si1 = Series(
            os.path.join(emptydir.name, 'empty_header'),
            template=template,
            geometry=geometry)
        self.assertEqual('dicom', si1.input_format)
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si2.input_format)
        # Compare constructed series si1 to original series si2
        self.assertEqual(si1.dtype, si2.dtype)
        np.testing.assert_array_equal(si1, si2)
        compare_headers(self, si1, si2)
        self.dropEmpty(emptydir)

    # @unittest.skip("skipping test_dicom_temp_slice")
    def test_dicom_temp_slice(self):
        # Read the DICOM empty header series,
        # adding DICOM template in Series constructor
        # Then slice the si1 Series
        emptydir = self.getEmpty('temp_slice')
        template = Series(os.path.join('data', 'dicom', 'time'), input_order='time')
        self.assertEqual('dicom', template.input_format)
        geometry = Series(os.path.join('data', 'dicom', 'time'), input_order='time')
        self.assertEqual('dicom', geometry.input_format)
        si1 = Series(
            os.path.join(emptydir.name, 'empty_header_time'),
            input_order='time',
            template=template,
            geometry=geometry)
        self.assertEqual('dicom', si1.input_format)
        si1_0 = si1[0]
        si2 = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', si2.input_format)
        # Compare constructed series si1_0 to original series si2
        self.assertEqual(si1_0.dtype, si2.dtype)
        np.testing.assert_array_equal(si1_0, si2)
        # Tags do not match, so copy them to enable header comparison.
        si1_0.tags = si2.tags
        compare_headers(self, si1_0, si2)
        self.dropEmpty(emptydir)


class TestDicomGeometryTemplate(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

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

        plugins = formats.get_plugins_list()
        self.dicom_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'dicom':
                self.dicom_plugin = pclass
        self.assertIsNotNone(self.dicom_plugin)

    def test_dicom_too_many_template_slices(self):
        # Construct simple series,
        # add DICOM template in Series constructor
        template = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', template.input_format)
        si1 = Series(
            np.zeros((2, 192, 152)),
            geometry=template)
        # Compare constructed series si1 to original series template
        np.testing.assert_array_equal(template.sliceLocations[:2], si1.sliceLocations)

    def test_dicom_too_few_template_slices(self):
        # Construct simple series,
        # add DICOM template in Series constructor
        template = Series(os.path.join('data', 'dicom', 'time', 'time00'))
        self.assertEqual('dicom', template.input_format)
        # Add an extra slice
        shape = (template.shape[0]+1, template.shape[1], template.shape[2])
        si1 = Series(np.zeros(shape), geometry=template)
        # Compare constructed series si1 to original series template
        # Append one slice location to template
        ds = template.sliceLocations[1] - template.sliceLocations[0]
        ns = template.sliceLocations[-1] + ds
        template_list = template.sliceLocations.tolist()
        template_list.append(ns)
        template_locations = np.array(template_list)
        np.testing.assert_array_equal(template_locations, si1.sliceLocations)

    def test_dicom_too_many_template_tags(self):
        # Construct simple series,
        # add DICOM template in Series constructor
        template = Series(os.path.join('data', 'dicom', 'time'), 'time')
        self.assertEqual('dicom', template.input_format)
        si1 = Series(
            np.zeros((2, 3, 192, 152)),
            template=template, geometry=template)
        # Compare constructed series si1 to original series template
        for _slice in range(3):
            np.testing.assert_array_equal(template.tags[_slice][:2], si1.tags[_slice])

    def test_dicom_too_few_template_tags(self):
        # Construct simple series,
        # add DICOM template in Series constructor
        template = Series(os.path.join('data', 'dicom', 'time'), 'time')
        self.assertEqual('dicom', template.input_format)
        si1 = Series(
            np.zeros((template.shape[0]+1, template.shape[1],
                      template.shape[2], template.shape[3])),
            'time',
            template=template, geometry=template)
        # Compare constructed series si1 to original series template
        for _slice in range(si1.slices):
            # Append one slice location to template
            ds = template.tags[_slice][1] - template.tags[_slice][0]
            ns = template.tags[_slice][-1] + ds
            template_list = template.tags[_slice].tolist()
            template_list.append(ns)
            template_tags = np.array(template_list)
            np.testing.assert_array_almost_equal(template_tags, si1.tags[_slice], decimal=4)


if __name__ == '__main__':
    unittest.main()
