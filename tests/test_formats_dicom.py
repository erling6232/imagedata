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
from .compare_headers import compare_headers

class TestDicomPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #sys.argv[1:] = ['aa', 'bb']
        #self.opts = parser.parse_args(['--order', 'none'])
        self.opts = parser.parse_args([])

        plugins = imagedata.formats.get_plugins_list()
        self.dicom_plugin = None
        for pname,ptype,pclass in plugins:
            if ptype == 'dicom': self.dicom_plugin = pclass
        self.assertIsNotNone(pclass)

    def tearDown(self):
        try:
            shutil.rmtree('ttd3d')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('ttd4d')
        except FileNotFoundError:
            pass

    #@unittest.skip("skipping test_read_dicom_3D")
    def test_read_dicom_3D(self):
        d=Series(
                'tests/dicom/NYRE_151204_T1/_fl3d1_0005',
                0,
                self.opts)
        #logging.debug("test_series_dicom: type(d) {}".format(type(d)))
        #logging.debug("Series.imageType {}".format(d.imageType))
        #logging.debug("Series.dir() {}".format(dir(d)))
        self.assertEqual(d.dtype, np.uint16)
        self.assertEqual(d.shape, (30,192,192))

    #@unittest.skip("skipping test_read_dicom_3D_no_opt")
    def test_read_dicom_3D_no_opt(self):
        d=Series(
                'tests/dicom/NYRE_151204_T1/_fl3d1_0005')
        #logging.debug("test_series_dicom: type(d) {}".format(type(d)))
        #logging.debug("Series.imageType {}".format(d.imageType))
        #logging.debug("Series.dir() {}".format(dir(d)))
        self.assertEqual(d.dtype, np.uint16)
        self.assertEqual(d.shape, (30,192,192))

    #@unittest.skip("skipping test_read_dicom_4D")
    def test_read_dicom_4D(self):
        si1 = Series(
                'tests/dicom/NYRE_151204_T1',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (8, 30, 192, 192))
        t = np.array([   0.  ,   66.81,  105.24,  142.24,  173.25,  194.3 ,  216.24, 237.27])
        np.testing.assert_array_almost_equal(t, si1.timeline, decimal=2)

    #@unittest.skip("skipping test_copy_dicom_4D")
    def test_copy_dicom_4D(self):
        si = Series(
                'tests/dicom/NYRE_151204_T1',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        #si.sort_on = imagedata.formats.SORT_ON_SLICE
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        newsi  = Series(si,
                imagedata.formats.INPUT_ORDER_TIME)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (8, 30, 192, 192))

    #@unittest.skip("skipping test_write_dicom_4D")
    def test_write_dicom_4D(self):
        si = Series(
                #'tests/dicom/NYRE_151204_T1',
                'tests/dicom/dynamic',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        #si.sort_on = imagedata.formats.SORT_ON_SLICE
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        si.write('ttd4d/dicom', 'Image_%05d', formats=['dicom'], opts=self.opts)
        newsi  = Series('ttd4d',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (8, 30, 192, 192))

    #@unittest.skip("skipping test_write_dicom_4D_no_opt")
    def test_write_dicom_4D_no_opt(self):
        si = Series(
                'tests/dicom/NYRE_151204_T1',
                imagedata.formats.INPUT_ORDER_TIME,
                self.opts)
        #si.sort_on = imagedata.formats.SORT_ON_SLICE
        logging.debug("si.sliceLocations: {}".format(si.sliceLocations))
        logging.debug("si.imagePositions.keys(): {}".format(si.imagePositions.keys()))
        #si.write('ttd4d/dicom', 'Image_%05d', formats=['dicom'], opts=self.opts)
        si.write('ttd4d/dicom', 'Image_%05d', formats=['dicom'])
        newsi  = Series('ttd4d',
                imagedata.formats.INPUT_ORDER_TIME)
        self.assertEqual(si.shape, newsi.shape)
        np.testing.assert_array_equal(si, newsi)
        compare_headers(self, si, newsi)
        self.assertEqual(newsi.dtype, np.uint16)
        self.assertEqual(newsi.shape, (8, 30, 192, 192))

if __name__ == '__main__':
    unittest.main()
