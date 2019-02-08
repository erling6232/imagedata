#!/usr/bin/env python3

"""Test file archive
"""

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

class test_file_archive_itk(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #self.opts = parser.parse_args(['--template', 'tests/dicom/time/',
        #        '--geometry', 'tests/dicom/time'])
        #self.opts = parser.parse_args(['--of', 'itk', '--sort', 'tag'])
        #self.opts = parser.parse_args(['--of', 'itk'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['itk']

        #self.hdr,self.si = imagedata.readdata.read(
        #        ('tests/dicom/time/',),
        #        imagedata.formats.INPUT_ORDER_TIME,
        #        self.opts)

    def tearDown(self):
        try:
            shutil.rmtree('tti3')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('tti4')
        except FileNotFoundError:
            pass

    #@unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            'tests/itk/time/Image_00000.mhd',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (30, 142, 115))

    #@unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            ['tests/itk/time/Image_00000.mhd',
             'tests/itk/time/Image_00001.mhd'],
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (2, 30, 142, 115))

    #@unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'tests/itk/time',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (10, 30, 142, 115))

class test_file_archive_dicom(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #self.opts = parser.parse_args(['--template', 'tests/dicom/time/',
        #        '--geometry', 'tests/dicom/time'])
        #self.opts = parser.parse_args(['--of', 'itk', '--sort', 'tag'])
        #self.opts = parser.parse_args(['--of', 'itk'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['dicom']

        #self.hdr,self.si = imagedata.readdata.read(
        #        ('tests/dicom/time/',),
        #        imagedata.formats.INPUT_ORDER_TIME,
        #        self.opts)

    def tearDown(self):
        try:
            shutil.rmtree('ttd3')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('ttd4')
        except FileNotFoundError:
            pass

    #@unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            'tests/dicom/volume/L_00000.dcm',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (1, 160, 160))

    #@unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [
            'tests/dicom/volume/L_00000.dcm',
            'tests/dicom/volume/L_00001.dcm'
            ],
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (2, 160, 160))

    #@unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'tests/dicom/volume',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (16, 160, 160))

class test_file_archive_biff(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #self.opts = parser.parse_args(['--template', 'tests/dicom/time/',
        #        '--geometry', 'tests/dicom/time'])
        #self.opts = parser.parse_args(['--of', 'itk', '--sort', 'tag'])
        #self.opts = parser.parse_args(['--of', 'itk'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['biff']

        #self.hdr,self.si = imagedata.readdata.read(
        #        ('tests/dicom/time/',),
        #        imagedata.formats.INPUT_ORDER_TIME,
        #        self.opts)

    def tearDown(self):
        try:
            shutil.rmtree('ttb3')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('ttb4')
        except FileNotFoundError:
            pass

    #@unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            'tests/biff/time/Image_00000.biff',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.float64)
        self.assertEqual(si1.shape, (30, 142, 115))

    #@unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            ['tests/biff/time/Image_00000.biff',
             'tests/biff/time/Image_00001.biff'],
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.float64)
        self.assertEqual(si1.shape, (2, 30, 142, 115))

    #@unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'tests/biff/time',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.float64)
        self.assertEqual(si1.shape, (10, 30, 142, 115))

class test_file_archive_mat(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        #self.opts = parser.parse_args(['--template', 'tests/dicom/time/',
        #        '--geometry', 'tests/dicom/time'])
        #self.opts = parser.parse_args(['--of', 'itk', '--sort', 'tag'])
        #self.opts = parser.parse_args(['--of', 'itk'])
        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['mat']

        #self.hdr,self.si = imagedata.readdata.read(
        #        ('tests/dicom/time/',),
        #        imagedata.formats.INPUT_ORDER_TIME,
        #        self.opts)

    def tearDown(self):
        try:
            shutil.rmtree('ttm3')
        except FileNotFoundError:
            pass
        try:
            shutil.rmtree('ttm4')
        except FileNotFoundError:
            pass

    #@unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            'tests/mat/time/Image_00000.mat',
            0,
            self.opts)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (30, 142, 115))

    #@unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            ['tests/mat/time/Image_00000.mat',
             'tests/mat/time/Image_00001.mat'],
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (2, 30, 142, 115))

    #@unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            'tests/mat/time',
            imagedata.formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.float32)
        self.assertEqual(si1.shape, (10, 30, 142, 115))

if __name__ == '__main__':
    unittest.main()