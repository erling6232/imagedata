#!/usr/bin/env python3

"""Test zip archive
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

class test_archive_zip(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['itk']

    def tearDown(self):
        shutil.rmtree('ttaz', ignore_errors=True)

    #@unittest.skip("skipping test_unknown_mimetype")
    def test_unknown_mimetype(self):
        try:
            archive = imagedata.archives.find_mimetype_plugin(
                'unknown',
                'data/itk/time.zip',
                'r')
            archive.close()
        except imagedata.archives.ArchivePluginNotFound:
            pass

    #@unittest.skip("skipping test_mimetype")
    def test_mimetype(self):
        archive = imagedata.archives.find_mimetype_plugin(
            'application/zip',
            'data/itk/time.zip',
            'r')
        archive.close()

    #@unittest.skip("skipping test_unknown_url")
    def test_unknown_url(self):
        try:
            archive = imagedata.archives.find_mimetype_plugin(
                'application/zip',
                'unknown',
                'r')
            archive.close()
        except imagedata.transports.RootDoesNotExist:
            pass

    #@unittest.skip("skipping test_new_archive")
    def test_new_archive(self):
        with imagedata.archives.find_mimetype_plugin(
            'application/zip',
            'ttaz/ar.zip',
            'w') as archive:
            with archive.open('test.txt', 'w') as f:
                f.write(b'Hello world!')
        with imagedata.archives.find_mimetype_plugin(
            'application/zip',
            'ttaz/ar.zip',
            'r') as read_archive:
            read_list = read_archive.getmembers('test.txt')
            self.assertEqual(len(read_list), 1)
            with read_archive.open(read_list[0], 'r') as f:
                contents = f.read()
        self.assertEqual(contents, 'Hello world!')

if __name__ == '__main__':
    unittest.main()
