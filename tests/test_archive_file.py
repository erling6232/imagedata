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
import imagedata.archives
import imagedata.transports

class test_archive_file(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['itk']

    def tearDown(self):
        shutil.rmtree('ttaf', ignore_errors=True)

    #@unittest.skip("skipping test_unknown_mimetype")
    def test_unknown_mimetype(self):
        try:
            archive = imagedata.archives.find_mimetype_plugin(
                'unknown',
                'data',
                'r')
        except imagedata.archives.ArchivePluginNotFound:
            pass

    #@unittest.skip("skipping test_mimetype")
    def test_mimetype(self):
        archive = imagedata.archives.find_mimetype_plugin(
            '*',
            'data',
            'r')

    #@unittest.skip("skipping test_unknown_url")
    def test_unknown_url(self):
        try:
            archive = imagedata.archives.find_mimetype_plugin(
                '*',
                'unknown',
                'r')
        except imagedata.transports.RootIsNotDirectory:
            pass

    #@unittest.skip("skipping test_new_archive")
    def test_new_archive(self):
        archive = imagedata.archives.find_mimetype_plugin(
            '*',
            'ttaf',
            'w')
        with archive.open('test.txt', 'w') as f:
            f.write(b'Hello world!')
        read_archive = imagedata.archives.find_mimetype_plugin(
            '*',
            'ttaf',
            'r')
        with read_archive.open('test.txt', 'r') as f:
            contents = f.read()
        self.assertEqual(contents, b'Hello world!')

if __name__ == '__main__':
    unittest.main()
