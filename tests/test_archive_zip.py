#!/usr/bin/env python3

"""Test zip archive
"""

import unittest
import os.path
import argparse
import tempfile
from .context import imagedata
import imagedata.cmdline
import imagedata.readdata
import imagedata.formats
import imagedata.archives
import imagedata.transports


class TestArchiveZip(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['itk']

    # @unittest.skip("skipping test_unknown_mimetype")
    def test_unknown_mimetype(self):
        try:
            archive = imagedata.archives.find_mimetype_plugin(
                'unknown',
                os.path.join('data', 'itk', 'time.zip'),
                'r')
            archive.close()
        except imagedata.transports.RootIsNotDirectory:
            pass

    # @unittest.skip("skipping test_mimetype")
    def test_mimetype(self):
        archive = imagedata.archives.find_mimetype_plugin(
            'application/zip',
            os.path.join('data', 'itk', 'time.zip'),
            'r')
        archive.close()

    # @unittest.skip("skipping test_unknown_url")
    def test_unknown_url(self):
        try:
            archive = imagedata.archives.find_mimetype_plugin(
                'application/zip',
                'unknown',
                'r')
            archive.close()
        except imagedata.transports.RootDoesNotExist:
            pass

    # @unittest.skip("skipping test_new_archive")
    def test_new_archive(self):
        with tempfile.TemporaryDirectory() as d:
            with imagedata.archives.find_mimetype_plugin(
                    'application/zip',
                    os.path.join(d, 'ar.zip'),
                    'w') as archive:
                with archive.open('test.txt', 'w') as f:
                    f.write(b'Hello world!')
            with imagedata.archives.find_mimetype_plugin(
                    'application/zip',
                    os.path.join(d, 'ar.zip'),
                    'r') as read_archive:
                read_list = read_archive.getmembers('test.txt')
                self.assertEqual(len(read_list), 1)
                with read_archive.open(read_list[0], 'r') as f:
                    contents = f.read()
        self.assertEqual(contents, 'Hello world!')


if __name__ == '__main__':
    unittest.main()
