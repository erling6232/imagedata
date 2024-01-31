#!/usr/bin/env python3

"""Test file archive
"""

import os
import unittest
import argparse
import tempfile
# from .context import imagedata
import src.imagedata.cmdline as cmdline
import src.imagedata.archives as archives
import src.imagedata.transports as transports


class TestArchiveFile(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['itk']

    # @unittest.skip("skipping test_unknown_mimetype")
    def test_unknown_mimetype(self):
        try:
            _ = archives.find_mimetype_plugin(
                'unknown',
                'data',
                'r')
        except archives.ArchivePluginNotFound:
            pass

    # @unittest.skip("skipping test_mimetype")
    def test_mimetype(self):
        _ = archives.find_mimetype_plugin(
            '*',
            'data',
            'r')

    # @unittest.skip("skipping test_unknown_url")
    def test_unknown_url(self):
        try:
            _ = archives.find_mimetype_plugin(
                '*',
                'unknown',
                'r')
        except FileNotFoundError:
            pass

    # @unittest.skip("skipping test_read_one_file")
    def test_read_one_file(self):
        archive = archives.find_mimetype_plugin(
            None,
            '.',
            'r')
        files = archive.getnames(os.path.join('.', 'data', 'lena_color.gif'))
        self.assertEqual(1, len(files))

    # @unittest.skip("skipping test_new_archive")
    def test_new_archive(self):
        with tempfile.TemporaryDirectory() as d:
            archive = archives.find_mimetype_plugin(
                '*',
                d,
                'w')
            with archive.open(os.path.join(d, 'test.txt'), 'w') as f:
                f.write(b'Hello world!')
            read_archive = archives.find_mimetype_plugin(
                '*',
                d,
                'r')
            with read_archive.open(os.path.join(d, 'test.txt'), 'r') as f:
                contents = f.read()
        self.assertEqual(contents, b'Hello world!')


if __name__ == '__main__':
    unittest.main()
