#!/usr/bin/env python3

"""Test file archive
"""

import os
import unittest
import tempfile
import logging
import argparse

from .context import imagedata
import imagedata.cmdline
import imagedata.transports
import imagedata.transports.filetransport


class TestFiletransport(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1:
            self.opts.output_format = ['dicom']

    # @unittest.skip("test_walk")
    def test_walk(self):
        tree = imagedata.transports.filetransport.FileTransport(
            root='data', mode='r', read_directory_only=False)
        walk_list = tree.walk('tree')
        # obj.assertEqual(len(walk_list), 1)
        for root, dirs, files in walk_list:
            logging.debug('test_walk: root {}'.format(root))
            logging.debug('test_walk: dirs {}'.format(dirs))
            logging.debug('test_walk: files {}'.format(files))
            expect = ['file0', 'file1', 'file2']

            self.assertEqual(root, 'tree')
            self.assertEqual(dirs, [])
            self.assertEqual(sorted(files), expect)

    # @unittest.skip("test_isfile")
    def test_isfile(self):
        tree = imagedata.transports.filetransport.FileTransport(
            root='data', mode='r', read_directory_only=False)
        self.assertEqual(tree.isfile('dicom/lena_color.dcm'), True)
        self.assertEqual(tree.isfile('dicom/time'), False)

    # @unittest.skip("test_readfile")
    def test_readfile(self):
        tree = imagedata.transports.filetransport.FileTransport(
            root='data', mode='r', read_directory_only=False)
        f = tree.open('text.txt')
        contents = f.read()
        self.assertEqual(len(contents),
                         408347 if os.name == 'nt' else 13)
        f.close()

    # @unittest.skip("test_open_file")
    def test_open_file(self):
        try:
            _ = imagedata.transports.filetransport.FileTransport(
                root='data/text.txt',
                mode='r', read_directory_only=True)
        except imagedata.transports.RootIsNotDirectory:
            pass

    # @unittest.skip("test_open_nonexist_dir")
    def test_open_nonexist_dir(self):
        try:
            _ = imagedata.transports.filetransport.FileTransport(
                root='data/nonexist',
                mode='r', read_directory_only=True)
        except imagedata.transports.RootIsNotDirectory:
            pass

    # @unittest.skip("test_open_nonexist")
    def test_open_nonexist(self):
        try:
            _ = imagedata.transports.filetransport.FileTransport(
                root='data/nonexist',
                mode='r', read_directory_only=False)
        except imagedata.transports.RootDoesNotExist:
            pass

    # @unittest.skip("test_open_new")
    def test_open_new(self):
        with tempfile.TemporaryDirectory() as d:
            _ = imagedata.transports.filetransport.FileTransport(
                root=d,
                mode='w', read_directory_only=False)

    # @unittest.skip("test_write_then_read")
    def test_write_then_read(self):
        if os.name == 'nt':
            return
        with tempfile.TemporaryDirectory() as d:
            tree = imagedata.transports.filetransport.FileTransport(
                root=d,
                mode='w', read_directory_only=False)
            f = tree.open('test.txt', mode='w')
            f.write(b'Hello world!')
            f.close()

            _ = imagedata.transports.filetransport.FileTransport(
                root=d,
                mode='r', read_directory_only=False)
            f = tree.open('test.txt')
            contents = f.read()
            f.close()
        self.assertEqual(contents, b'Hello world!')


if __name__ == '__main__':
    unittest.main()
