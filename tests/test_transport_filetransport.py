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
import imagedata.transports
import imagedata.transports.filetransport

def list_files(startpath):
    import os
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))

class test_filetransport(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        imagedata.cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args([])
        if len(self.opts.output_format) < 1: self.opts.output_format=['itk']

    def tearDown(self):
        shutil.rmtree('ttft', ignore_errors=True)

    #@unittest.skip("test_walk")
    def test_walk(self):
        tree = imagedata.transports.filetransport.FileTransport(
                'data', mode='r', read_directory_only=False)
        walk_list = tree.walk('ps/pages')
        self.assertEqual(len(walk_list), 1)
        root, dirs, files = walk_list[0]
        logging.debug('test_walk: root {}'.format(root))
        logging.debug('test_walk: dirs {}'.format(dirs))
        logging.debug('test_walk: files {}'.format(files))
        expect = ['A_Lovers_Complaint_1.ps',
                'A_Lovers_Complaint_2.ps', 'A_Lovers_Complaint_3.ps',
                'A_Lovers_Complaint_4.ps', 'A_Lovers_Complaint_5.ps',
                'A_Lovers_Complaint_6.ps']

        self.assertEqual(root, 'ps/pages')
        self.assertEqual(dirs, [])
        self.assertEqual(files, expect)

    #@unittest.skip("test_isfile")
    def test_isfile(self):
        tree = imagedata.transports.filetransport.FileTransport(
                'data', mode='r', read_directory_only=False)
        self.assertEqual(tree.isfile('ps/A_Lovers_Complaint.ps'), True)
        self.assertEqual(tree.isfile('ps/pages'), False)

    #@unittest.skip("test_readfile")
    def test_readfile(self):
        tree = imagedata.transports.filetransport.FileTransport(
                'data', mode='r', read_directory_only=False)
        f = tree.open('ps/A_Lovers_Complaint.ps')
        contents = f.read()
        self.assertEqual(len(contents), 385176)

if __name__ == '__main__':
    unittest.main()
