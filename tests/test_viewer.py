#!/usr/bin/env python3

# import nose.tools
import unittest
import numpy as np
import copy
import tempfile
# import logging
import matplotlib
import matplotlib.pyplot as plt

from .context import imagedata
from imagedata.series import Series
from imagedata.viewer import Viewer, default_layout
import imagedata.axis
from .compare_headers import compare_axes


class TestViewer(unittest.TestCase):

    def setUp(self):
        matplotlib.use('agg')
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_get_rgb_voxel(self):
        si = Series('data/dicom/time/time00')

        rgb = si.to_rgb()

        fig = plt.figure()
        axes = default_layout(fig, 1)
        _ = Viewer([rgb], fig=fig, ax=axes)
        plt.savefig(self.tmpdir.name + 'show.png')

        _slice = rgb[1]
        voxel = _slice[1, 1]
        self.assertEqual(1, len(voxel.axes))


if __name__ == '__main__':
    unittest.main()
