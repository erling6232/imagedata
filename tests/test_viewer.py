import unittest
import os.path
import tempfile
import matplotlib
import matplotlib.pyplot as plt

from imagedata.series import Series
from imagedata.viewer import Viewer, default_layout


class TestViewer(unittest.TestCase):

    def setUp(self):
        matplotlib.use('agg')
        self.tmpdir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmpdir.cleanup()

    def test_view_2D(self):
        si = Series('data/dicom/time/time00', input_format='dicom')[0]
        fig = plt.figure()
        axes = default_layout(fig, 1)
        # _ = Viewer([si], fig=fig, ax=axes)
        si.show(ax=axes)
        with tempfile.TemporaryDirectory() as d:
            plt.savefig(os.path.join(d, 'view.png'))
            pass

    def test_view_3D(self):
        si = Series('data/dicom/time/time00')
        fig = plt.figure()
        axes = default_layout(fig, 1)
        # _ = Viewer([si], fig=fig, ax=axes)
        si.show(ax=axes)
        with tempfile.TemporaryDirectory() as d:
            plt.savefig(os.path.join(d, 'view.png'))
            pass

    def test_get_rgb_voxel(self):
        si = Series('data/dicom/time/time00')

        rgb = si.to_rgb()

        fig = plt.figure()
        axes = default_layout(fig, 1)
        _ = Viewer([rgb], fig=fig, ax=axes)
        with tempfile.TemporaryDirectory() as d:
            plt.savefig(os.path.join(d, 'show.png'))

        _slice = rgb[1]
        voxel = _slice[1, 1]
        self.assertEqual((0, 0, 0), voxel)


if __name__ == '__main__':
    unittest.main()
