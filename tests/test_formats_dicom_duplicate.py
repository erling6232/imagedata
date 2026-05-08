import unittest
import os.path
import tempfile
import numpy as np
import imagedata.formats as formats
from imagedata import Series


class TestDuplicateDicom2D(unittest.TestCase):
    """Test duplicate 2D data.
    Three slices at same slice position.
    Expected result:
        duplicate.shape (3, 1, 192, 152), axes: none, slice, row, column
    """
    def setUp(self):
        # Prepare duplicate dataset
        si = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            input_format='dicom'
        )
        self.d = tempfile.TemporaryDirectory()
        si[0].write(os.path.join(self.d.name, '0'), formats=['dicom'], keep_uid=True)
        si[0].write(os.path.join(self.d.name, '1'), formats=['dicom'], keep_uid=True)
        si[0].write(os.path.join(self.d.name, '2'), formats=['dicom'], keep_uid=True)

    def tearDown(self):
        self.d.cleanup()

    def test_no_duplicate_2d(self):
        """Setting accept_duplicate_tag=True should not fail on a non-duplicate dicom series."""
        no_duplicate = Series(os.path.join(self.d.name, '0.dcm'),
                           'none', input_format='dicom', accept_duplicate_tag=True)
        self.assertEqual((192, 152), no_duplicate.shape)

    def test_duplicate_2d(self):
        duplicate = Series(self.d.name, 'none', input_format='dicom', accept_duplicate_tag=True)
        self.assertEqual((3, 1, 192, 152), duplicate.shape)

    def test_duplicate_error_2d(self):
        with self.assertRaises(formats.CannotSort) as context:
            _ = Series(self.d.name, 'none', input_format='dicom', accept_duplicate_tag=False)


class TestDuplicateDicom3D(unittest.TestCase):
    """Test duplicate 3D data.
    Three volumes at same tag.
    Expected result:
        duplicate.shape (3, 3, 192, 152), axes: none, slice, row, column
    """
    def setUp(self):
        # Prepare duplicate dataset
        si0 = Series(
            os.path.join('data', 'dicom', 'time', 'time00'),
            'none',
            input_format='dicom'
        )
        self.d = tempfile.TemporaryDirectory()
        si0.write(os.path.join(self.d.name, '0'), formats=['dicom'], keep_uid=True)
        si0.write(os.path.join(self.d.name, '1'), formats=['dicom'], keep_uid=True)
        si0.write(os.path.join(self.d.name, '2'), formats=['dicom'], keep_uid=True)

    def tearDown(self):
        self.d.cleanup()

    def test_no_duplicate_3d(self):
        """Setting accept_duplicate_tag=True should not fail on a non-duplicate dicom series."""
        no_duplicate = Series(os.path.join(self.d.name, '0'),
                              'none', input_format='dicom', accept_duplicate_tag=True)
        self.assertEqual((3, 192, 152), no_duplicate.shape)

    def test_duplicate_3d(self):
        duplicate = Series(self.d.name, 'none', input_format='dicom', accept_duplicate_tag=True)
        self.assertEqual((3, 3, 192, 152), duplicate.shape)

    def test_duplicate_error_3d(self):
        with self.assertRaises(formats.CannotSort) as context:
            _ = Series(self.d.name, 'none', input_format='dicom', accept_duplicate_tag=False)


class TestDuplicateDicom4D(unittest.TestCase):
    """Test duplicate 4D data.
    Three volumes at same tag.
    Expected result:
        duplicate.shape (9, 3, 192, 152), axes: time, slice, row, column.
        duplicate.axes.time: each time point repeated three times.
    """
    def setUp(self):
        # Prepare duplicate dataset
        si0 = Series(
            os.path.join('data', 'dicom', 'time'),
            'time',
            input_format='dicom'
        )
        self.d = tempfile.TemporaryDirectory()
        si0.write(os.path.join(self.d.name, '0'), formats=['dicom'], keep_uid=True)
        si0.write(os.path.join(self.d.name, '1'), formats=['dicom'], keep_uid=True)
        si0.write(os.path.join(self.d.name, '2'), formats=['dicom'], keep_uid=True)

    def tearDown(self):
        self.d.cleanup()

    def test_no_duplicate_4d(self):
        """Setting accept_duplicate_tag=True should not fail on a non-duplicate dicom series."""
        no_duplicate = Series(os.path.join(self.d.name, '0'),
                              'time', input_format='dicom', accept_duplicate_tag=True)
        self.assertEqual((3, 3, 192, 152), no_duplicate.shape)

    def test_duplicate_4d(self):
        duplicate = Series(self.d.name, 'time', input_format='dicom', accept_duplicate_tag=True)
        self.assertEqual((9, 3, 192, 152), duplicate.shape)

    def test_duplicate_error_4d(self):
        with self.assertRaises(formats.CannotSort) as context:
            _ = Series(self.d.name, 'time', input_format='dicom', accept_duplicate_tag=False)


class TestDuplicateDicom5D(unittest.TestCase):
    """Test duplicate 5D data.
    Three 4D datasets at same tag.
    Expected result:
        duplicate.shape (3, 12, 3, 72, 96), axes: time, te, slice, row, column.
        duplicate.axes.te: each echo time (TE) repeated three times.
    """
    def setUp(self):
        # Prepare duplicate dataset
        si0 = Series(
            os.path.join('data', 'dicom', '5D.zip?t1_fl2d_DE_4TEs'),
            'time,te',
            input_format='dicom'
        )
        self.d = tempfile.TemporaryDirectory()
        si0.write(os.path.join(self.d.name, '0'), formats=['dicom'], keep_uid=True)
        si0.write(os.path.join(self.d.name, '1'), formats=['dicom'], keep_uid=True)
        si0.write(os.path.join(self.d.name, '2'), formats=['dicom'], keep_uid=True)

    def tearDown(self):
        self.d.cleanup()

    def test_no_duplicate_5d(self):
        """Setting accept_duplicate_tag=True should not fail on a non-duplicate dicom series."""
        no_duplicate = Series(os.path.join(self.d.name, '0'),
                              'time,te', input_format='dicom', accept_duplicate_tag=True)
        self.assertEqual((3, 4, 3, 72, 96), no_duplicate.shape)

    def test_duplicate_5d(self):
        duplicate = Series(self.d.name, 'time,te', input_format='dicom', accept_duplicate_tag=True)
        assert duplicate.shape == (3, 12, 3, 72, 96)
        self.assertEqual((3, 12, 3, 72, 96), duplicate.shape)

    def test_duplicate_error_5d(self):
        with self.assertRaises(formats.CannotSort) as context:
            _ = Series(self.d.name, 'time,te', input_format='dicom', accept_duplicate_tag=False)


if __name__ == '__main__':
    unittest.main()
