import unittest
import os.path
import tempfile

from imagedata.image_data import conversion, sort


class TestImageData(unittest.TestCase):

    def test_sort(self):
        with tempfile.TemporaryDirectory() as d:
            sort([os.path.join(d, 'sort'),
                  os.path.join('data', 'dicom')
                  ])

    def test_conversion(self):
        with tempfile.TemporaryDirectory() as d:
            outfile = os.path.join(d, 'dump.nii.gz')
            conversion([
                '--of', 'nifti',
                outfile,
                os.path.join('data', 'dicom', 'time')
            ])
            if not os.path.isfile(outfile):
                raise AssertionError('File does not exist: {}'.format(outfile))

    def test_conversion_2(self):
        with tempfile.TemporaryDirectory() as d:
            outfile = os.path.join(d, '%p')
            conversion([
                '--of', 'nifti',
                '--of', 'dicom',
                outfile,
                os.path.join('data', 'dicom', 'time')
            ])
            dicom_dir = os.path.join(d, 'dicom')
            nifti_file = os.path.join(d, 'nifti.nii.gz')
            if not os.path.isdir(dicom_dir):
                raise AssertionError('File does not exist: {}'.format(dicom_dir))
            if not os.path.isfile(nifti_file):
                raise AssertionError('File does not exist: {}'.format(nifti_file))


if __name__ == '__main__':
    unittest.main()
