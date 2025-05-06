import unittest
import os.path
import tempfile
import numpy as np
import logging
import argparse

# from .context import imagedata
import src.imagedata.cmdline as cmdline
import src.imagedata.formats as formats
from src.imagedata.series import Series
from src.imagedata.collection import Cohort


def list_files(startpath):
    import os
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = ' ' * 4 * level
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files:
            print('{}{}'.format(subindent, f))


class Test3DMatPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'mat', '--serdes', '1'])

    def test_mat_plugin(self):
        plugins = formats.get_plugins_list()
        self.mat_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'mat':
                self.mat_plugin = pclass
        self.assertIsNotNone(self.mat_plugin)

    # @unittest.skip("skipping test_read_single_file")
    def test_read_single_file(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.input_format, 'mat')
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    def test_dtype_int64(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
            'none',
            dtype=int,
            input_format='mat')
        self.assertEqual(si1.dtype, np.int64)

    def test_dtype_float(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
            'none',
            dtype=float,
            input_format='mat')
        self.assertEqual(si1.dtype, np.float64)

    # @unittest.skip("skipping test_read_2D")
    def test_read_2D(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        si2 = si1[0, 0, ...]
        with tempfile.TemporaryDirectory() as d:
            si2.write(d, formats=['mat'])
            si3 = Series(d)
        self.assertEqual(si2.dtype, si3.dtype)
        self.assertEqual(si2.shape, si3.shape)
        np.testing.assert_array_equal(si2, si3)

    # @unittest.skip("skipping test_read_two_files")
    def test_read_two_files(self):
        si1 = Series(
            [os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
             os.path.join('data', 'mat', 'time', 'Image_00000.mat')],
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_read_single_directory")
    def test_read_single_directory(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_zipread_single_file")
    def test_zipread_single_file(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time.zip?time/Image_00000.mat'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_zipread_single_file_wildcard")
    def test_zipread_single_file_wildcard(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time.zip?*Image_00000.mat'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_zipread_single_file_relative")
    def test_zipread_single_file_relative(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time.zip?time/Image_00000.mat'),
            'none',
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_zipread_single_directory")
    def test_zipread_single_directory(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time.zip?time'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    # @unittest.skip("skipping test_zipread_all_files")
    def test_zipread_all_files(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time.zip'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))

    def test_write_ndarray(self):
        with tempfile.TemporaryDirectory() as d:
            Series(np.eye(128)).write(d, formats=['mat'])

    # @unittest.skip("skipping test_write_single_file")
    def test_write_single_file(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
            'none',
            self.opts)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'Image00000.mat'), formats=['mat'])
            si2 = Series(os.path.join(d, 'Image00000.mat'))
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    def test_write_single_file_not_directory(self):
        a = Series(np.eye(128))
        with tempfile.TemporaryDirectory() as d:
            filename = os.path.join(d, 'test.mat')
            a.write(
                filename,
                formats=['mat']
            )
            if not os.path.isfile(filename):
                raise AssertionError('File does not exist: {}'.format(filename))


    # @unittest.skip("skipping test_write_single_directory")
    def test_write_single_directory(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time'),
            'none',
            self.opts)
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['mat'])
            si2 = Series(d)
        self.assertEqual(si1.dtype, si2.dtype)
        self.assertEqual(si1.shape, si2.shape)

    # @unittest.skip("skipping test_read_3d_mat_no_opt")
    def test_read_3d_mat_no_opt(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
            'none',
            self.opts)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'mat', 'Image_00000.mat'), formats=['mat'], opts=self.opts)
            si2 = Series(os.path.join(d, 'mat', 'Image_00000.mat'))

    # @unittest.skip("skipping test_write_3d_mat_no_opt")
    def test_write_3d_mat_no_opt(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
            'none',
            self.opts)
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'mat', 'Image_{:05d}.mat'), formats=['mat'])

    # @unittest.skip("skipping test_read_3d_mat")
    def test_write_3d_mat(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
            'none',
            self.opts)
        logging.debug('test_write_3d_mat: si1 {} {} {} {}'.format(type(si1), si1.dtype, si1.min(), si1.max()))
        logging.debug('test_write_3d_mat: si1.shape {}, si1.slices {}'.format(si1.shape, si1.slices))

        logging.debug('test_write_3d_mat: si1.tags {}'.format(si1.tags))
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'mat', 'Image_00000.mat'), formats=['mat'], opts=self.opts)
            logging.debug('test_write_3d_mat: si1 {} {} {}'.format(si1.dtype, si1.min(), si1.max()))
            logging.debug('test_write_3d_mat: si1.shape {}, si1.slices {}'.format(si1.shape, si1.slices))

            si2 = Series(
                os.path.join(d, 'mat', 'Image_00000.mat'),
                'none',
                self.opts)
        logging.debug('test_write_3d_mat: si2 {} {} {}'.format(si2.dtype, si2.min(), si2.max()))

        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

        logging.debug('test_write_3d_mat: Get si1.slices {}'.format(si1.slices))
        logging.debug('test_write_3d_mat: Set s3')
        s3 = si1.astype(np.float64)
        logging.debug('test_write_3d_mat: s3 {} {} {} {}'.format(type(s3),
                                                                 issubclass(type(s3), Series), s3.dtype, s3.shape))
        logging.debug('test_write_3d_mat: s3 {} {} {}'.format(s3.dtype,
                                                              s3.min(), s3.max()))
        logging.debug('test_write_3d_mat: s3.slices {}'.format(s3.slices))
        si3 = Series(s3)
        np.testing.assert_array_almost_equal(si1, si3, decimal=4)
        logging.debug('test_write_3d_mat: si3.slices {}'.format(si3.slices))
        logging.debug('test_write_3d_mat: si3 {} {} {}'.format(type(si3), si3.dtype, si3.shape))

        s3 = si1 - si2
        with tempfile.TemporaryDirectory() as d:
            s3.write(os.path.join(d, 'diff', 'Image_{:05d}.mat'), formats=['mat'], opts=self.opts)


class Test4DMatPlugin(unittest.TestCase):
    def setUp(self):
        parser = argparse.ArgumentParser()
        cmdline.add_argparse_options(parser)

        self.opts = parser.parse_args(['--of', 'mat'])

        plugins = formats.get_plugins_list()
        self.mat_plugin = None
        for pname, ptype, pclass in plugins:
            if ptype == 'mat':
                self.mat_plugin = pclass
        self.assertIsNotNone(self.mat_plugin)

    # @unittest.skip("skipping test_write_4d_mat")
    def test_write_4d_mat(self):
        si1 = Series(
            os.path.join('data', 'mat', 'time', 'Image_00000.mat'),
            formats.INPUT_ORDER_TIME,
            self.opts)
        self.assertEqual(si1.dtype, np.uint16)
        self.assertEqual(si1.shape, (3, 3, 192, 152))
        with tempfile.TemporaryDirectory() as d:
            si1.write(os.path.join(d, 'mat', 'Image_00000.mat'), formats=['mat'], opts=self.opts)

            # Read back the MAT data and compare to original si1
            si2 = Series(
                os.path.join(d, 'mat', 'Image_00000.mat'),
                formats.INPUT_ORDER_TIME,
                self.opts)
        self.assertEqual(si1.shape, si2.shape)
        np.testing.assert_array_equal(si1, si2)

    def test_write_cohort(self):
        cohort = Cohort('data/dicom/cohort.zip')
        with tempfile.TemporaryDirectory() as d:
            cohort.write(d, formats=['mat'])


if __name__ == '__main__':
    unittest.main()
