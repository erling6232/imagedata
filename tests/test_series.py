#!/usr/bin/env python3

# import nose.tools
import unittest
import numpy as np
import copy
# import logging
import pydicom.datadict

from .context import imagedata
from imagedata.series import Series
import imagedata.axis
from .compare_headers import compare_axes


class TestSeries(unittest.TestCase):

    #@unittest.skip("skipping test_repr")
    def test_repr(self):
        si = Series(
            'data/dicom/time/time00/Image_00020.dcm')
        r = si.__repr__()

    #@unittest.skip("skipping test_repr_vol")
    def test_repr_vol(self):
        si = Series(
            'data/dicom/time/time00')
        r = si.__repr__()

    #@unittest.skip("skipping test_str")
    def test_str(self):
        si = Series(
            'data/dicom/time/time00/Image_00020.dcm')
        r = '{}'.format(si)

    #@unittest.skip("skipping test_max")
    def test_max(self):
        si = Series(
            'data/dicom/time/time00/Image_00020.dcm')
        mi = si.max()
        self.assertEqual(type(mi), np.uint16)

    #@unittest.skip("skipping test_get_keyword")
    def test_get_keyword(self):
        si1 = Series(
            'data/dicom/time/time00/Image_00020.dcm')
        pname = si1.getDicomAttribute('PatientName')
        self.assertEqual(
            si1.getDicomAttribute('PatientName'),
            'PHANTOM^T1')
        self.assertEqual(
            si1.getDicomAttribute(
                pydicom.datadict.tag_for_keyword('PatientID')),
            '19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035')

    #@unittest.skip("skipping test_create_series_1")
    def test_create_series_1(self):
        si = Series(np.uint16(1))
        self.assertEqual(np.uint16, si.dtype)
        self.assertEqual((1,), si.shape)

    #@unittest.skip("skipping test_create_series_tuple_1D")
    def test_create_series_tuple_1D(self):
        si = Series((1, 2, 3))
        self.assertEqual(np.int64, si.dtype)
        self.assertEqual((3,), si.shape)

    #@unittest.skip("skipping test_create_series")
    def test_create_series(self):
        a = np.eye(128)
        si = Series(a)
        self.assertEqual(si.dtype, np.float64)
        self.assertEqual(si.shape, (128, 128))

    #@unittest.skip("skipping test_print_header")
    def test_print_header(self):
        a = np.eye(128)
        si = Series(a)
        si.spacing = (1, 1, 1)
        self.assertEqual(si.slices, 1)
        np.testing.assert_array_equal(si.spacing, np.array((1, 1, 1)))

    #@unittest.skip("skipping test_copy_series")
    def test_copy_series(self):
        a = np.eye(128)
        si1 = Series(a)
        si1.spacing = (1, 1, 1)

        si2 = si1
        self.assertEqual(si2.slices, 1)
        np.testing.assert_array_equal(si2.spacing, np.array((1, 1, 1)))

    #@unittest.skip("skipping test_copy_series2")
    def test_copy_series2(self):
        a = np.eye(128)
        si1 = Series(a)
        si1.spacing = (1, 1, 1)

        si2 = si1.copy()
        self.assertEqual(si2.slices, 1)
        np.testing.assert_array_equal(si2.spacing, np.array((1, 1, 1)))

    #@unittest.skip("skipping test_subtract_series")
    def test_subtract_series(self):
        a = Series(np.eye(128))
        b = Series(np.eye(128))
        a.spacing = (1, 1, 1)
        b.spacing = a.spacing

        si = a - b
        self.assertEqual(si.slices, 1)
        np.testing.assert_array_equal(si.spacing, np.array((1, 1, 1)))

    #@unittest.skip("skipping test_increase_ndim")
    def test_increase_ndim(self):
        a = np.eye(128)
        s = Series(a)
        with self.assertRaises(IndexError):
            s.shape = (1,1,128,128)

    #@unittest.skip("skipping test_slicing_dim")
    def test_slicing_dim(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a = np.vstack([a1, a1, a1, a1, a1])
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[0])

        s_slice = s[2]
        self.assertEqual(s_slice.ndim, 2)
        self.assertEqual(len(s_slice.axes), 2)
        self.assertEqual(s_slice.axes[0].name, 'row')
        self.assertEqual(s_slice.axes[1].name, 'column')

    #@unittest.skip("skipping test_slicing_y")
    def test_slicing_y(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a = np.vstack([a1, a1, a1])
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[0])

        a_slice = a[:,:,3:5]
        s_slice = s[:,:,3:5]
        np.testing.assert_array_equal(a_slice, s_slice)
        self.assertEqual(s_slice.slices, 3)

    #@unittest.skip("skipping test_slicing_y_neg")
    def test_slicing_y_neg(self):
        from numpy.random import default_rng
        rng = default_rng()
        s = Series(rng.standard_normal(64).reshape((4,4,4)))
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[0])
        np.testing.assert_array_equal(s[:,3,:], s[:,-1,:])
        np.testing.assert_array_equal(s[:,2,:], s[:,-2,:])
        np.testing.assert_array_equal(s[:,1,:], s[:,-3,:])
        np.testing.assert_array_equal(s[:,0,:], s[:,-4,:])
        np.testing.assert_array_equal(s[:,2:3,:], s[:,2:-1,:])
        np.testing.assert_array_equal(s[:,1:2,:], s[:,1:-2,:])
        np.testing.assert_array_equal(s[:,0:2,:], s[:,0:-2,:])

    #@unittest.skip("skipping test_slicing_x")
    def test_slicing_x(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a = np.vstack([a1, a1, a1])
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[0])

        a_slice = a[:,3:5,...]
        s_slice = s[:,3:5,...]
        np.testing.assert_array_equal(a_slice, s_slice)
        self.assertEqual(s_slice.slices, 3)

    #@unittest.skip("skipping test_slicing_x_neg")
    def test_slicing_x_neg(self):
        from numpy.random import default_rng
        rng = default_rng()
        s = Series(rng.standard_normal(64).reshape((4,4,4)))
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[0])
        np.testing.assert_array_equal(s[:,:,3], s[:,:,-1])
        np.testing.assert_array_equal(s[:,:,2], s[:,:,-2])
        np.testing.assert_array_equal(s[:,:,1], s[:,:,-3])
        np.testing.assert_array_equal(s[:,:,0], s[:,:,-4])
        np.testing.assert_array_equal(s[:,:,2:3], s[:,:,2:-1])
        np.testing.assert_array_equal(s[:,:,1:2], s[:,:,1:-2])
        np.testing.assert_array_equal(s[:,:,0:2], s[:,:,0:-2])

    #@unittest.skip("skipping test_assign_slice_x")
    def test_assign_slice_x(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a = np.vstack([a1, a1, a1])
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[0])
        n = np.ones_like(a) * 4
        p = Series(n)
        p.spacing = (1,1,1)
        p.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, p.shape[0])

        a[:,3:5,...] = n[:,3:5,...]
        s[:,3:5,...] = p[:,3:5,...]
        np.testing.assert_array_equal(a, s)
        np.testing.assert_array_equal(s[:,3:5,...], p[:,3:5,...])
        self.assertEqual(s.slices, 3)

    #@unittest.skip("skipping test_assign_slice")
    def test_assign_slice(self):
        a = np.array(range(4*5*6), dtype=np.uint16)
        a.shape = (4,5,6)
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[0])
        n = np.zeros((2,2,2), dtype=np.uint16)
        p = Series(n)
        p.spacing = (1,1,1)
        p.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, p.shape[0])

        a[1:3,2:4,2:4] = n[:]
        s[1:3,2:4,2:4] = p[:]
        np.testing.assert_array_equal(a, s)
        self.assertEqual(s.slices, 4)

    #@unittest.skip("skipping test_assign_slice_input_order")
    def test_assign_slice_input_order(self):
        si = Series('data/dicom/time', 'time')
        sic = Series(si)
        self.assertEqual(si.input_order, sic.input_order)

    #@unittest.skip("skipping test_slicing_z")
    def test_slicing_z(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a = np.vstack([a1, a1, a1])
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.sliceLocations = [3, 6, 9]
        for slice in range(s.slices):
            s.imagePositions = {
                slice: np.array([slice, 0, 0])
            }
        s.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[0])

        a_slice = a[0:2,...]
        s_slice = s[0:2,...]
        np.testing.assert_array_equal(a_slice, s_slice)
        self.assertEqual(s_slice.slices, 2)
        np.testing.assert_array_equal(s_slice.sliceLocations, np.array([3, 6]))
        # Compare imagePositions
        ipp2 = {}
        for slice in range(2):
            ipp2[slice] = np.array([slice, 0, 0])
        self.assertEqual(len(s_slice.imagePositions), len(ipp2))
        for slice in range(2):
            np.testing.assert_array_equal(
                s_slice.imagePositions[slice],
                ipp2[slice])

    #@unittest.skip("skipping test_slicing_z_neg")
    def test_slicing_z_neg(self):
        from numpy.random import default_rng
        rng = default_rng()
        s = Series(rng.standard_normal(64).reshape((4,4,4)))
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[0])
        np.testing.assert_array_equal(s[3,:,:], s[-1,:,:])
        np.testing.assert_array_equal(s[2,:,:], s[-2,:,:])
        np.testing.assert_array_equal(s[1,:,:], s[-3,:,:])
        np.testing.assert_array_equal(s[0,:,:], s[-4,:,:])
        np.testing.assert_array_equal(s[2:3], s[2:-1])
        np.testing.assert_array_equal(s[2:3,:,:], s[2:-1,:,:])
        np.testing.assert_array_equal(s[1:2,:,:], s[1:-2,:,:])
        np.testing.assert_array_equal(s[0:2,:,:], s[0:-2,:,:])

    #@unittest.skip("skipping test_slicing_t")
    def test_slicing_t(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a2 = np.vstack([a1, a1, a1])
        a2.shape = (1,3,128,128)
        a = np.vstack([a2,a2,a2,a2])
        s = Series(a, input_order=imagedata.formats.INPUT_ORDER_TIME)
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('time', 0, s.shape[0])
        s.axes[1] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[1])
        tags = {}
        k = 0
        for i in range(s.slices):
            tags[i] = np.arange(k, k+s.shape[0])
            k += s.shape[0]
        s.tags = tags

        a_slice = a[1:3,...]
        s_slice = s[1:3,...]
        np.testing.assert_array_equal(a_slice, s_slice)
        self.assertEqual(s_slice.slices, 3)
        self.assertEqual(len(s_slice.tags[0]), 2)
        for s in range(s_slice.slices):
            np.testing.assert_array_equal(s_slice.tags[s], tags[s][1:3])

    #@unittest.skip("skipping test_slicing_t_neg")
    def test_slicing_t_neg(self):
        from numpy.random import default_rng
        rng = default_rng()
        s = Series(rng.standard_normal(192).reshape((3,4,4,4)))
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('time', 0, s.shape[0])
        s.axes[1] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[1])
        np.testing.assert_array_equal(s[2], s[-1])
        np.testing.assert_array_equal(s[1], s[-2])
        np.testing.assert_array_equal(s[0], s[-3])
        np.testing.assert_array_equal(s[1:2], s[1:-1])
        np.testing.assert_array_equal(s[1:2,:,:], s[1:-1,:,:])
        np.testing.assert_array_equal(s[0:1,:,:], s[0:-2,:,:])

    #@unittest.skip("skipping test_slicing_t_drop")
    def test_slicing_t_drop(self):
        from numpy.random import default_rng
        rng = default_rng()
        s = Series(rng.standard_normal(192).reshape((3,4,4,4)))
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('time', 0, s.shape[0])
        s.axes[1] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[1])
        s_axes = copy.copy(s.axes)
        self.assertEqual(len(s_axes), 4)

        sum = np.sum(s, axis=0)
        compare_axes(self, s.axes, s_axes)
        del sum.axes[0]  # TODO
        self.assertEqual(len(s_axes), 4)
        compare_axes(self, s_axes[1:], sum.axes)

    #@unittest.skip("skipping test_multiple_ellipses")
    def test_multiple_ellipses(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a2 = np.vstack([a1, a1, a1])
        a2.shape = (1,3,128,128)
        a = np.vstack([a2,a2,a2,a2])
        s = Series(a, input_order=imagedata.formats.INPUT_ORDER_TIME)
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('time', 0, s.shape[0])
        s.axes[1] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[1])
        tags = {}
        k = 0
        for i in range(s.slices):
            tags[i] = np.arange(k, k+s.shape[0])
            k += s.shape[0]
        s.tags = tags

        with self.assertRaises(IndexError):
            s_slice = s[...,1:3,...]

    #@unittest.skip("skipping test_ellipsis_first")
    def test_ellipsis_first(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a2 = np.vstack([a1, a1, a1])
        a2.shape = (1,3,128,128)
        a = np.empty([4, 3,128,128])
        for i in range(4):
            a[i] = a2
        # a = np.vstack([a2,a2,a2,a2])
        s = Series(a, input_order=imagedata.formats.INPUT_ORDER_TIME)
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('time', 0, s.shape[0])
        s.axes[1] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[1])
        tags = {}
        k = 0
        for i in range(s.slices):
            tags[i] = np.arange(k, k+s.shape[0])
            k += s.shape[0]
        s.tags = tags

        a_slice = a[..., 3:5]
        s_slice = s[..., 3:5]
        np.testing.assert_array_equal(a_slice, s_slice)
        self.assertEqual(s_slice.slices, s.slices)
        self.assertEqual(len(s_slice.tags[0]), len(s.tags[0]))

    #@unittest.skip("skipping test_ellipsis_middle")
    def test_ellipsis_middle(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a2 = np.vstack([a1, a1, a1])
        a2.shape = (1,3,128,128)
        a = np.vstack([a2,a2,a2,a2])
        s = Series(a, input_order=imagedata.formats.INPUT_ORDER_TIME)
        s.spacing = (1, 1, 1)
        s.axes[0] = imagedata.axis.UniformLengthAxis('time', 0, s.shape[0])
        s.axes[1] = imagedata.axis.UniformLengthAxis('slice', 0, s.shape[1])
        tags = {}
        k = 0
        for i in range(s.slices):
            tags[i] = np.arange(k, k+s.shape[0])
            k += s.shape[0]
        s.tags = tags

        a_slice = a[1:3, ..., 3:5]
        s_slice = s[1:3, ..., 3:5]
        np.testing.assert_array_equal(a_slice, s_slice)
        self.assertEqual(s_slice.slices, s.slices)
        self.assertEqual(len(s_slice.tags[0]), 2)

    #@unittest.skip("skipping test_cross_talk")
    def test_cross_talk(self):
        si = Series('data/dicom/time/time00')
        # print('si before', si.getDicomAttribute('SeriesInstanceUID'), si.seriesInstanceUID)
        si1 = si[0]
        si1.seriesNumber = si.seriesNumber + 10
        self.assertNotEqual(si.seriesNumber, si1.seriesNumber)
        # print('si after', si.getDicomAttribute('SeriesInstanceUID'), si.seriesInstanceUID)
        # print('si1', si1.getDicomAttribute('SeriesInstanceUID'), si1.seriesInstanceUID)
        self.assertNotEqual(si.seriesInstanceUID, si1.seriesInstanceUID)

        si2 = Series('data/dicom/time/time00')
        si2.seriesNumber += 10
        self.assertNotEqual(si.seriesNumber, si2.seriesNumber)

    #@unittest.skip("skipping test_cross_talk_wl_ref")
    def test_cross_talk_wl_ref(self):
        si = Series('data/dicom/time/time00')
        si1 = si[0] * 10
        self.assertNotEqual(si.getDicomAttribute('WindowWidth'), si1.getDicomAttribute('WindowWidth'))

    #@unittest.skip("skipping test_cross_talk_wl")
    def test_cross_talk_wl(self):
        si = Series('data/dicom/time/time00')
        si1 = si.deepcopy()[0] * 10
        self.assertNotEqual(si.getDicomAttribute('WindowWidth'), si1.getDicomAttribute('WindowWidth'))

    #@unittest.skip("skipping test_cross_talk_series_ref")
    def test_cross_talk_series_ref(self):
        si = Series('data/dicom/time/time00')
        si1 = Series(si, input_order=si.input_order)
        si1.setDicomAttribute('WindowWidth', 1)
        self.assertNotEqual(si.getDicomAttribute('WindowWidth'), si1.getDicomAttribute('WindowWidth'))

    #@unittest.skip("skipping test_cross_talk_series")
    def test_cross_talk_series(self):
        si = Series('data/dicom/time/time00')
        si1 = si.deepcopy()
        si1.setDicomAttribute('WindowWidth', 1)
        self.assertEqual(1, si1.getDicomAttribute('WindowWidth'))
        self.assertNotEqual(si.getDicomAttribute('WindowWidth'), si1.getDicomAttribute('WindowWidth'))

    #@unittest.skip("skipping test_cross_talk_series_template")
    def test_cross_talk_series_template(self):
        si = Series('data/dicom/time/time00')
        si1 = si.deepcopy()
        si1.setDicomAttribute('WindowWidth', 1)
        self.assertNotEqual(si.getDicomAttribute('WindowWidth'), si1.getDicomAttribute('WindowWidth'))

    #@unittest.skip("skipping test_cross_talk_spacing")
    def test_cross_talk_spacing(self):
        si = Series('data/dicom/time', 'time')
        si1 = si[0]
        si1.spacing = (1,1,1)
        self.assertNotEqual(si.spacing.tolist(), si1.spacing.tolist())

    #@unittest.skip("skipping test_cross_talk_2")
    def test_cross_talk_2(self):
        si1 = Series('data/dicom/time/time00')
        si2 = si1
        si2.seriesNumber += 10
        self.assertEqual(si1.seriesNumber, si2.seriesNumber)

    #@unittest.skip("skipping test_cross_talk_3")
    def test_cross_talk_3(self):
        si1 = Series('data/dicom/time/time00')
        si2 = copy.copy(si1)
        si2.seriesNumber += 10
        self.assertNotEqual(si1.seriesNumber, si2.seriesNumber)

    #@unittest.skip("skipping test_set_axes")
    def test_set_axes(self):
        si1 = Series('data/dicom/time/time00')

        shape = si1.shape + (3,)
        img = np.zeros(shape, dtype=np.uint8)
        img[...,0] = si1[:]
        img[...,1] = si1[:]
        img[...,2] = si1[:]

        rgb = Series(img, geometry=si1,
                     axes=si1.axes + [imagedata.axis.VariableAxis('rgb',['r', 'g', 'b'])]
                     )

    #@unittest.skip("skipping test_header_axes")
    def test_header_axes(self):
        geometry = Series('data/dicom/time/time00')
        si = Series(np.eye(128), geometry=geometry)
        self.assertEqual(len(si.axes), 2)
        for i in range(len(si.axes)):
            self.assertEqual(len(si.axes[i]), si.shape[i])


if __name__ == '__main__':
    unittest.main()
