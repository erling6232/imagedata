import unittest
import os.path
import tempfile
from collections import namedtuple
import pickle
import numpy as np
from numpy.random import default_rng
import copy
import pydicom.datadict

# from .context import imagedata
from src.imagedata.series import Series
import src.imagedata.axis as axis
import src.imagedata.formats as formats
from src.imagedata.viewer import grid_from_roi
from .compare_headers import compare_axes


class TestSeries(unittest.TestCase):

    def test_repr(self):
        si = Series(
            'data/dicom/time/time00/Image_00020.dcm')
        r = si.__repr__()

    def test_repr_vol(self):
        si = Series(
            'data/dicom/time/time00')
        r = si.__repr__()

    def test_str(self):
        si = Series(
            'data/dicom/time/time00/Image_00020.dcm')
        r = '{}'.format(si)

    def test_max(self):
        si = Series(
            'data/dicom/time/time00/Image_00020.dcm')
        mi = si.max()
        self.assertEqual(type(mi), np.uint16)

    def test_kwargs(self):
        def _read_series():
            si2 = Series(
                'data/dicom/time/time00/Image_00020.dcm',
                input_format='dicom',
                input_echo=2)

        si1 = Series(
            'data/dicom/time/time00/Image_00020.dcm',
            input_format='dicom',
            input_echo=1)
        self.assertRaises(formats.UnknownInputError, _read_series)

    def test_get_keyword(self):
        si1 = Series(
            'data/dicom/time/time00/Image_00020.dcm')
        self.assertEqual('dicom', si1.input_format)
        pname = si1.getDicomAttribute('PatientName')
        self.assertEqual(
            si1.getDicomAttribute('PatientName'),
            'PHANTOM^T1')
        self.assertEqual(
            si1.getDicomAttribute(
                pydicom.datadict.tag_for_keyword('PatientID')),
            '19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035')

    def test_create_series_1(self):
        si = Series(np.uint16(1))
        self.assertEqual(np.uint16, si.dtype)
        self.assertEqual((1,), si.shape)

    def test_create_series_tuple_1D(self):
        si = Series((1, 2, 3))
        self.assertEqual(np.int64, si.dtype)
        self.assertEqual((3,), si.shape)

    def test_create_series(self):
        a = np.eye(128)
        si = Series(a)
        self.assertEqual(si.dtype, np.float64)
        self.assertEqual(si.shape, (128, 128))

    def test_create_series_5D(self):
        rng = default_rng()
        s = Series(rng.standard_normal(2*3*4*4*4).reshape((2,3,4,4,4)), 'time,te')

    def test_slicing_series_5D(self):
        rng = default_rng()
        s = Series(rng.standard_normal(2*3*4*4*4).reshape((2,3,4,4,4)), 'time,te')
        s1 = s[1]
        self.assertEqual(s.axes[1:], s1.axes)
        s2 = s[:,1]
        s_axes = tuple([s.axes[_] for _ in (0, 2, 3, 4)])
        self.assertEqual(s_axes, s2.axes)

    def test_print_header(self):
        a = np.eye(128)
        si = Series(a)
        si.spacing = (1, 1, 1)
        self.assertEqual(si.slices, 1)
        np.testing.assert_array_equal(si.spacing, np.array((1, 1, 1)))

    def test_get_float_voxel(self):
        rng = default_rng()
        s = Series(rng.standard_normal(64).reshape((4,4,4)), dtype=np.float64)
        v = s[2, 2, 2]
        t = s[((2,), (2,), (2,))]
        pass

    def test_dtype_int(self):
        rng = default_rng()
        s = Series(rng.standard_normal(64).reshape((4,4,4))*100, dtype=int)
        self.assertEqual(s.dtype, int)

    def test_dtype_float64(self):
        rng = default_rng()
        s = Series((rng.standard_normal(64).reshape((4,4,4))*100).astype(int), dtype=np.float64)
        self.assertEqual(s.dtype, np.float64)

    def test_copy_series(self):
        a = np.eye(128)
        si1 = Series(a)
        si1.spacing = (1, 1, 1)

        si2 = si1
        self.assertEqual(si2.slices, 1)
        np.testing.assert_array_equal(si2.spacing, np.array((1, 1, 1)))

    def test_copy_series2(self):
        a = np.eye(128)
        si1 = Series(a)
        si1.spacing = (1, 1, 1)

        si2 = si1.copy()
        self.assertEqual(si2.slices, 1)
        np.testing.assert_array_equal(si2.spacing, np.array((1, 1, 1)))

    def test_subtract_series(self):
        a = Series(np.eye(128))
        b = Series(np.eye(128))
        a.spacing = (1, 1, 1)
        b.spacing = a.spacing

        si = a - b
        self.assertEqual(si.slices, 1)
        np.testing.assert_array_equal(si.spacing, np.array((1, 1, 1)))

    def test_increase_ndim(self):
        a = np.eye(128)
        s = Series(a)
        with self.assertRaises(IndexError):
            s.shape = (1,1,128,128)

    def test_set_variable_slice_locations(self):
        s = Series(np.zeros((3,12,12)))
        new_loc = np.array([1, 3, 6])
        s.sliceLocations = new_loc
        np.testing.assert_array_equal(new_loc, s.sliceLocations)

    def test_set_incorrect_slice_locations(self):
        s = Series(np.zeros((3,12,12)))
        with self.assertRaises(ValueError):
            s.sliceLocations = [3, 6]

    def test_slicing_dim(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a = np.vstack([a1, a1, a1, a1, a1])
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[0]))

        s_slice = s[2]
        self.assertEqual(s_slice.ndim, 2)
        self.assertEqual(len(s_slice.axes), 2)
        self.assertEqual(s_slice.axes[0].name, 'row')
        self.assertEqual(s_slice.axes[1].name, 'column')

    def test_slicing_y(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a = np.vstack([a1, a1, a1])
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[0]))

        a_slice = a[:,:,3:5]
        s_slice = s[:,:,3:5]
        np.testing.assert_array_equal(a_slice, s_slice)
        self.assertEqual(s_slice.slices, 3)

    def test_slicing_y_neg(self):
        rng = default_rng()
        s = Series(rng.standard_normal(64).reshape((4,4,4)))
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[0]))
        np.testing.assert_array_equal(s[:,3,:], s[:,-1,:])
        np.testing.assert_array_equal(s[:,2,:], s[:,-2,:])
        np.testing.assert_array_equal(s[:,1,:], s[:,-3,:])
        np.testing.assert_array_equal(s[:,0,:], s[:,-4,:])
        np.testing.assert_array_equal(s[:,2:3,:], s[:,2:-1,:])
        np.testing.assert_array_equal(s[:,1:2,:], s[:,1:-2,:])
        np.testing.assert_array_equal(s[:,0:2,:], s[:,0:-2,:])

    def test_slicing_x(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a = np.vstack([a1, a1, a1])
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[0]))

        a_slice = a[:,3:5,...]
        s_slice = s[:,3:5,...]
        np.testing.assert_array_equal(a_slice, s_slice)
        self.assertEqual(s_slice.slices, 3)

    def test_slicing_x_neg(self):
        rng = default_rng()
        s = Series(rng.standard_normal(64).reshape((4,4,4)))
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[0]))
        np.testing.assert_array_equal(s[:,:,3], s[:,:,-1])
        np.testing.assert_array_equal(s[:,:,2], s[:,:,-2])
        np.testing.assert_array_equal(s[:,:,1], s[:,:,-3])
        np.testing.assert_array_equal(s[:,:,0], s[:,:,-4])
        np.testing.assert_array_equal(s[:,:,2:3], s[:,:,2:-1])
        np.testing.assert_array_equal(s[:,:,1:2], s[:,:,1:-2])
        np.testing.assert_array_equal(s[:,:,0:2], s[:,:,0:-2])

    def test_assign_slice_x(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a = np.vstack([a1, a1, a1])
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[0]))
        n = np.ones_like(a) * 4
        p = Series(n)
        p.spacing = (1,1,1)
        p.axes = p.axes._replace(slice=axis.UniformLengthAxis('slice', 0, p.shape[0]))

        a[:,3:5,...] = n[:,3:5,...]
        s[:,3:5,...] = p[:,3:5,...]
        np.testing.assert_array_equal(a, s)
        np.testing.assert_array_equal(s[:,3:5,...], p[:,3:5,...])
        self.assertEqual(s.slices, 3)

    def test_assign_slice(self):
        a = np.array(range(4*5*6), dtype=np.uint16)
        a.shape = (4,5,6)
        s = Series(a)
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[0]))
        n = np.zeros((2,2,2), dtype=np.uint16)
        p = Series(n)
        p.spacing = (1,1,1)
        p.axes = p.axes._replace(slice=axis.UniformLengthAxis('slice', 0, p.shape[0]))

        a[1:3,2:4,2:4] = n[:]
        s[1:3,2:4,2:4] = p[:]
        np.testing.assert_array_equal(a, s)
        self.assertEqual(s.slices, 4)

    def test_assign_slice_input_order(self):
        si = Series('data/dicom/time', 'time')
        sic = Series(si)
        self.assertEqual(si.input_order, sic.input_order)

    def test_zeros_like(self):
        si = Series('data/dicom/time', 'time')
        a = np.zeros_like(si)
        self.assertEqual(si.input_order, a.input_order)
        for i in range(si.ndim):
            self.assertEqual(si.axes[i].name, a.axes[i].name)
            try:
                np.testing.assert_array_almost_equal(si.axes[i].values, a.axes[i].values, 4)
            except AttributeError:
                self.assertEqual(si.axes[i].slice, a.axes[i].slice)
        np.testing.assert_array_equal(si.transformationMatrix, a.transformationMatrix)
        np.testing.assert_array_equal(si.spacing, a.spacing)
        a[:, :, 10:50, 10:50]  = 1

    def test_sum_with_series_mask(self):
        si = Series('data/dicom/time', 'time')
        mask = np.zeros(si.shape[1:], dtype=np.uint8)
        mask[:, 10:20, 10:20] = 1
        mask = Series(mask, input_order=si.input_order,
                      template=si, geometry=si)
        time_course = np.sum(np.array(si),
                             axis=(1, 2, 3), where=mask == 1) / np.count_nonzero(mask)

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

    def test_slicing_z_neg(self):
        rng = default_rng()
        s = Series(rng.standard_normal(64).reshape((4,4,4)))
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[0]))
        np.testing.assert_array_equal(s[3,:,:], s[-1,:,:])
        np.testing.assert_array_equal(s[2,:,:], s[-2,:,:])
        np.testing.assert_array_equal(s[1,:,:], s[-3,:,:])
        np.testing.assert_array_equal(s[0,:,:], s[-4,:,:])
        np.testing.assert_array_equal(s[2:3], s[2:-1])
        np.testing.assert_array_equal(s[2:3,:,:], s[2:-1,:,:])
        np.testing.assert_array_equal(s[1:2,:,:], s[1:-2,:,:])
        np.testing.assert_array_equal(s[0:2,:,:], s[0:-2,:,:])

    def test_slicing_t(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a2 = np.vstack([a1, a1, a1])
        a2.shape = (1,3,128,128)
        a = np.vstack([a2,a2,a2,a2])
        s = Series(a, input_order=formats.INPUT_ORDER_TIME)
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(time=axis.UniformLengthAxis('time', 0, s.shape[0]))
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[1]))
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

    def test_slicing_t_neg(self):
        rng = default_rng()
        s = Series(rng.standard_normal(192).reshape((3,4,4,4)))
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(none=axis.UniformLengthAxis('none', 0, s.shape[0]))
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[1]))
        np.testing.assert_array_equal(s[2], s[-1])
        np.testing.assert_array_equal(s[1], s[-2])
        np.testing.assert_array_equal(s[0], s[-3])
        np.testing.assert_array_equal(s[1:2], s[1:-1])
        np.testing.assert_array_equal(s[1:2,:,:], s[1:-1,:,:])
        np.testing.assert_array_equal(s[0:1,:,:], s[0:-2,:,:])

    def test_slicing_t_drop(self):
        rng = default_rng()
        s = Series(rng.standard_normal(192).reshape((3,4,4,4)), 'time')
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(time=axis.UniformLengthAxis('time', 0, s.shape[0]))
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[1]))
        s_axes = copy.copy(s.axes)
        self.assertEqual(len(s_axes), 4)

        sum = np.sum(s, axis=0)
        self.assertEqual(len(s.axes), len(s_axes))
        compare_axes(self, s.axes, s_axes)
        self.assertEqual(len(s_axes), 4)
        compare_axes(self, s_axes, sum.axes)

    def test_newaxis(self):
        rng = default_rng()
        s = Series(rng.standard_normal(10))
        s = s[:, np.newaxis]
        assert type(s) == Series
        assert s.shape == (10, 1)
        assert len(s.axes) == 2

    def test_newaxis_4d(self):
        rng = default_rng()
        s = Series(rng.standard_normal(192).reshape((3,4,4,4)), 'time')
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(time=axis.UniformLengthAxis('time', 0, s.shape[0]))
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[1]))
        sum = np.sum(s, axis=(1, 2, 3))
        sum = sum[:, np.newaxis]

    def test_multiple_ellipses(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a2 = np.vstack([a1, a1, a1])
        a2.shape = (1,3,128,128)
        a = np.vstack([a2,a2,a2,a2])
        s = Series(a, input_order=formats.INPUT_ORDER_TIME)
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(time=axis.UniformLengthAxis('time', 0, s.shape[0]))
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[1]))
        tags = {}
        k = 0
        for i in range(s.slices):
            tags[i] = np.arange(k, k+s.shape[0])
            k += s.shape[0]
        s.tags = tags

        with self.assertRaises(IndexError):
            s_slice = s[...,1:3,...]

    def test_ellipsis_first(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a2 = np.vstack([a1, a1, a1])
        a2.shape = (1,3,128,128)
        a = np.empty([4, 3,128,128])
        for i in range(4):
            a[i] = a2
        s = Series(a, input_order=formats.INPUT_ORDER_TIME)
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(time=axis.UniformLengthAxis('time', 0, s.shape[0]))
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[1]))
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

    def test_ellipsis_middle(self):
        a1 = np.eye(128)
        a1.shape = (1,128,128)
        a2 = np.vstack([a1, a1, a1])
        a2.shape = (1,3,128,128)
        a = np.vstack([a2,a2,a2,a2])
        s = Series(a, input_order=formats.INPUT_ORDER_TIME)
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(time=axis.UniformLengthAxis('time', 0, s.shape[0]))
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[1]))
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

    def test_tags_after_z_slicing(self):
        rng = default_rng()
        s = Series(rng.standard_normal(3*5*4*4).reshape((3,5,4,4)), 'time')
        s.spacing = (1, 1, 1)
        s.axes = s.axes._replace(slice=axis.UniformLengthAxis('slice', 0, s.shape[1]))
        s.axes = s.axes._replace(time=axis.VariableAxis('time', [1.1, 1.2, 1.3]))
        t = s[:, 2:4]
        assert t.tags.keys() == {0, 1}

    def test_cross_talk(self):
        si = Series('data/dicom/time/time00')
        self.assertEqual('dicom', si.input_format)
        # print('si before', si.getDicomAttribute('SeriesInstanceUID'), si.seriesInstanceUID)
        si1 = si[0]
        si1.seriesNumber = si.seriesNumber + 10
        self.assertNotEqual(si.seriesNumber, si1.seriesNumber)
        # print('si after', si.getDicomAttribute('SeriesInstanceUID'), si.seriesInstanceUID)
        # print('si1', si1.getDicomAttribute('SeriesInstanceUID'), si1.seriesInstanceUID)
        self.assertEqual(si.seriesInstanceUID, si1.seriesInstanceUID)

        si2 = Series('data/dicom/time/time00')
        self.assertEqual('dicom', si2.input_format)
        si2.seriesNumber += 10
        self.assertNotEqual(si.seriesNumber, si2.seriesNumber)

    def test_cross_talk_wl_ref(self):
        si = Series('data/dicom/time/time00')
        self.assertEqual('dicom', si.input_format)
        si1 = si[0] * 10
        self.assertNotEqual(si.windowWidth, si1.windowWidth)

    def test_cross_talk_wl(self):
        si = Series('data/dicom/time/time00')
        self.assertEqual('dicom', si.input_format)
        si1 = copy.deepcopy(si)[0] * 10
        self.assertNotEqual(si.windowWidth, si1.windowWidth)

    def test_cross_talk_series_ref(self):
        si = Series('data/dicom/time/time00')
        self.assertEqual('dicom', si.input_format)
        si1 = Series(si, input_order=si.input_order)
        si1.windowWidth = 1
        with tempfile.TemporaryDirectory() as d:
            si1.write(d, formats=['dicom'])
            si2 = Series(d)
        np.testing.assert_array_almost_equal(si2.windowWidth, si1.windowWidth, 4)

    def test_cross_talk_dicom_series_template(self):
        template = Series('data/dicom/time/time00')
        template_window = template.windowWidth
        si = Series('data/dicom/time/time01', template=template)
        si1_window = si.windowWidth
        self.assertEqual('dicom', si.input_format)
        si.windowWidth = 1
        with tempfile.TemporaryDirectory() as d:
            si.write(d, formats=['dicom'])
            si2 = Series(d)
        self.assertNotEqual(si2.windowWidth, template.windowWidth)

    def test_cross_talk_series_template(self):
        template = Series('data/dicom/time/time00')
        self.assertEqual('dicom', template.input_format)
        template_window = template.windowWidth
        si = Series(np.zeros_like(template), template=template)
        si_window = si.windowWidth
        si.windowWidth = 1
        with tempfile.TemporaryDirectory() as d:
            si.write(d, formats=['dicom'])
            si2 = Series(d)
        self.assertNotEqual(si.windowWidth, template.windowWidth)
        self.assertEqual(si2.windowWidth, si.windowWidth)

    def test_cross_talk_spacing(self):
        si = Series('data/dicom/time', 'time')
        self.assertEqual('dicom', si.input_format)
        si1 = si[0]
        si1.spacing = (1,1,1)
        self.assertNotEqual(si.spacing.tolist(), si1.spacing.tolist())

    def test_cross_talk_2(self):
        si1 = Series('data/dicom/time/time00')
        self.assertEqual('dicom', si1.input_format)
        si2 = si1
        si2.seriesNumber += 10
        self.assertEqual(si1.seriesNumber, si2.seriesNumber)

    def test_cross_talk_3(self):
        si1 = Series('data/dicom/time/time00')
        self.assertEqual('dicom', si1.input_format)
        si2 = copy.copy(si1)
        si2.seriesNumber += 10
        self.assertNotEqual(si1.seriesNumber, si2.seriesNumber)

    def test_slice_column_row(self):
        si1 = Series(os.path.join('data', 'dicom', 'time'), formats.INPUT_ORDER_TIME)
        self.assertEqual('dicom', si1.input_format)
        selection = [0, 2, 4]
        si2 = si1[:, :, :, selection]
        np.testing.assert_array_equal(si1.axes[3][selection], si2.axes[3])
        si3 = si1[:, :, selection, :]
        np.testing.assert_array_equal(si1.axes[2][selection], si3.axes[2])
        selection = [0, 2]
        si4 = si1[:, selection]
        for i, idx in enumerate(selection):
            np.testing.assert_array_equal(si1.axes[1][idx], si4.axes[1][i])

    def test_slice_timeline(self):
        si1 = Series(os.path.join('data', 'dicom', 'time'), formats.INPUT_ORDER_TIME)
        self.assertEqual('dicom', si1.input_format)
        t = np.array([0., 2.99, 5.97])
        np.testing.assert_array_almost_equal(t, si1.timeline, decimal=2)
        si2 = si1[[0, 2]]
        np.testing.assert_array_equal(si1.timeline[[0, 2]], si2.timeline)
        a = np.array(si1.axes.time.values)
        np.testing.assert_array_equal(a[[0, 2]], si2.axes.time.values)

    def test_set_axes(self):
        si1 = Series('data/dicom/time/time00')
        self.assertEqual('dicom', si1.input_format)

        shape = si1.shape + (3,)
        img = np.zeros(shape, dtype=np.uint8)
        img[...,0] = si1[:]
        img[...,1] = si1[:]
        img[...,2] = si1[:]

        Axes = namedtuple('Axes', si1.axes._fields + ('rgb',))
        rgb = Series(img, geometry=si1,
                     axes=Axes._make(
                         si1.axes + (axis.VariableAxis('rgb',['r', 'g', 'b']),)
                     ))


    def test_header_axes(self):
        geometry = Series('data/dicom/time/time00')
        self.assertEqual('dicom', geometry.input_format)
        si = Series(np.eye(128), geometry=geometry)
        self.assertEqual(len(si.axes), 2)
        for i in range(len(si.axes)):
            self.assertEqual(len(si.axes[i]), si.shape[i])

    def test_get_rgb_voxel(self):
        si1 = Series('data/dicom/time/time00')
        self.assertEqual('dicom', si1.input_format)

        rgb = si1.to_rgb()
        _slice = rgb[1]
        voxel = _slice[1, 1]
        self.assertEqual(3, len(voxel))

    def test_get_rgb_voxel_np_rgb(self):
        si1 = Series(np.zeros((4,10,10,3), dtype=np.uint8))

        _slice = si1[1]
        voxel = _slice[1, 1]
        self.assertEqual(3, len(voxel))

    def test_get_rgb_voxel_np(self):
        si1 = Series(np.zeros((4,10,10), dtype=np.uint8))

        rgb = si1.to_rgb()
        _slice = rgb[1]
        voxel = _slice[1, 1]
        self.assertEqual(3, len(voxel))

    def test_fuse_mask_3d_bw_uint8(self):
        si1 = Series(np.zeros((4,10,10), dtype=float))
        mask = np.zeros_like(si1, dtype=np.uint8)
        mask[2, 2:7, 2:7] = 1
        fused = si1.fuse_mask(mask)
        self.assertEqual(3, fused.ndim)
        self.assertEqual((0, 0, 0), fused[1, 7, 7])
        self.assertEqual((58, 0, 0), fused[2, 3, 4])

    def test_fuse_mask_3d_bw_float(self):
        si1 = Series(np.zeros((4,10,10), dtype=float))
        mask = np.zeros_like(si1, dtype=np.uint8)
        mask[2, 2:7, 2:7] = 1
        fused = si1.fuse_mask(mask)
        self.assertEqual(3, fused.ndim)
        np.testing.assert_array_equal((0, 0, 0), fused[1, 7, 7])
        np.testing.assert_array_equal((58, 0, 0), fused[2, 3, 4])

    def test_fuse_mask_3d_rgb_uint8(self):
        si = Series(np.zeros((4,10,10), dtype=np.uint8))
        si1 = si.to_rgb()
        mask = np.zeros(si1.shape, dtype=np.uint8)
        mask[2, 3, 4] = 1
        fused = si1.fuse_mask(mask)
        self.assertEqual(3, fused.ndim)
        np.testing.assert_array_equal((0, 0, 0), fused[1, 7, 7])
        np.testing.assert_array_equal((5, 0, 0), fused[2, 3, 4])

    def test_fuse_mask_3d_rgb_float(self):
        si = Series(np.zeros((4,10,10), dtype=float))
        si1 = si.to_rgb()
        mask = np.zeros(si1.shape, dtype=np.uint8)
        mask[2, 3, 4] = 1
        fused = si1.fuse_mask(mask)
        self.assertEqual(3, fused.ndim)
        np.testing.assert_array_equal((0, 0, 0), fused[1, 7, 7])
        np.testing.assert_array_equal((5, 0, 0), fused[2, 3, 4])

    def test_fuse_mask_lena(self):
        # si1 = Series(Image.open(os.path.join('data', 'lena_color.jpg')))
        si1 = Series(os.path.join('data', 'dicom', 'lena_color.dcm'))
        # mask = np.zeros(si1.shape[:-1], dtype=np.uint8)
        mask = np.zeros(si1.shape, dtype=np.uint8)
        mask[100:200, 100:200] = 1
        fused = si1.fuse_mask(mask)
        self.assertEqual(2, fused.ndim)
        np.testing.assert_array_equal((255, 68, 80), fused[150, 150])
        np.testing.assert_array_equal((231, 136, 108), fused[50, 50])

    def test_fuse_mask_3d_variable(self):
        si1 = Series(np.zeros((4,100,100), dtype=float))

        N = 100
        x = np.linspace(-3.0, 3.0, N)
        y = np.linspace(-2.0, 2.0, N)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X**2 - Y**2)
        Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
        z = Z1 + 50 * Z2
        mask = np.zeros_like(si1, dtype=np.float64)
        mask[2, :100, :100] = z
        fused = si1.fuse_mask(mask)
        self.assertEqual(3, fused.ndim)
        np.testing.assert_array_equal((0, 0, 1), fused[1, 7, 7])
        np.testing.assert_array_equal((3, 2, 9), fused[2, 45, 50])
        np.testing.assert_array_equal((75, 76, 57), fused[2, 50, 50])

    def test_fuse_mask_3d_variable_maskrange(self):
        si1 = Series(np.zeros((4,100,100), dtype=float))

        N = 100
        x = np.linspace(-3.0, 3.0, N)
        y = np.linspace(-2.0, 2.0, N)
        X, Y = np.meshgrid(x, y)
        Z1 = np.exp(-X**2 - Y**2)
        Z2 = np.exp(-(X * 10)**2 - (Y * 10)**2)
        z = Z1 + 50 * Z2
        mask = np.zeros_like(si1, dtype=np.float64)
        mask[2, :100, :100] = z
        fused = si1.fuse_mask(mask, maskrange=(20, 44))
        self.assertEqual(3, fused.ndim)
        np.testing.assert_array_equal((0, 0, 1), fused[1, 7, 7])
        np.testing.assert_array_equal((0, 0, 1), fused[2, 45, 50])
        np.testing.assert_array_equal((75, 76, 57), fused[2, 50, 50])

    def test_align_3d(self):
        reference = Series(
            os.path.join('data', 'dicom', 'time', 'time00')
        )
        moving = Series(
            os.path.join('data', 'dicom', 'time', 'time01')
        )
        moved = moving.align(reference)
        with tempfile.TemporaryDirectory() as d:
            moved.write(d, formats=['dicom'])

    def test_align_3d_few_slices_on_many(self):
        rng = default_rng()
        reference = Series(rng.standard_normal(80).reshape((5,4,4)))
        reference.spacing = (1, 1, 1)
        reference.axes = reference.axes._replace(
            slice=axis.UniformLengthAxis('slice', 0, reference.shape[0])
        )
        moving = Series(
            os.path.join('data', 'dicom', 'time', 'time01')
        )
        moved = moving.align(reference, force=True)
        with tempfile.TemporaryDirectory() as d:
            moved.write(d, formats=['dicom'])

    def test_align_2d(self):
        reference = Series(
            os.path.join('data', 'dicom', 'time', 'time00')
        )
        moving = Series(
            os.path.join('data', 'dicom', 'TI',
                         'TI_1.MR.0021.0001.2021.06.08.10.04.29.806302.203193459.IMA'),
            input_order='ti',
            opts={'ti': 'InversionTime'}
        )
        with self.assertRaises(ValueError):
            moved = moving.align(reference)

    def test_align_3d_on_4d(self):
        reference = Series(
            os.path.join('data', 'dicom', 'time')
        )
        moving = Series(
            os.path.join('data', 'dicom', 'time', 'time01')
        )
        moved = moving.align(reference)
        with tempfile.TemporaryDirectory() as d:
            moved.write(d, formats=['dicom'])

    def test_align_4d_on_3d(self):
        moving = Series(
            os.path.join('data', 'dicom', 'time')
        )
        reference = Series(
            os.path.join('data', 'dicom', 'time', 'time01')
        )
        moved = moving.align(reference)
        with tempfile.TemporaryDirectory() as d:
            moved.write(d, formats=['dicom'])

    def test_numpy_mask(self):
        si = Series(os.path.join('data', 'dicom', 'time'))
        mask = si < 10
        curve = np.sum(si, axis=(1, 2, 3), where=mask==True)

    def test_vertices_from_3d_grid(self):
        si = Series(os.path.join('data', 'dicom', 'time', 'time01'))
        with open(os.path.join('data', 'vertices', 'vertices3d.pickle'), 'rb') as f:
            vertices = pickle.load(f)
        grid = grid_from_roi(si, vertices, single=False)
        new_vertices = grid.vertices_from_grid(grid, align=False)
        new_grid = grid_from_roi(si, new_vertices, single=False)
        # Accept 100 pixels mismatch
        diff = np.absolute(new_grid.astype(float) - grid.astype(float))
        s = np.sum(diff)
        self.assertLess(s, 100, "Too many mismatch pixels")

    def test_write_default_keep_uid_false(self):
        si = Series(os.path.join('data', 'dicom', 'time', 'time01'))
        serInsUid = si.seriesInstanceUID
        with tempfile.TemporaryDirectory() as d:
            si.write(d, formats=['dicom'])
            self.assertEqual(serInsUid, si.seriesInstanceUID)
            check = Series(d, input_format='dicom')
            self.assertNotEqual(serInsUid, check.seriesInstanceUID)

    def test_write_kwargs(self):
        si = Series(os.path.join('data', 'dicom', 'time', 'time01'))
        serInsUid = si.seriesInstanceUID
        with tempfile.TemporaryDirectory() as d:
            si.write(os.path.join(d, 'true'), formats=['dicom'], keep_uid=True)
            self.assertEqual(serInsUid, si.seriesInstanceUID)
            si.write(os.path.join(d, 'false'), formats=['dicom'], keep_uid=False)
            true = Series(os.path.join(d, 'true'), input_format='dicom')
            false = Series(os.path.join(d, 'false'), input_format='dicom')
            self.assertEqual(si.seriesInstanceUID, true.seriesInstanceUID)
            self.assertNotEqual(si.seriesInstanceUID, false.seriesInstanceUID)

if __name__ == '__main__':
    unittest.main()
    # import logging
    # logging.basicConfig(level=logging.DEBUG)
    # runner = unittest.TextTestRunner(verbosity=2)
    # unittest.main(testRunner=runner)
