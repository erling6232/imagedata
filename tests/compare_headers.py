import src.imagedata.axis as axis
import numpy as np


def compare_headers(self, hdr, newhdr, uid=True):
    compare_template_headers(self, hdr, newhdr, uid)
    compare_geometry_headers(self, hdr, newhdr)


def compare_template_headers(self, hdr, newhdr, uid=True):
    # logging.debug('compare_headers: name {} {}'.format(hdr.name, newhdr.name))
    self.assertEqual(hdr.name, newhdr.name)
    # logging.debug('compare_headers: description {} {}'.format(hdr.description, newhdr.description))
    self.assertEqual(hdr.description, newhdr.description)
    self.assertEqual(hdr.authors, newhdr.authors)
    self.assertEqual(hdr.version, newhdr.version)
    self.assertEqual(hdr.url, newhdr.url)
    self.assertEqual(hdr.input_order, newhdr.input_order)
    # obj.assertEqual(hdr.sort_on, newhdr.sort_on)

    # DicomHeaderDict[slice].tuple(tagvalue, filename, dicomheader)
    try:
        self.assertEqual(hdr.dicomTemplate.keys(), newhdr.dicomTemplate.keys())
        # self.assertEqual(hdr.DicomHeaderDict.keys(), newhdr.DicomHeaderDict.keys())
        # for k in hdr.DicomHeaderDict.keys():
        #    obj.assertEqual(hdr.DicomHeaderDict[k], newhdr.DicomHeaderDict[k])
    except ValueError:
        pass
    self.assertEqual(hdr.tags.keys(), newhdr.tags.keys())
    for k in hdr.tags.keys():
        np.testing.assert_array_equal(hdr.tags[k], newhdr.tags[k])
    if uid:
        compare_optional(self, hdr, newhdr, 'studyInstanceUID')
        # compare_optional(obj, hdr, newhdr, 'seriesInstanceUID')
        compare_optional(self, hdr, newhdr, 'frameOfReferenceUID')
    compare_optional(self, hdr, newhdr, 'seriesNumber')
    compare_optional(self, hdr, newhdr, 'seriesDescription')
    compare_optional(self, hdr, newhdr, 'imageType')
    self.assertEqual(hdr.color, newhdr.color)
    self.assertEqual(hdr.photometricInterpretation,
                     newhdr.photometricInterpretation)


def compare_optional(self, a, b, attr):
    try:
        a_attr = getattr(a, attr, None)
    except ValueError:
        a_attr = None
    try:
        b_attr = getattr(b, attr, None)
    except ValueError:
        b_attr = None
    self.assertEqual(a_attr, b_attr)


def compare_geometry_headers(self, hdr, newhdr):
    try:
        np.testing.assert_array_equal(hdr.sliceLocations, newhdr.sliceLocations)
    except ValueError:
        pass
    np.testing.assert_array_almost_equal(hdr.spacing, newhdr.spacing, decimal=4)
    np.testing.assert_array_almost_equal(hdr.orientation, newhdr.orientation,
                                         decimal=4)
    self.assertEqual(hdr.imagePositions.keys(), newhdr.imagePositions.keys())
    for k in hdr.imagePositions.keys():
        # logging.debug('compare_headers:    hdr.imagePositions[{}]={}'.format(k,hdr.imagePositions[k]))
        # logging.debug('compare_headers: newhdr.imagePositions[{}]={}'.format(k,newhdr.imagePositions[k]))
        np.testing.assert_array_almost_equal(
            hdr.imagePositions[k],
            newhdr.imagePositions[k],
            decimal=4)
    np.testing.assert_array_almost_equal(hdr.transformationMatrix, newhdr.transformationMatrix, decimal=3)


def compare_axes(self, axes, new_axes):
    for _new_axis in new_axes:
        _axis = getattr(axes, _new_axis.name)
        self.assertEqual(type(_axis), type(_new_axis))
        self.assertEqual(_axis.name, _new_axis.name)
        if isinstance(_axis, axis.VariableAxis):
            np.testing.assert_array_equal(_axis.values, _new_axis.values)
        elif isinstance(_axis, axis.UniformLengthAxis):
            self.assertEqual(_axis.n, _new_axis.n)
            self.assertEqual(_axis.start, _new_axis.start)
            self.assertEqual(_axis.stop, _new_axis.stop)
            self.assertEqual(_axis.step, _new_axis.step)
        elif isinstance(_axis, axis.UniformAxis):
            self.assertEqual(_axis.start, _new_axis.start)
            self.assertEqual(_axis.stop, _new_axis.stop)
            self.assertEqual(_axis.step, _new_axis.step)


def compare_tags_in_slice(self, tags, new_tags, axis=None, slicing=None):
    if axis is None:
        raise ValueError('Axis direction must be defined')
    for _slice in tags.keys():
        if isinstance(axis, tuple):
            _t = tags[_slice]
            for i in reversed(axis):
                _slicing = slicing[i]
                if i == 2:
                    _t = _t[:, :, _slicing]
                elif i == 1:
                    _t = _t[:, _slicing]
                elif i == 0:
                    _t = _t[_slicing]
                else:
                    raise ValueError('Too many axes')
        elif axis == 0:
            _t = tags[_slice][ slicing]
        elif axis == 1:
            _t = tags[_slice][:, slicing]
        elif axis == 2:
            _t = tags[_slice][:, :, slicing]
        else:
            raise ValueError('Axis direction must be defined')
        for tag in np.ndindex(new_tags[_slice].shape):
            if _t[tag] is None and new_tags[_slice][tag] is None:
                continue
            for t,n in zip(_t[tag], new_tags[_slice][tag]):
                if issubclass(type(t), np.ndarray):
                    if t.size == n.size == 0:
                        continue
                    else:
                        np.testing.assert_array_almost_equal(t, n)
                else:
                    self.assertAlmostEqual(t, n)


def compare_tags(self, tags, new_tags):
    for _slice in tags.keys():
        for tag in np.ndindex(new_tags[_slice].shape):
            if tags[_slice][tag] is None and new_tags[_slice][tag] is None:
                continue
            for t,n in zip(tags[_slice][tag], new_tags[_slice][tag]):
                if issubclass(type(t), np.ndarray):
                    if t.size == n.size == 0:
                        continue
                    else:
                        np.testing.assert_array_almost_equal(t, n)
                else:
                    self.assertAlmostEqual(t, n)


def compare_pydicom(self, orig, temp, uid=False):
    dont_verify = ['Content Date', 'Content Time', 'Instance Number',
                   'Largest Pixel Value in Series', 'Window Center',
                   'Window Width', 'Acquisition Number']
    for data_element in orig:
        if data_element.VR == "SQ":
            pass
        elif data_element.name in dont_verify:
            pass
        elif data_element.VR == "DS":
            tag = data_element.tag
            np.testing.assert_allclose(
                data_element.value, temp[tag].value,
                rtol=1e-2,
                err_msg='Name="{}", VR={}'.format(
                    data_element.name, data_element.VR)
            )
        else:
            tag = data_element.tag
            if not uid and data_element.VR == "UI":
                continue
            try:
                self.assertEqual(data_element.value, temp[tag].value,
                             msg='Name="{}", VR={}'.format(
                                 data_element.name, data_element.VR)
                             )
            except AssertionError as e:
                if data_element.private_creator == 'SIEMENS CSA HEADER':
                    pass
                else:
                    raise
    np.testing.assert_array_almost_equal(orig.SliceLocation, temp.SliceLocation)
    np.testing.assert_array_almost_equal(
        orig.ImagePositionPatient, temp.ImagePositionPatient)

