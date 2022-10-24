import imagedata.axis
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
        self.assertEqual(hdr.DicomHeaderDict.keys(), newhdr.DicomHeaderDict.keys())
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
    self.assertEqual(len(axes), len(new_axes))
    for axis, new_axis in zip(axes, new_axes):
        self.assertEqual(type(axis), type(new_axis))
        self.assertEqual(axis.name, new_axis.name)
        if isinstance(axis, imagedata.axis.VariableAxis):
            np.testing.assert_array_equal(axis.values, new_axis.values)
        elif isinstance(axis, imagedata.axis.UniformLengthAxis):
            self.assertEqual(axis.n, new_axis.n)
            self.assertEqual(axis.start, new_axis.start)
            self.assertEqual(axis.stop, new_axis.stop)
            self.assertEqual(axis.step, new_axis.step)
        elif isinstance(axis, imagedata.axis.UniformAxis):
            self.assertEqual(axis.start, new_axis.start)
            self.assertEqual(axis.stop, new_axis.stop)
            self.assertEqual(axis.step, new_axis.step)