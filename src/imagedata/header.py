"""Image series header

"""

import copy
import logging
import pydicom.dataset
import pydicom.datadict
import imagedata.formats
import imagedata.formats.dicomlib.uid


logger = logging.getLogger(__name__)

header_tags = ['input_format',
               'seriesNumber',
               'seriesDescription', 'imageType', 'frameOfReferenceUID',
               'studyInstanceUID', 'studyID', 'seriesInstanceUID',
               'SOPClassUID',
               'accessionNumber',
               'patientName', 'patientID', 'patientBirthDate',
               'input_sort']
geometry_tags = ['sliceLocations', 'tags', 'spacing',
                 'imagePositions', 'orientation', 'transformationMatrix',
                 'color', 'photometricInterpretation', 'axes']


class Header(object):
    """Image header object.

    Attributes:
        input_order
        sort_on
        input_format
        DicomHeaderDict
        seriesNumber
        seriesDescription
        imageType
        frameOfReferenceUID
        studyInstanceUID
        studyID
        seriesInstanceUID
        SOPClassUID
        accessionNumber
        patientName
        patientID
        patientBirthDate
        input_sort
        sliceLocations
        tags
        spacing
        imagePositions
        orientation
        transformationMatrix
        color
        photometricInterpretation
        axes
        __uid_generator
        studyInstanceUID
        seriesInstanceUID
        frameOfReferenceUID
        DicomHeaderDict
        tags
        axes
    """

    def __init__(self):
        """Initialize image header attributes to defaults

        This is created in Series.__array_finalize__() and Series.__new__()
        """
        object.__init__(self)
        self.input_order = imagedata.formats.INPUT_ORDER_NONE
        self.sort_on = None
        for attr in header_tags + geometry_tags:
            try:
                setattr(self, attr, None)
            except AttributeError:
                pass
        self.__uid_generator = imagedata.formats.dicomlib.uid.get_uid()
        self.studyInstanceUID = self.new_uid()
        self.seriesInstanceUID = self.new_uid()
        self.frameOfReferenceUID = self.new_uid()
        self.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
        # pydicom.datadict.DicomDictionary
        # from pydicom.uid import UID
        # UID.
        self.color = False
        self.DicomHeaderDict = None
        self.tags = None
        self.axes = None

    def __repr__(self):
        return object.__repr__(self)

    def __str__(self):
        items = []
        for attr in header_tags + geometry_tags:
            items.append("{0!r}: {1!r}".format(attr, getattr(self, attr, "")))
        return "{" + ", ".join(items) + "}"

    def new_uid(self) -> str:
        """Return the next available UID from the UID generator.
        """
        return self.__uid_generator.__next__()

    def set_default_values(self, axes):
        """Set default values.
        """
        self.color = False
        if self.DicomHeaderDict is not None:
            return
        self.axes = copy.copy(axes)

        slices = tags = 1
        if axes is not None:
            for axis in axes:
                if axis.name == 'slice':
                    slices = len(axis)
                elif axis.name not in {'row', 'column', 'rgb'}:
                    tags = len(axis)

            self.DicomHeaderDict = {}
            i = 0
            for _slice in range(slices):
                self.DicomHeaderDict[_slice] = []
                for tag in range(tags):
                    self.DicomHeaderDict[_slice].append(
                        (tag, None, self._empty_ds())
                    )
                    i += 1
            if self.tags is None:
                self.tags = {}
                for _slice in range(slices):
                    self.tags[_slice] = [i for i in range(tags)]

    # noinspection PyPep8Naming
    def _empty_ds(self):
        SOPInsUID = self.new_uid()

        ds = pydicom.dataset.Dataset()

        # Add the data elements
        ds.StudyInstanceUID = self.studyInstanceUID
        ds.StudyID = '1'
        ds.SeriesInstanceUID = self.seriesInstanceUID
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
        ds.SOPInstanceUID = SOPInsUID
        ds.FrameOfReferenceUID = self.frameOfReferenceUID

        ds.PatientName = 'ANONYMOUS'
        ds.PatientID = 'ANONYMOUS'
        ds.PatientBirthDate = '00000000'
        ds.AccessionNumber = ''
        ds.Modality = 'SC'

        return ds


def add_template(this, template):
    """Add template data to this header.
    Does not add geometry data.

    Args:
        this: header or dict
        template: template header or dict. Can be None.
    Raises:
        ValueError: When the template is not a Header or dict.
    """

    if template is None:
        return
    if not issubclass(type(this), Header) and not issubclass(type(this), dict):
        raise ValueError('Object is not Header or dict.')
    for attr in __attributes(template):
        if attr in header_tags and attr not in ['seriesInstanceUID', 'input_format']:
            __set_attribute(this, attr, __get_attribute(template, attr))
    # Make sure DicomHeaderDict is set last
    template_dhd = copy.deepcopy(__get_attribute(template, 'DicomHeaderDict'))
    if template_dhd is not None:
        __set_attribute(this, 'DicomHeaderDict',
                        __make_DicomHeaderDict_from_template(this, template_dhd))


def __get_tags_and_slices(obj):
    slices = tags = 1
    try:
        axes = __get_attribute(obj, 'axes')
    except Exception as e:
        print(e)
        raise
    for axis in axes:
        if axis.name == 'slice':
            slices = len(axis)
        elif axis.name not in {'row', 'column', 'rgb'}:
            tags = len(axis)
    return tags, slices


def __make_DicomHeaderDict_from_template(this, template):
    DicomHeaderDict = {}
    default_header = template[0][0][2]
    tags, slices = __get_tags_and_slices(this)
    for _slice in range(slices):
        DicomHeaderDict[_slice] = []
        for tag in range(tags):
            try:
                template_tag = template[_slice][tag][0]
            except KeyError:
                template_tag = tag
            try:
                templateHeader = template[_slice][tag][2]
            except KeyError:
                templateHeader = default_header
            DicomHeaderDict[_slice].append((template_tag, None, templateHeader))
    return DicomHeaderDict


def __make_tags_from_template(this, template, geometry):
    tag_dict = {}
    tags, slices = __get_tags_and_slices(this)
    for _slice in range(slices):
        tag_dict[_slice] = []
        if issubclass(type(geometry[_slice]), dict):
            geometry_tag_list = list(geometry[_slice].values())
        else:
            geometry_tag_list = list(geometry[_slice])
        template_tag_list = []
        if template is not None:
            if issubclass(type(template[_slice]), dict):
                template_tag_list = list(template[_slice].values())
            else:
                template_tag_list = list(template[_slice])
        if len(geometry_tag_list) >= tags:
            tag_dict[_slice] = geometry_tag_list[:tags]
        elif len(template_tag_list) >= tags:
            tag_dict[_slice] = template_tag_list[:tags]
        else:
            raise IndexError('Cannot get tag list with length {}'.format(tags))
    return tag_dict


def __make_axes_from_template(this, template_axes, geometry_axes):
    axes = __get_attribute(this, 'axes')
    for i, axis in enumerate(axes):
        if geometry_axes is not None:
            for geometry_axis in geometry_axes:
                if geometry_axis.name == axis.name:
                    axes[i] = copy.copy(geometry_axis)
        elif template_axes is not None:
            for template_axis in template_axes:
                if template_axis.name == axis.name:
                    axes[i] = copy.copy(template_axis)
    return axes


def add_geometry(this, template, geometry):
    """Add geometry data to this header.

    Args:
        this: header or dict
        template: template header or dict. Can be None.
        geometry: geometry template header or dict. Can be None.
    Raises:
        ValueError: When the template is not a Header or dict.
    """

    if geometry is None:
        return
    if not issubclass(type(this), Header) and not issubclass(type(this), dict):
        raise ValueError('Object is not Header or dict.')
    for attr in __attributes(geometry):
        if attr in geometry_tags and attr not in ['tags', 'axes', 'input_format']:
            __set_attribute(this, attr, __get_attribute(geometry, attr))
    # Make sure tags and axes are set last
    __set_attribute(this, 'tags',
                    __make_tags_from_template(
                        this,
                        __get_attribute(template, 'tags'),
                        __get_attribute(geometry, 'tags')
                    ))
    __set_attribute(this, 'axes',
                    __make_axes_from_template(
                        this,
                        __get_attribute(template, 'axes'),
                        __get_attribute(geometry, 'axes')
                    ))
    return


def __attributes(obj):
    if issubclass(type(obj), Header):
        for attr in obj.__dict__:
            yield attr
    elif issubclass(type(obj), dict):
        for attr in obj:
            yield attr
    else:
        raise ValueError('Template is not Header nor dict.')


def __get_attribute(obj, name):
    if obj is None:
        return None
    if issubclass(type(obj), Header):
        return getattr(obj, name, None)
    elif issubclass(type(obj), dict):
        return copy.copy(obj[name])
    else:
        raise ValueError('Object is not Header nor dict.')


def __set_attribute(obj, name, value):
    if issubclass(type(obj), Header):
        setattr(obj, name, value)
    elif issubclass(type(obj), dict):
        obj[name] = value
    else:
        raise ValueError('Source is not Header nor dict.')
