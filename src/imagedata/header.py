"""Image series header

"""

import copy
import logging
import pydicom.dataset
import pydicom.datadict
import imagedata.formats
import imagedata.formats.dicomlib.uid

# from imagedata.series import Series

logging.getLogger(__name__).addHandler(logging.NullHandler())

header_tags = ['input_format',
               'seriesNumber',
               'seriesDescription', 'imageType', 'frameOfReferenceUID',
               'studyInstanceUID', 'studyID', 'seriesInstanceUID',
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
        # self.axes = list()
        # for d in range(len(shape)):
        #    self.axes.append(imagedata.axis.UniformAxis('%d' % d,
        #                                                0,
        #                                                shape[d]))
        self.axes = copy.copy(axes)
        # logging.debug('Header.set_default_values: study  UID {}'.format(self.studyInstanceUID))
        # logging.debug('Header.set_default_values: series UID {}'.format(self.seriesInstanceUID))

        slices = tags = 1
        if axes is not None:
            for axis in axes:
                if axis.name == 'slice':
                    slices = len(axis)
                elif axis.name not in {'row', 'column', 'rgb'}:
                    tags = len(axis)

            self.DicomHeaderDict = {}
            i = 0
            # logging.debug('Header.set_default_values %d tags' % tags)
            # logging.debug('Header.set_default_values tags {}'.format(self.tags))
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
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # SC
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
    if issubclass(type(template), Header):
        for attr in template.__dict__:
            if attr in header_tags and attr != 'seriesInstanceUID':
                setattr(this, attr, getattr(template, attr, None))
        # Make sure DicomHeaderDict is set last
        templateDHD = getattr(template,'DicomHeaderDict',None)
        if templateDHD is not None:
            setattr(this, 'DicomHeaderDict',
                    __make_DicomHeaderDict_from_template(this, getattr(template, 'DicomHeaderDict', None)))
    elif issubclass(type(template), dict):
        for attr in template:
            if attr in header_tags and attr != 'seriesInstanceUID':
                if issubclass(type(this), Header):
                    setattr(this, attr, copy.copy(template[attr]))
                elif issubclass(type(this), dict):
                    this[attr] = copy.copy(template[attr])
        # Make sure DicomHeaderDict is set last
        if 'DicomHeaderDict' in template:
            if issubclass(type(this), Header):
                setattr(this, 'DicomHeaderDict',
                        __make_DicomHeaderDict_from_template(this, getattr(template, 'DicomHeaderDict', None)))
            elif issubclass(type(this), dict):
                this['DicomHeaderDict'] = copy.copy(__make_DicomHeaderDict_from_template(this, template['DicomHeaderDict']))
    else:
        raise ValueError('Template is not Header or dict.')


def __get_tags_and_slices(obj):
    slices = tags = 1
    try:
        if issubclass(type(obj), Header):
            axes = obj.axes
        elif issubclass(type(obj), dict):
            axes = obj['axes']
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
    defaultHeader = template[0][0][2]
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
                templateHeader = defaultHeader
            DicomHeaderDict[_slice].append((template_tag, None, templateHeader))
    return DicomHeaderDict


def add_geometry(this, template):
    """Add geometry data to this header.

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
    if issubclass(type(template), Header):
        for attr in template.__dict__:
            if attr in geometry_tags:
                setattr(this, attr, getattr(template, attr, None))
    elif issubclass(type(template), dict):
        for attr in template:
            if attr in geometry_tags:
                if issubclass(type(this), Header):
                    setattr(this, attr, copy.copy(template[attr]))
                elif issubclass(type(this), dict):
                    this[attr] = copy.copy(template[attr])
    else:
        raise ValueError('Template is not Header or dict.')
