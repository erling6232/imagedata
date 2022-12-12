"""Image series header

"""

import copy
import logging
import numpy as np
import pydicom.dataset
import pydicom.datadict
from .formats import INPUT_ORDER_NONE
from .formats.dicomlib.uid import get_uid
from .axis import UniformAxis, UniformLengthAxis, VariableAxis

logger = logging.getLogger(__name__)

header_tags = ['input_format',
               'seriesNumber',
               'seriesDescription', 'imageType', 'frameOfReferenceUID',
               'studyInstanceUID', 'studyID', 'seriesInstanceUID',
               'SOPClassUID',
               'accessionNumber',
               'patientName', 'patientID', 'patientBirthDate',
               'input_sort']
geometry_tags = ['tags', 'spacing',
                 'imagePositions', 'orientation', 'transformationMatrix',
                 'photometricInterpretation', 'axes']


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
        # sliceLocations
        tags
        spacing
        imagePositions
        orientation
        transformationMatrix
        photometricInterpretation
        axes
        __uid_generator
        studyInstanceUID
        seriesInstanceUID
        frameOfReferenceUID
        DicomHeaderDict
        tags
    """

    def __init__(self):
        """Initialize image header attributes to defaults

        This is created in Series.__array_finalize__() and Series.__new__()
        """
        object.__init__(self)
        self.input_order = INPUT_ORDER_NONE
        self.sort_on = None
        for attr in header_tags + geometry_tags:
            try:
                setattr(self, attr, None)
            except AttributeError:
                pass
        self.__uid_generator = get_uid()
        self.studyInstanceUID = self.new_uid()
        self.seriesInstanceUID = self.new_uid()
        self.frameOfReferenceUID = self.new_uid()
        self.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
        # pydicom.datadict.DicomDictionary
        # from pydicom.uid import UID
        # UID.
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
        if self.DicomHeaderDict is not None:
            return
        # self.axes = copy.copy(axes)
        self.axes = []

        self.spacing = np.array([1, 1, 1])
        self.orientation = np.array([0, 0, 1, 0, 1, 0], dtype=np.float32)
        self.DicomHeaderDict = {}
        self.imagePositions = {}

        if axes is None:
            return

        slices = tags = 1
        # Construct new axes, copy to avoid crosstalk to template axes
        for _axis in axes:
            axis = copy.copy(_axis)
            if axis.name == 'slice':
                slices = len(axis)
            elif axis.name not in {'row', 'column', 'rgb'}:
                tags = len(axis)
                if axis.name == 'unknown':
                    axis.name = self.input_order
            self.axes.append(axis)

        for _slice in range(slices):
            self.imagePositions[_slice] = np.array([_slice, 0, 0])
            self.DicomHeaderDict[_slice] = []
            for tag in range(tags):
                self.DicomHeaderDict[_slice].append(
                    (tag, None, self.empty_ds())
                )
        if self.tags is None:
            self.tags = {}
            for _slice in range(slices):
                self.tags[_slice] = np.arange(tags)

    # noinspection PyPep8Naming
    def empty_ds(self):
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

    def add_template(self, template):
        """Add template data to this header.
        Does not add geometry data.

        Args:
            template: template header. Can be None.
        """

        if template is None:
            return
        # for attr in attributes(template):
        for attr in template.__dict__:
            if attr in header_tags and attr not in ['seriesInstanceUID', 'input_format']:
                value = getattr(template, attr, None)
                if value is not None:
                    setattr(self, attr, getattr(template, attr, None))
        # Make sure DicomHeaderDict is set last
        if template.DicomHeaderDict is not None:
            self.DicomHeaderDict = self.__make_DicomHeaderDict_from_template(
                template.DicomHeaderDict)

    def __get_tags_and_slices(self):
        slices = tags = 1
        if self.axes is not None:
            for axis in self.axes:
                if axis.name == 'slice':
                    slices = len(axis)
                elif axis.name not in {'row', 'column', 'rgb'}:
                    tags = len(axis)
        return tags, slices

    def __make_DicomHeaderDict_from_template(self, template):
        """Shallow copy of template Dataset.
        When modifying attributes with Series.setDicomAttribute,
        new attribute will be set to there avoid cross-talk.
        """

        DicomHeaderDict = {}
        default_header = template[0][0][2]
        tags, slices = self.__get_tags_and_slices()
        for _slice in range(slices):
            DicomHeaderDict[_slice] = []
            for tag in range(tags):
                try:
                    template_tag = template[_slice][tag][0]
                except (KeyError, IndexError):
                    template_tag = tag
                try:
                    templateHeader = copy.copy(template[_slice][tag][2])
                except KeyError:
                    templateHeader = copy.copy(default_header)
                DicomHeaderDict[_slice].append((template_tag, None, templateHeader))
        return DicomHeaderDict

    # def __copy_DicomHeaderDict(self, source, filename=None):
    #     sop_ins_uid = self.new_uid()
    #
    #     # Populate required values for file meta information
    #     file_meta = pydicom.dataset.FileMetaDataset()
    #     # file_meta.MediaStorageSOPClassUID = template.SOPClassUID
    #     file_meta.MediaStorageSOPInstanceUID = sop_ins_uid
    #     # file_meta.ImplementationClassUID = "%s.1" % obj.root
    #     file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian
    #
    #     ds = pydicom.dataset.FileDataset(
    #         filename,
    #         {},
    #         file_meta=file_meta,
    #         preamble=b"\0" * 128
    #     )
    #
    #     for element in source.iterall():
    #         if element.tag == 0x7fe00010:
    #             continue  # Do not copy pixel data, will be added later
    #         # ds.add(copy.copy(element))
    #         ds.add(element)
    #     ds.SOPInstanceUID = sop_ins_uid
    #
    #     return ds

    def __set_tags_from_template(self, template, geometry):
        self.tags = {}
        _last_tags = None
        tags, slices = self.__get_tags_and_slices()
        geometry_tag_list = self.__construct_geometry_tag_list(geometry)
        for _slice in range(slices):
            _tags = []
            template_tag_list = []
            if template is not None:
                try:
                    if issubclass(type(template[_slice]), dict):
                        template_tag_list = list(template[_slice].values())
                    else:
                        template_tag_list = list(template[_slice])
                except KeyError:
                    # Re-use last template_tag_list for this _slice
                    pass
            if len(geometry_tag_list[_slice]) >= tags:
                _tags = geometry_tag_list[_slice][:tags]
                assert isinstance(_tags, np.ndarray),\
                    "__set_tags_from_template not np.ndarray ({})".format(type(_tags))
            elif len(template_tag_list) >= tags:
                _tags = template_tag_list[:tags]
                assert isinstance(_tags, np.ndarray), \
                    "__set_tags_from_template not np.ndarray ({})".format(type(_tags))
            else:
                _tags = _last_tags
                # raise IndexError('Cannot get tag list with length {}'.format(tags))
            self.tags[_slice] = np.array(_tags)
            _last_tags = self.tags[_slice].copy()

    def __construct_geometry_tag_list(self, geometry):
        """Construct tag_list from self and geometry.
        Extend tag_list when geometry has to few tags.

        Args:
            self.input_order
            geometry[slice]: dict of np.ndarray
        Returns:
            tag_list[slice]: dict of np.ndarray
        Raises:
            ValueError: when no tag axis is found
        """
        def tag_increment(tag_list):
            if len(tag_list) < 2:
                return 1.0
            else:
                return tag_list[-1] - tag_list[-2]  # Difference of last to tags

        tags, slices = self.__get_tags_and_slices()
        tag_list = {}

        if self.input_order == 'none':
            # There will be one tag only per slice
            for _slice in range(slices):
                try:
                    if issubclass(type(geometry[_slice]), np.ndarray):
                        tag_list[_slice] = geometry[_slice].copy()
                    else:
                        tag_list[_slice] = np.array(geometry[_slice])
                    assert isinstance(tag_list[_slice], np.ndarray),\
                        "__construct_geometry_tag_list not np.ndarray (is {})".format(type(
                            tag_list[_slice]
                        ))
                except KeyError:
                    tag_list[_slice] = np.zeros((1,))
            return tag_list

        # Possibly multiple tags per slice
        for _slice in range(slices):
            try:
                _list = list(geometry[_slice])
                assert issubclass(type(geometry[_slice]), np.ndarray),\
                    "geometry[] should be np.ndarray (is: {})".format(type(_list))
            except KeyError:
                _list = [0.0]
            except AssertionError:
                raise
            while len(_list) < tags:
                _list.append(_list[-1] + tag_increment(_list))  # Append increasing tag
            tag_list[_slice] = np.array(_list)
        return tag_list

    def __set_axes_from_template(self, template_axes, geometry_axes):
        if self.axes is None:
            ndim = 1
            if geometry_axes is not None:
                ndim = len(geometry_axes)
            elif template_axes is not None:
                ndim = len(template_axes)
            self.axes = [False for _ in range(ndim)]
        for i, axis in enumerate(self.axes):
            if geometry_axes is not None:
                for geometry_axis in geometry_axes:
                    if geometry_axis.name == axis.name:
                        # Ensure geometry_axis length agree with matrix size
                        self.axes[i] = self.__adjust_axis_from_template(axis, geometry_axis)
            elif template_axes is not None:
                for template_axis in template_axes:
                    if template_axis.name == axis.name:
                        # Ensure template_axis length agree with matrix size
                        self.axes[i] = self.__adjust_axis_from_template(axis, template_axis)

    def __adjust_axis_from_template(self, axis, template):
        """Construct new axis from template, retaining axis length.
        """
        # UniformLengthAxis is subclassed from UniformAxis, so check first
        if isinstance(template, UniformLengthAxis):
            return UniformLengthAxis(axis.name,
                                     template.start,
                                     len(axis),
                                     template.step)
        elif isinstance(template, UniformAxis):
            return UniformAxis(axis.name,
                               template.start,
                               template.start + (len(axis)+1)*template.step,  # stop
                               template.step)
        elif isinstance(template, VariableAxis):
            return VariableAxis(axis.name,
                                template.values[:len(axis)])
        else:
            raise ValueError('Unknown template axis class: {}'.format(
                type(template)))

    def add_geometry(self, template, geometry):
        """Add geometry data to obj header.

        Args:
            self: header or dict
            template: template header or dict. Can be None.
            geometry: geometry template header or dict. Can be None.
        """

        if geometry is None:
            return
        for attr in geometry.__dict__:
            if attr in geometry_tags and attr not in ['tags', 'axes', 'input_format']:
                value = getattr(geometry, attr, None)
                if value is not None:
                    setattr(self, attr, value)
        # Make sure tags and axes are set last. Template and/or geometry may be None
        self.__set_tags_from_template(
            getattr(template, 'tags', None),
            getattr(geometry, 'tags', None)
        )
        self.__set_axes_from_template(
            getattr(template, 'axes', None),
            getattr(geometry, 'axes', None)
        )

    def find_axis(self, name):
        """Find axis with given name

        Args:
            name: Axis name to search for

        Returns:
            axis object with given name

        Raises:
            ValueError: when no axis object has given name

        Usage:
            >>> from imagedata.series import Series
            >>> si = Series(np.array([3, 3, 3]))
            >>> axis = si.find_axis('slice')
        """
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise ValueError("No axis object with name %s exist" % name)


def deepcopy_DicomHeaderDict(source, filename=None):
    """Deepcopy contents of DicomHeaderDict."""

    if isinstance(source, dict):
        ds = {}
        for tag, element in source.items():
            if tag == 0x7fe00010:
                continue  # Do not copy pixel data, will be added later
            ds[tag] = copy.deepcopy(element)
            # ds.add(element)
    else:
        # sop_ins_uid = obj.new_uid()

        # Populate required values for file meta information
        file_meta = pydicom.dataset.FileMetaDataset()
        # file_meta.MediaStorageSOPClassUID = template.SOPClassUID
        # file_meta.MediaStorageSOPInstanceUID = sop_ins_uid
        # file_meta.ImplementationClassUID = "%s.1" % obj.root
        file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

        ds = pydicom.dataset.FileDataset(
            filename,
            {},
            file_meta=file_meta,
            preamble=b"\0" * 128
        )

        for element in source.iterall():
            # print('deepcopy_DicomHeaderDict: element {} {}'.format(
            #    element.tag,
            #    get_size(element)))
            if element.tag == 0x7fe00010:
                continue  # Do not copy pixel data, will be added later
            ds.add(copy.deepcopy(element))
            # ds.add(element)

    # print('deepcopy_DicomHeaderDict: {} -> {}'.format(
    #    get_size(source),
    #    get_size(ds)
    # ))
    return ds
