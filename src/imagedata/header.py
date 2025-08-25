"""Image series header

"""

import copy
import numpy as np
from collections import namedtuple
import pydicom.uid
import pydicom.dataset
import pydicom.datadict
from .formats import INPUT_ORDER_NONE, SORT_ON_SLICE, get_uid


header_tags = ['input_format',
               'modality', 'laterality', 'protocolName', 'bodyPartExamined',
               'seriesDate', 'seriesTime', 'seriesNumber',
               'seriesDescription', 'imageType', 'frameOfReferenceUID',
               'studyInstanceUID', 'studyID', 'seriesInstanceUID',
               'SOPClassUID', 'SOPInstanceUIDs',
               'accessionNumber',
               'patientName', 'patientID', 'patientBirthDate',
               # 'windowCenter', 'windowWidth',
               'dicomTemplate', 'dicomToDo',
               # 'tags',
               'colormap', 'colormap_norm', 'colormap_label', 'color',
               'echoNumbers', 'acquisitionNumber',
               'datasets',
               'input_sort']
geometry_tags = ['spacing', 'imagePositions', 'orientation', 'transformationMatrix',
                 'sliceLocations',
                 'patientPosition',
                 'tags',
                 'photometricInterpretation', 'axes']


class Header(object):
    """Image header object.

    Attributes:
        input_order
        sort_on
        input_format
        dicomTemplate
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
    """

    axes: namedtuple = None

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
        self.input_sort = SORT_ON_SLICE
        self.__uid_generator = get_uid()
        self.studyInstanceUID = self.new_uid()
        self.seriesInstanceUID = self.new_uid()
        self.frameOfReferenceUID = self.new_uid()
        self.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7'  # Secondary Capture Image Storage
        self.dicomToDo = []
        self.windowCenter = None
        self.windowWidth = None

    def __repr__(self):
        return object.__repr__(self)

    def __str__(self) -> str:
        items = []
        for attr in header_tags + geometry_tags:
            items.append("{0!r}: {1!r}".format(attr, getattr(self, attr, "")))
        return "{" + ", ".join(items) + "}"

    def __copy__(self):
        obj = Header()
        obj.set_default_values(self.axes)
        obj.add_template(self)
        obj.add_geometry(self)
        obj.input_order = self.input_order
        obj.input_format = self.input_format
        obj.windowCenter = None
        obj.windowWidth = None
        return obj

    @property
    def shape(self) -> tuple:
        """Return matrix shape as given by axes properties.
        """
        _shape = tuple()
        for _ in self.axes:
            _shape += (len(_),)
        return _shape

    def new_uid(self) -> str:
        """Return the next available UID from the UID generator.
        """
        return self.__uid_generator.__next__()

    def set_default_values(self, axes: namedtuple) -> None:
        """Set default values.
        """
        self.axes = None

        self.spacing = np.array([1, 1, 1])
        self.orientation = np.array([0, 0, 1, 0, 1, 0], dtype=np.float32)
        self.dicomTemplate = None
        self.imagePositions = {}
        self.windowCenter = None
        self.windowWidth = None
        self.color = False

        if axes is None:
            return

        try:
            slices = len(axes.slice)
        except AttributeError:
            slices = 1
        tags = (1,)
        axis_tags = tuple()
        for axis in axes:
            if axis.name not in ('slice', 'row', 'column'):
                axis_tags += (len(axis),)
        if len(axis_tags):
            tags = axis_tags

        # Construct new axes, copy to avoid crosstalk to template axes
        new_axes = namedtuple('Axes', axes._fields)
        self.axes = new_axes._make(axes)
        if self.axes[0].name[:7] == 'unknown':
            new_keys = [self.input_order] + list(self.axes._fields[1:])
            values = list(self.axes)
            values[0].name = self.input_order
            new_axes = namedtuple('Axes', new_keys)
            self.axes = new_axes._make(values)

        for _slice in range(slices):
            self.imagePositions[_slice] = np.array([_slice, 0, 0])
        if self.tags is None:
            self.tags = {}
            for _slice in range(slices):
                _tags = np.empty(tags, dtype=tuple)
                for tag in np.ndindex(tags):
                    _tags[tag] = tag
                self.tags[_slice] = _tags

    # noinspection PyPep8Naming
    def empty_ds(self) -> pydicom.dataset.Dataset:
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

    def add_template(self, template) -> None:
        """Add template data to this header.
        Does not add geometry data.

        Args:
            template: template header. Can be None.
        """

        if template is None:
            return
        for attr in template.__dict__:
            if attr in header_tags and attr not in ['seriesInstanceUID', 'tags', 'input_format']:
                value = getattr(template, attr, None)
                if value is not None:
                    setattr(self, attr, value)
        if 'keep_uid' in template.__dict__:
            value = getattr(template, 'seriesInstanceUID', None)
            if value is not None:
                setattr(self, 'seriesInstanceUID', value)

        # Make sure tags are set last. Template may be None
        # self.__set_tags_from_template(template)

    def get_tags_and_slices(self) -> tuple[tuple[int], int]:
        tags = tuple()
        slices = 1
        if self.axes is not None:
            for axis in self.axes:
                if axis.name == 'slice':
                    slices = len(axis)
                elif axis.name not in {'row', 'column'}:
                    tags += (len(axis),)
        return tags, slices

    def __set_tags_from_template(self, template) -> None:
        """Set tags from template tags, alternatively from template axes.

        Args:
            template (Header): template header
        Returns:
            self.tags
        Raises:
            ValueError: when no tag axis is found
        """
        def tuple_to_rectangle(shape: tuple[int]) -> tuple[slice]:
            rectangle = tuple()
            for s in shape:
                rectangle += (slice(0, s),)
            return rectangle

        self.tags = {}
        _last_tags = np.array([])
        tags, slices = self.get_tags_and_slices()
        new_tag_list = self.__construct_tags_from_template_axes(template)
        for _slice in range(slices):
            _tags = []
            try:
                template_tags = template.tags[_slice]
                if template_tags.ndim == 0:
                    template_tags = _last_tags
            except (TypeError, KeyError):
                template_tags = _last_tags
            # Use original template tags when possible, otherwise calculated tags
            if template_tags.shape >= tags:
                _tags = template_tags[tuple_to_rectangle(tags)]
            else:
                _tags = new_tag_list
            if _tags.ndim > 1:
                self.tags[_slice] = _tags.squeeze()
            else:
                self.tags[_slice] = _tags
            _last_tags = self.tags[_slice].copy()

    def __construct_tags_from_template_axes(self, template) -> np.ndarray:
        """Construct tag_list from self and template.
        Extend tag_list when template has to few tags.

        Args:
            self.input_order
            template (Header): template header
        Returns:
            tag_list (np.ndarray): calculated tags
        Raises:
            ValueError: when no tag axis is found
        """
        def previous_tag(tag: tuple[int], axis: int) -> tuple[int]:
            if tag is None or tag[axis] < 1:
                return None
            pre_tag = tuple()
            for i, t in enumerate(tag):
                if i == axis:
                    pre_tag += (tag[i]-1,)
                else:
                    pre_tag += (tag[i],)
            return pre_tag

        def tag_increment(tag: tuple[int], tag_list: np.ndarray[tuple[int]]) \
                -> tuple[float]:
            new_tag = tuple()
            for i, t in enumerate(tag):
                try:
                    pre1 = previous_tag(tag, i)
                    pre2 = previous_tag(pre1, i)
                    diff = tag_list[pre1][i] - tag_list[pre2][i]
                except IndexError:
                    diff = 1.0
                new_tag += (tag_list[pre1][i] + diff,)
            return new_tag

        if template.tags is None:
            return np.ndarray([])

        if self.input_order == 'none':
            # There will be one tag only per slice
            try:
                if issubclass(type(template.tags[0]), np.ndarray):
                    tag_list = template.tags[0].copy()
                else:
                    tag_list = np.array(template.tags[0])
                assert isinstance(tag_list, np.ndarray), \
                    "__construct_tags_from_axis not np.ndarray (is {})".format(type(
                        tag_list
                    ))
            except KeyError:
                tag_list = np.zeros((1,))
            return tag_list

        # Multiple tags
        tags, slices = self.get_tags_and_slices()
        new_tags = np.empty(tags, dtype=tuple)
        for tag in np.ndindex(tags):
            try:
                new_tags[tag] = tuple()
                for _ in range(len(tags)):
                    new_tags[tag] += (template.axes[_].values[tag[_]],)
            except IndexError:
                new_tags[tag] = tag_increment(tag, new_tags)
        return new_tags

    def add_geometry(self, geometry):
        """Add geometry data to obj header.

        Args:
            self: header or dict
            geometry: geometry template header or dict. Can be None.
        """

        if geometry is None:
            return
        for attr in geometry.__dict__:
            if attr not in ['tags', 'axes', 'input_format', 'sliceLocations']:
                if attr in geometry_tags:
                    value = getattr(geometry, attr, None)
                    if value is not None:
                        setattr(self, attr, value)
        # Make sure axes are set last. Geometry may be None
        self.__set_axes_from_template(
            getattr(geometry, 'axes', None)
        )
        self.__set_slice_locations_from_template(
            getattr(geometry, 'sliceLocations', None)
        )
        self.__set_tags_from_template(geometry)

    def __set_axes_from_template(self, geometry_axes: namedtuple):
        if geometry_axes is None:
            return
        _axes = []
        _axis_names = []
        for axis in self.axes:
            try:
                geometry_axis = getattr(geometry_axes, axis.name)
                # Ensure geometry_axis length agree with matrix size
                _axes.append(geometry_axis.copy(axis.name, n=len(axis)))
            except AttributeError:
                if axis.name == 'none':
                    try:
                        geometry_axis = getattr(geometry_axes, self.input_order)
                        _axes.append(geometry_axis.copy(self.input_order, n=len(axis)))
                    except AttributeError:
                        _axes.append(axis.copy(axis.name, n=len(axis)))
                else:
                    _axes.append(axis.copy(axis.name, n=len(axis)))
            _axis_names.append(axis.name)
        Axes = namedtuple('Axes', _axis_names)
        self.axes = Axes._make(_axes)

    def __set_slice_locations_from_template(self, geometry_sloc):
        if geometry_sloc is None:
            return
        if issubclass(type(geometry_sloc), list):
            sloc = geometry_sloc
        else:
            sloc = geometry_sloc.tolist()
        try:
            ds = 1
            if len(sloc) > 1:
                ds = sloc[1] - sloc[0]  # Distance in slice location
            while len(sloc) < len(self.axes.slice):
                sloc.append(sloc[-1] + ds)
            sloc = sloc[:len(self.axes.slice)]
            self.sliceLocations = np.array(sloc)
        except (AttributeError, ValueError):
            self.sliceLocations = geometry_sloc
