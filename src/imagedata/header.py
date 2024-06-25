"""Image series header

"""

import copy
import numpy as np
import pydicom.uid
import pydicom.dataset
import pydicom.datadict
from .formats import INPUT_ORDER_NONE, SORT_ON_SLICE
from .formats.dicomlib.uid import get_uid


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
               'tags', 'colormap', 'colormap_norm', 'colormap_label', 'color',
               'echoNumbers', 'acquisitionNumber',
               'input_sort']
geometry_tags = ['spacing', 'imagePositions', 'orientation', 'transformationMatrix',
                 'sliceLocations',
                 'patientPosition',
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
        shape = []
        for axis in self.axes:
            shape.append(len(axis))
        return tuple(shape)

    def new_uid(self) -> str:
        """Return the next available UID from the UID generator.
        """
        return self.__uid_generator.__next__()

    def set_default_values(self, axes) -> None:
        """Set default values.
        """
        self.axes = []

        self.spacing = np.array([1, 1, 1])
        self.orientation = np.array([0, 0, 1, 0, 1, 0], dtype=np.float32)
        self.dicomTemplate = None
        self.imagePositions = {}
        self.windowCenter = None
        self.windowWidth = None
        self.color = False

        if axes is None:
            return

        slices = tags = 1
        # Construct new axes, copy to avoid crosstalk to template axes
        for _axis in axes:
            axis = copy.copy(_axis)
            if axis.name == 'slice':
                slices = len(axis)
            elif axis.name not in {'row', 'column'}:
                tags = len(axis)
                if axis.name == 'unknown':
                    axis.name = self.input_order
            self.axes.append(axis)

        for _slice in range(slices):
            self.imagePositions[_slice] = np.array([_slice, 0, 0])
        if self.tags is None:
            self.tags = {}
            for _slice in range(slices):
                self.tags[_slice] = np.arange(tags)

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
        self.__set_tags_from_template(template)

    def __get_tags_and_slices(self):  # -> tuple[int, int]:
        slices = tags = 1
        if self.axes is not None:
            for axis in self.axes:
                if axis.name == 'slice':
                    slices = len(axis)
                elif axis.name not in {'row', 'column'}:
                    tags = len(axis)
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
        self.tags = {}
        _last_tags = []
        tags, slices = self.__get_tags_and_slices()
        new_tag_list = self.__construct_tags_from_axis(template)
        for _slice in range(slices):
            _tags = []
            try:
                if issubclass(type(template.tags[_slice]), dict):
                    template_tag_list = list(template.tags[_slice].values())
                else:
                    template_tag_list = list(template.tags[_slice])
            except (TypeError, KeyError):
                template_tag_list = _last_tags
            # Use original template tags when possible, otherwise calculated tags
            if len(template_tag_list) >= tags:
                _tags = template_tag_list[:tags]
            else:
                _tags = new_tag_list
            self.tags[_slice] = np.array(_tags)
            _last_tags = self.tags[_slice].copy()

    def __construct_tags_from_axis(self, template) -> np.ndarray:
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
        def tag_increment(tag_list):
            if len(tag_list) < 2:
                return 1.0
            else:
                return tag_list[-1] - tag_list[-2]  # Difference of last to tags

        if template.tags is None:
            return np.ndarray([])

        tags, slices = self.__get_tags_and_slices()
        tag_list = {}

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
        # new_tags = [template.axes[0][_].values[0] for _ in range(len(template.axes[0]))]
        new_tags = [_ for _ in template.axes[0]]
        while len(new_tags) < tags:
            new_tags.append(new_tags[-1] + tag_increment(new_tags))
        return np.array(new_tags)

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

    def __set_axes_from_template(self, geometry_axes):
        if self.axes is None:
            ndim = 1
            if geometry_axes is not None:
                ndim = len(geometry_axes)
            self.axes = [False for _ in range(ndim)]
        for i, axis in enumerate(self.axes):
            if geometry_axes is not None:
                for geometry_axis in geometry_axes:
                    if geometry_axis.name == axis.name:
                        # Ensure geometry_axis length agree with matrix size
                        self.axes[i] = geometry_axis.copy(axis.name, n=len(axis))

    def __set_slice_locations_from_template(self, geometry_sloc):
        if geometry_sloc is None:
            return
        if issubclass(type(geometry_sloc), list):
            sloc = geometry_sloc
        else:
            sloc = geometry_sloc.tolist()
        try:
            slice_axis = self.find_axis('slice')
            ds = 1
            if len(sloc) > 1:
                ds = sloc[1] - sloc[0]  # Distance in slice location
            while len(sloc) < len(slice_axis):
                sloc.append(sloc[-1] + ds)
            sloc = sloc[:len(slice_axis)]
            self.sliceLocations = np.array(sloc)
        except ValueError:
            self.sliceLocations = geometry_sloc

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
