"""Image series
"""

import copy
import logging
import pydicom.dataset
import pydicom.datadict
import imagedata.formats
import imagedata.formats.dicomlib.uid
#from imagedata.series import Series

logging.getLogger(__name__).addHandler(logging.NullHandler())

header_tags = ['input_format',
               'DicomHeaderDict',
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
    """

    def __init__(self):
        """Initialize image header attributes to defaults
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

    def new_uid(self) -> str:
        return self.__uid_generator.__next__()

    def set_default_values(self, shape, axes):
        self.color = False
        if self.DicomHeaderDict is not None:
            return
        #self.axes = list()
        #for d in range(len(shape)):
        #    self.axes.append(imagedata.axis.UniformAxis('%d' % d,
        #                                                0,
        #                                                shape[d]))
        self.axes = copy.copy(axes)
        #logging.debug('Header.set_default_values: study  UID {}'.format(self.studyInstanceUID))
        #logging.debug('Header.set_default_values: series UID {}'.format(self.seriesInstanceUID))

        slices = tags = 1
        if axes is not None:
            for axis in axes:
                if axis.name == 'slice':
                    slices = len(axis)
                elif axis.name not in {'row', 'column', 'rgb'}:
                    tags = len(axis)

            self.DicomHeaderDict = {}
            i = 0
            #logging.debug('Header.set_default_values %d tags' % tags)
            #logging.debug('Header.set_default_values tags {}'.format(self.tags))
            for _slice in range(slices):
                self.DicomHeaderDict[_slice] = []
                for tag in range(tags):
                        self.DicomHeaderDict[_slice].append(
                            ( tag, None, self._empty_ds(i))
                            )
                        i += 1
            if self.tags is None:
                self.tags = {}
                for _slice in range(slices):
                    self.tags[_slice] = [i for i in range(tags)]

    def _empty_ds(self, i):
        SOPInsUID = self.new_uid()

        ds = pydicom.dataset.Dataset()

        # Add the data elements
        ds.StudyInstanceUID = self.studyInstanceUID
        ds.StudyID = '1'
        ds.SeriesInstanceUID = self.seriesInstanceUID
        ds.SOPClassUID = '1.2.840.10008.5.1.4.1.1.7' # SC
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

    Input:
    - this: header or dict
    - template: template header or dict. Can be None.
    """

    if template is None:
        return
    if not issubclass(type(this), Header) and not issubclass(type(this), dict):
        raise ValueError('Object is not Header or dict.')
    if issubclass(type(template), Header):
        for attr in template.__dict__:
            if attr in header_tags:
                setattr(this, attr, getattr(template, attr, None))
    elif issubclass(type(template), dict):
        for attr in template:
            if attr in header_tags:
                if issubclass(type(this), Header):
                    setattr(this, attr, copy.copy(template[attr]))
                elif issubclass(type(this), dict):
                    this[attr] = copy.copy(template[attr])
    else:
        raise ValueError('Template is not Header or dict.')

def add_geometry(this, template):
    """Add geometry data to this header.

    Input:
    - this: header or dict
    - template: template header or dict. Can be None.
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
