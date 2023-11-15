"""Image cohort

The Cohort class is a collection of Patient objects.

  Typical example usage:

  cohort = Cohort('input')

"""

from datetime import datetime, date, time
from sortedcontainers import SortedDict
import logging
import argparse
from pathlib import Path

from .series import Series
from .readdata import read as r_read

logger = logging.getLogger(__name__)


def _get_attribute(_data, _attr):
    # Get attribute from first instance in _data
    if len(_data) < 1:
        raise IndexError('No instance in present {}.'.format(type(_data)))
    _instance = list(_data.values())[0]
    if issubclass(type(_instance), Series):
        return _instance.getDicomAttribute(_attr)
    # Recurse to lower level, until finding a Series instance
    return _get_attribute(_instance, _attr)


def _sort_in_series(_data, _opts):
    """Sort series into a dict of Series
    Called by Cohort for url data
    Called by Patient, Study ?

    Args:
        _data: url (str) or dict of Series instances
        _opts
    Returns:
        dict of Series instances, key is SeriesInstanceUID

    """
    _series_dict = {}
    if issubclass(type(_data), list):
        if len(_data):
            if issubclass(type(_data[0]), Series):
                # _data is list of Series
                for _series in _data:
                    _series_dict[_series.seriesInstanceUID] = _series
            else:
                raise ValueError(
                    'Unexpected series data type {}'.format(
                        type(_data[0])))
    elif issubclass(type(_data), dict):
        if len(_data):
            if issubclass(type(list(_data.values())[0]), Series):
                # _data is list of Series
                for _seriesInstanceUID in _data:
                    _series_dict[_seriesInstanceUID] = _data[_seriesInstanceUID]
            else:
                raise ValueError(
                    'Unexpected series data type {}'.format(
                        type(list(_data.values())[0])))
    elif issubclass(type(_data), Study):
        raise ValueError('Why here')
        # return {_data.seriesInstanceUID: _data}
    elif issubclass(type(_data), str) or issubclass(type(_data), Path):
        # _data is URL
        # Read input, hdr is dict of attributes
        _opts['separate_series'] = True
        _hdr, _si = r_read(_data, order='auto', opts=_opts)

        for _uid in _hdr:
            _series = Series(_si[_uid])
            _series.header = _hdr[_uid]
            _series_dict[_series.seriesInstanceUID] = _series
    else:
        raise ValueError('Unexpected series type {}'.format(type(_data)))
    return _series_dict


def _sort_in_studies(_series_dict, _opts):
    """Sort series in studies
    Called by Cohort for series dict

    Args:
        _series_dict: dict of Series instances
        _opts:

    Returns:
        dict of Study instances, key is StudyInstanceUID
    """
    _studies = {}
    for _seriesInstanceUID in _series_dict:
        _series = _series_dict[_seriesInstanceUID]
        _studyInstanceUID = _series.studyInstanceUID
        if _studyInstanceUID not in _studies:
            _studies[_studyInstanceUID] = {}
        _studies[_studyInstanceUID][_seriesInstanceUID] = _series
    # Make found _studies into Study instances
    _study_dict = {}
    for _studyInstanceUID in _studies:
        _study_dict[_studyInstanceUID] = Study(_studies[_studyInstanceUID], opts=_opts)
    return _study_dict
    # if issubclass(type(_data), dict):
    #     # _data is list of Study
    #     for _seriesInstanceUID in _data:
    #         _series = _data[_seriesInstanceUID]
    #         if _series.studyInstanceUID not in _studies:
    #             _studies[_series.studyInstanceUID] = {}
    #         _studies[_series.studyInstanceUID][_seriesInstanceUID] = _series
    #     return _studies
    # raise ValueError('Why here?')


def _sort_in_patients(_study_dict, _opts):
    """Sort studies in patients
    Called by Cohort for study dict

    Args:
        _study_dict: dict of Study instances
        _opts:

    Returns:
        dict of Patient instances
    """
    _patients = {}
    for _studyInstanceUID in _study_dict:
        _study = _study_dict[_studyInstanceUID]
        _series = list(_study.values())[0]
        _patientID = _series.patientID
        if _patientID not in _patients:
            _patients[_patientID] = {}
        _study = Study(_study_dict[_studyInstanceUID])
        _patients[_patientID][_studyInstanceUID] = _study
    # Make found _patients into Patient instances
    _patient_dict = {}
    for _patientID in _patients:
        # _patient_dict[_patientID] = _patients[_patientID]
        _patient_dict[_patientID] = Patient(_patients[_patientID], opts=_opts)
    return _patient_dict
    # return _patients


class UnknownOptionType(Exception):
    pass


class GeneralEquipment(object):

    _equipment_attributes = [
        'manufacturer', 'manufacturersModelName', 'stationName', 'deviceSerialNumber',
        'softwareVersions'
    ]

    def __init__(self, obj):
        super(GeneralEquipment, self).__init__()
        for _attr in self._equipment_attributes:
            _dicom_attribute = _attr[0].upper() + _attr[1:]
            setattr(self, _attr, obj.getDicomAttribute(_dicom_attribute))


class ClinicalTrialSubject(object):
    _clinical_trial_attributes = [
        'sponsorName',
        'protocolID',
        'protocolName',
        'siteID',
        'siteName',
        'subjectID',
        'subjectReadingID',
        'protocolEthicsCommitteeName',
        'protocolEthicsCommitteeApprovalNumber'
    ]

    def __init__(self, obj):
        super(ClinicalTrialSubject, self).__init__()
        for _attr in self._clinical_trial_attributes:
            _dicom_attribute = 'ClinicalTrial' + _attr[0].upper() + _attr[1:]
            setattr(self, _attr, _get_attribute(obj, _dicom_attribute))


class Study(SortedDict):
    """Study -- Read and sort images into a collection of Series objects.

    Study(data, opts=None)

    Examples:

        >>> from imagedata import Study
        >>> study = Study('directory/')
        >>> for uid in study:
        >>>     series = study[uid]

    Args:
        data: URL to input data, or list of Series instances
        opts: Dict of input options, mostly for format specific plugins
            (argparse.Namespace or dict)
            * 'strict_values': Whether study attributes should match in each series (bool)

    Returns:
        Study instance

    """

    name = "Study"
    description = "Image study"
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

    _attributes = [
        'studyDate', 'studyTime', 'studyDescription', 'studyID', 'studyInstanceUID',
        'referringPhysiciansName'
    ]

    def __init__(self, data, opts=None):
        super(Study, self).__init__()
        for _attr in self._attributes:
            setattr(self, _attr, None)

        if opts is None:
            _in_opts = {}
        elif issubclass(type(opts), dict):
            _in_opts = opts
        elif issubclass(type(opts), argparse.Namespace):
            _in_opts = vars(opts)
        else:
            raise UnknownOptionType('Unknown opts type ({}): {}'.format(type(opts),
                                                                        opts))

        _strict_values = True if 'strict_values' not in _in_opts \
            else _in_opts['strict_values']

        if issubclass(type(data), dict):
            _series_dict = data
        elif issubclass(type(data), Study):
            raise ValueError("Why here?")
        else:
            # Assume data is URL
            try:
                _series_dict = _sort_in_series(data, _in_opts)
            except Exception as e:
                print(e)
                print('data: {}'.format(type(data)))
                raise

        for _seriesInstanceUID in _series_dict:
            _series = _series_dict[_seriesInstanceUID]
            self[_series.seriesInstanceUID] = _series
            for _attr in self._attributes:
                _dicom_attribute = _attr[0].upper() + _attr[1:]
                _value = self[_seriesInstanceUID].getDicomAttribute(_dicom_attribute)
                if _value is not None and _attr == 'studyDate':
                    try:
                        _value = datetime.strptime(_value, "%Y%m%d")
                    except ValueError:
                        _value = None
                elif _value is not None and _attr == 'studyTime':
                    try:
                        if '.' in _value:
                            _value = datetime.strptime(_value, "%H%M%S.%f").time()
                        else:
                            _value = datetime.strptime(_value, "%H%M%S").time()
                    except ValueError:
                        _value = datetime.time(_value)
                # Update self property if None from series
                if getattr(self, _attr, None) is None:
                    # _series = self[_seriesInstanceUID]
                    setattr(self, _attr, _value)
                elif _strict_values and \
                        getattr(self, _attr, None) != _value:
                    # Study attributes differ, should be considered an exception.
                    raise ValueError('Study attribute "{}" differ ("{}" vs. "{}")'.format(
                        _attr,
                        getattr(self, _attr, None),
                        _value
                    ))
            setattr(self, 'generalEquipment', GeneralEquipment(self[_seriesInstanceUID]))

    def __str__(self):
        _date = self.studyDate if self.studyDate is not None else date.min
        _time = self.studyTime if self.studyTime is not None else time.min
        _descr = self.studyDescription if self.studyDescription is not None else ''
        return \
            "Study: {} {}".format(datetime.combine(_date, _time), _descr)

    def write(self, url, opts=None, formats=None):
        """Write image data, calling appropriate format plugins

        Args:
            self: Study instance
            url: output destination url
            opts: Output options (argparse.Namespace or dict)
            formats: list of output formats, overriding opts.output_format (list or str)
        Raises:
            imagedata.collections.UnknownOptionType: When opts cannot be made into a dict.
            TypeError: List of output format is not list().
            ValueError: Wrong number of destinations given,
                or no way to write multidimensional image.
            imagedata.formats.WriteNotImplemented: Cannot write this image format.
        """

        _used_urls = []
        for _seriesInstanceUID in self.keys():
            _series = self[_seriesInstanceUID]
            try:
                series_str = '{}-{}'.format(_series.seriesNumber, _series.seriesDescription)
            except TypeError:
                series_str = _seriesInstanceUID
            _url = "{}/{}".format(
                url,
                series_str
            )
            while _url in _used_urls:
                _url = _url + "0"  # Make unique output file
            _used_urls.append(_url)
            try:
                _series.write(_url, opts=opts, formats=formats)
            except Exception as e:
                raise Exception(_url) from e


class Patient(SortedDict):
    """Patient -- Read and sort images into a collection of Study objects.

    Patient(data, opts=None)

    Examples:

        >>> from imagedata import Patient
        >>> patient = Patient('directory/')
        >>> for uid in patient:
        >>>     study = patient[uid]

    Args:
        data: URL to input data, or list of Study instances
        opts: Dict of input options, mostly for format specific plugins
            (argparse.Namespace or dict)
            * 'strict_values': Whether attributes should match in each study (bool)

    Returns:
        Patient instance

    """

    name = "Patient"
    description = "Image patient"
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

    _attributes = ['patientName', 'patientID', 'patientBirthDate', 'patientSex',
                   'patientAge', 'patientSize', 'patientWeight', 'qualityControlSubject',
                   'patientIdentityRemoved', 'deidentificationMethod'
                   ]

    def __init__(self, data, opts=None):

        super(Patient, self).__init__()
        for _attr in self._attributes:
            setattr(self, _attr, None)

        if opts is None:
            _in_opts = {}
        elif issubclass(type(opts), dict):
            _in_opts = opts
        elif issubclass(type(opts), argparse.Namespace):
            _in_opts = vars(opts)
        else:
            raise UnknownOptionType('Unknown opts type ({}): {}'.format(type(opts),
                                                                        opts))

        _strict_values = True if 'strict_values' not in _in_opts \
            else _in_opts['strict_values']

        if issubclass(type(data), list) and len(data) and issubclass(type(data[0]), Study):
            # Add each study to self patient
            self._add_studies(data)
        elif issubclass(type(data), dict) and issubclass(type(list(data.values())[0]), Study):
            self._add_studies(data)
            # _study_dict = data
            # for _studyInstanceUID in data:
            #     self[_studyInstanceUID] = _study_dict[_studyInstanceUID]
        else:
            _series_dict = _sort_in_series(data, _in_opts)
            _study_dict = _sort_in_studies(_series_dict, _in_opts)
            # Add each study to self patient
            for _studyInstanceUID in _study_dict:
                # self[_studyInstanceUID] = Study(_study_dict[_studyInstanceUID])
                self[_studyInstanceUID] = _study_dict[_studyInstanceUID]

        for _studyInstanceUID in self.keys():
            for _attr in self._attributes:
                _dicom_attribute = _attr[0].upper() + _attr[1:]
                _value = _get_attribute(self[_studyInstanceUID], _dicom_attribute)
                if _value is not None and (_attr == 'patientSize' or _attr == 'patientWeight'):
                    try:
                        _value = float(_value)
                    except ValueError:
                        _value = None
                # Update self property if None from study
                if getattr(self, _attr, None) is None:
                    setattr(self, _attr, _value)
                elif _strict_values and \
                        getattr(self, _attr, None) != _value:
                    # Patient attributes differ, should be considered an exception.
                    raise ValueError('Patient attribute "{}" differ ("{}" vs. "{}")'.format(
                        _attr,
                        getattr(self, _attr, None),
                        _value
                    ))
            setattr(self, 'clinicalTrialSubject', ClinicalTrialSubject(self[_studyInstanceUID]))

    def _add_studies(self, data):
        """Add each study to self patient

        Args:
            data: dict of Study instances

        Returns:
            self: added Study instances to self
        """

        if issubclass(type(data), dict):
            for _studyInstanceUID in data:
                if issubclass(type(data[_studyInstanceUID]), Study):
                    self[_studyInstanceUID] = data[_studyInstanceUID]
                else:
                    raise ValueError(
                        'Unexpected patient data type {}'.format(
                            type(data[_studyInstanceUID])
                        ))
        elif issubclass(type(data), list):
            for _study in data:
                if issubclass(type(_study), Study):
                    self[_study.studyInstanceUID] = _study
                else:
                    raise ValueError(
                        'Unexpected study data type {}'.format(
                            type(_study)
                        ))
        else:
            raise ValueError('Unexpected data type {} (expected: dict or list)'.format(type(data)))

    def __str__(self):
        try:
            _id = self.patientID
        except ValueError:
            _id = ''
        try:
            _patientName = self.patientName
        except ValueError:
            _patientName = ''
        return \
            "Patient: [{}] {}".format(_id, _patientName)

    def write(self, url, opts=None, formats=None):
        """Write image data, calling appropriate format plugins

        Args:
            self: Patient instance
            url: output destination url
            opts: Output options (argparse.Namespace or dict)
            formats: list of output formats, overriding opts.output_format (list or str)
        Raises:
            imagedata.collections.UnknownOptionType: When opts cannot be made into a dict.
            TypeError: List of output format is not list().
            ValueError: Wrong number of destinations given,
                or no way to write multidimensional image.
            imagedata.formats.WriteNotImplemented: Cannot write this image format.
        """

        _used_urls = []
        for _studyInstanceUID in self.keys():
            _study = self[_studyInstanceUID]
            try:
                study_str = datetime.combine(
                    _study.studyDate, _study.studyTime
                ).strftime('%Y%m%d-%H%M%S')
            except TypeError:
                study_str = _studyInstanceUID
            for _seriesInstanceUID in _study:
                _series = _study[_seriesInstanceUID]
                try:
                    series_str = '{}-{}'.format(_series.seriesNumber, _series.seriesDescription)
                except TypeError:
                    series_str = _seriesInstanceUID
                _url = "{}/{}/{}".format(
                    url,
                    study_str,
                    series_str
                )
                while _url in _used_urls:
                    _url = _url + "0"  # Make unique output file
                _used_urls.append(_url)
                try:
                    _series.write(_url, opts=opts, formats=formats)
                except Exception as e:
                    raise Exception(_url) from e


class Cohort(SortedDict):
    """Cohort -- Read and sort images into a collection of Patient objects.

    Cohort(data, opts=None)

    Examples:

        >>> from imagedata import Cohort
        >>> cohort = Cohort('directory/')
        >>> for id in cohort:
        >>>     patient = cohort[id]

    Args:
        data: URL to input data, or list of Patient instances
        opts: Dict of input options, mostly for format specific plugins
            (argparse.Namespace or dict)
            * 'strict_values': Whether attributes should match in each study (bool)

    Returns:
        Cohort instance

    """

    name = "Cohort"
    description = "Image cohort"
    authors = "Erling Andersen"
    version = "1.0.0"
    url = "www.helse-bergen.no"

    _attributes = []

    def __init__(self, data, opts=None):

        super(Cohort, self).__init__()
        self.data = data
        for _attr in self._attributes:
            setattr(self, _attr, None)

        if opts is None:
            _in_opts = {}
        elif issubclass(type(opts), dict):
            _in_opts = opts
        elif issubclass(type(opts), argparse.Namespace):
            _in_opts = vars(opts)
        else:
            raise UnknownOptionType('Unknown opts type ({}): {}'.format(type(opts),
                                                                        opts))

        _strict_values = True if 'strict_values' not in _in_opts \
            else _in_opts['strict_values']

        if issubclass(type(data), list) and len(data) and issubclass(type(data[0]), Patient):
            # Add each patient to self cohort
            self._add_patients(data, _in_opts)
        elif issubclass(type(data), str) or issubclass(type(data), Path):
            # Read input data from url
            _series_dict = _sort_in_series(data, _in_opts)
            _study_dict = _sort_in_studies(_series_dict, _in_opts)
            _patient_dict = _sort_in_patients(_study_dict, _in_opts)
            # Add each patient to self cohort
            for _patientID in _patient_dict:
                # self[_patientID] = Patient(_patient_dict[_patientID])
                self[_patientID] = _patient_dict[_patientID]
        else:
            raise ValueError('Unexpected cohort data type {}'.format(type(data)))

        for _patientID in self.keys():
            for _attr in self._attributes:
                _dicom_attribute = _attr[0].upper() + _attr[1:]
                # Update self property if None from study
                if getattr(self, _attr, None) is None:
                    setattr(self, _attr,
                            _get_attribute(self[_patientID], _dicom_attribute))
                elif _strict_values and\
                    getattr(self, _attr, None) != \
                        _get_attribute(self[_patientID], _dicom_attribute):
                    # Patient attributes differ, should be considered an exception.
                    raise ValueError('Cohort attribute "{}" differ ("{}" vs. "{}")'.format(
                        _attr,
                        getattr(self, _attr, None),
                        _get_attribute(self[_patientID], _dicom_attribute)
                    ))

    def __str__(self):
        return \
            "Cohort: {}".format(self.data)

    def _add_patients(self, data, opts=None):
        """Add each patient to self cohort

        Args:
            data: dict of Patient instances
            opts:

        Returns:
            self: added Patient instances to self
        """

        if issubclass(type(data), dict):
            for _patientID in data:
                if issubclass(type(data[_patientID], Patient)):
                    self[_patientID] = data[_patientID]
                else:
                    raise ValueError(
                        'Unexpected patient data type {}'.format(
                            type(data[_patientID])
                        ))
        else:
            raise ValueError('Unexpected data type {} (expected: dict)'.format(type(data)))

    def write(self, url, opts=None, formats=None):
        """Write image data, calling appropriate format plugins

        Args:
            self: Cohort instance
            url: output destination url
            opts: Output options (argparse.Namespace or dict)
            formats: list of output formats, overriding opts.output_format (list or str)
        Raises:
            imagedata.collections.UnknownOptionType: When opts cannot be made into a dict.
            TypeError: List of output format is not list().
            ValueError: Wrong number of destinations given,
                or no way to write multidimensional image.
            imagedata.formats.WriteNotImplemented: Cannot write this image format.
        """

        _used_urls = []
        for _patientID in self.keys():
            _patient = self[_patientID]
            for _studyInstanceUID in _patient:
                _study = _patient[_studyInstanceUID]
                try:
                    study_str = datetime.combine(
                        _study.studyDate, _study.studyTime
                    ).strftime('%Y%m%d-%H%M%S')
                except TypeError:
                    study_str = _studyInstanceUID
                for _seriesInstanceUID in _study:
                    _series = _study[_seriesInstanceUID]
                    try:
                        series_str = '{}-{}'.format(
                            _series.seriesNumber, _series.seriesDescription
                        )
                    except TypeError:
                        series_str = _seriesInstanceUID
                    _url = "{}/{}/{}/{}".format(
                        url,
                        _patientID,
                        study_str,
                        series_str
                    )
                    while _url in _used_urls:
                        _url = _url + "0"  # Make unique output file
                    _used_urls.append(_url)
                    try:
                        _series.write(_url, opts=opts, formats=formats)
                    except Exception as e:
                        raise Exception(_url) from e
