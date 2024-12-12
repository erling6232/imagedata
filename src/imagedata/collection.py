"""Image cohort

The Cohort class is a collection of Patient objects.

  Typical example usage:

  cohort = Cohort('input')

"""

# Copyright (c) 2023-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

from datetime import datetime, date, time
import argparse
from pathlib import Path

from .series import Series
from .readdata import read as r_read
from .formats import UnknownInputError


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
        _hdr, _si = r_read(_data, order='auto', opts=_opts)
        if len(_hdr) < 1:
            raise UnknownInputError('No input data found.')

        for _uid in _hdr:
            if _uid in _si:
                _series = Series(_si[_uid], opts=_opts)
            else:
                _series = Series(None, opts=_opts)
            _series.header = _hdr[_uid]
            _series_dict[_uid] = _series
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


# class IndexedDict(UserDict):
class IndexedDict(dict):

    def __init__(self):
        super(IndexedDict, self).__init__()

    def _item_to_key_(self, item):
        if isinstance(item, int):
            return [list(self.keys())[item]]
        elif isinstance(item, slice):
            uids = []
            start = 0 if item.start is None else item.start
            stop = len(self) if item.stop is None else item.stop
            step = 1 if item.step is None else item.step
            for i in range(start, stop, step):
                uids.append(list(self.keys())[i])
            return uids
        return [item]

    def __getitem__(self, item):
        keys = self._item_to_key_(item)
        items = []
        for key in keys:
            items.append(
                super(IndexedDict, self).__getitem__(key)
            )
        if len(items) == 1:
            return items[0]
        else:
            return items

    def __setitem__(self, item, val):
        keys = self._item_to_key_(item)
        if len(keys) != 1:
            raise IndexError('Bad index: {}'.format(item))
        super(IndexedDict, self).__setitem__(keys[0], val)

    def __repr__(self, item=None):
        if item is None:
            dictrepr = super(IndexedDict, self).__repr__()
            return '%s(%s)' % (type(self).__name__, dictrepr)
        keys = self._item_to_key_(item)
        s = ""
        for key in keys:
            dictrepr = super(IndexedDict, self).__repr__(key)
            s += '%s(%s)' % (type(self).__name__, dictrepr)
        return s

    def __str__(self):
        dictrepr = super(IndexedDict, self).__str__()
        return '%s(%s)' % (type(self).__name__, dictrepr)

    def update(self, *args, **kwargs):
        for item, v in dict(*args, **kwargs).items():
            keys = self._item_to_key_(item)
            key = keys[0]
            self[key] = v


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


class Study(IndexedDict):
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

    def __init__(self, data, opts=None, **kwargs):
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
        for key, value in kwargs.items():
            _in_opts[key] = value
        if 'input_options' in _in_opts:
            for key, value in _in_opts['input_options'].items():
                _in_opts[key] = value

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
            except Exception:
                raise

        for _seriesInstanceUID in _series_dict:
            _series = _series_dict[_seriesInstanceUID]
            self[_seriesInstanceUID] = _series
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
                elif _value is None:
                    pass
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
            except (TypeError, ValueError):
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


class Patient(IndexedDict):
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

    _strict_attributes = ['patientName', 'patientID', 'patientBirthDate', 'patientSex']

    def __init__(self, data, opts=None, **kwargs):

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
        for key, value in kwargs.items():
            _in_opts[key] = value
        if 'input_options' in _in_opts:
            for key, value in _in_opts['input_options'].items():
                _in_opts[key] = value

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
                        _attr in self._strict_attributes and \
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
            except (TypeError, ValueError):
                study_str = _studyInstanceUID
            for _seriesInstanceUID in _study:
                _series = _study[_seriesInstanceUID]
                try:
                    series_str = '{}-{}'.format(_series.seriesNumber, _series.seriesDescription)
                except (TypeError, ValueError):
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


class Cohort(IndexedDict):
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

    def __init__(self, data, opts=None, **kwargs):

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
        for key, value in kwargs.items():
            _in_opts[key] = value
        if 'input_options' in _in_opts:
            for key, value in _in_opts['input_options'].items():
                _in_opts[key] = value

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
                except (TypeError, ValueError):
                    study_str = _studyInstanceUID
                for _seriesInstanceUID in _study:
                    _series = _study[_seriesInstanceUID]
                    try:
                        series_str = '{}-{}'.format(
                            _series.seriesNumber, _series.seriesDescription
                        )
                    except (TypeError, ValueError):
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
