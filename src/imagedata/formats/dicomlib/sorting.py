from typing import Any
import logging
import numpy as np
from numbers import Number
from collections import Counter
from collections.abc import Iterable
from pydicom.dataset import Dataset

from .datatypes import SortedDatasetList
from ...formats import CannotSort, INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_TRIGGERTIME
from .instance import Instance, _get_fixed_tags


logger = logging.getLogger(__name__)



def get_tag_value(im: Dataset, input_order: str, input_options: dict = None) -> Number:
    """Calculate value to sort on from the DICOM header"""
    _object = im.get_tag(input_order, input_options)
    if not isinstance(_object, Iterable):
        _object = [_object]

    _sum = 0
    for _item in _object:
        if issubclass(type(_item), np.ndarray):
            # Typical array value is the MRI diffusion b vector
            # To ensure consistent sorting of b-vectors, the different directions are
            # weighted (arbitrarily) by the position index in the vector
            _sum += np.dot(_item, np.array(np.arange(_item.size) + 1))
        else:
            _sum += _item
    return _sum


def compare_tag_values(t1, t2):
    if t1 is None:
        return 1
    if issubclass(type(t1), np.ndarray):
        if t1.size == 0 and t2.size == 0:
            return 0
        elif t1.size == 0:
            return 1
        elif t2.size == 0:
            return -1
        elif np.allclose(t1, t2, rtol=1e-3, atol=1e-2):
            return 0
        else:
            return 1  # Changed ndarray is always treated as larger
    elif t1 == t2:
        return 0
    else:
        return (t1 < t2) * 2 - 1


def compare_tuples(t1: tuple, t2: tuple) -> int:
    """Compare each element of two tuples
    Return -1, 0 or 1 depending on whether the first argument is lower than, equal
    to or greater than the second argument
    """
    for i in range(len(t1)):
        if issubclass(type(t1[i]), np.ndarray):
            if t1[i].size == 0:
                return -1
                # return True
            elif t2[i].size == 0:
                return 1
                # return False
            elif np.allclose(t1[i], t2[i], rtol=1e-3, atol=1e-2):
                continue
            else:
                return -1 if np.all(t1[i] < t2[i]) else 1
                # return np.all(t1[i] < t2[i])
        elif isinstance(t1[i], tuple):
            _ = compare_tuples(t1[i], t2[i])
            if _ != 0:
                return _
        elif t1[i] == t2[i]:
            continue
        else:
            return -1 if t1[i] < t2[i] else 1
            # return t1[i] < t2[i]
    return 0


def compare_tags(im1: Instance, im2: Instance) -> int:
    t1 = im1.tags
    t2 = im2.tags
    return compare_tuples(t1, t2)


def determine_sorting(sorted_dataset_dict: SortedDatasetList,
                      input_order: str,
                      input_options: dict[str]) -> str:

    def _single_slice_over_time(tags):
        """If time and slice both varies, the time stamps address slices of a single volume
        """
        count_time = {}
        count_sloc = {}
        for time, sloc in tags:
            if time not in count_time:
                count_time[time] = 0
            if sloc not in count_sloc:
                count_sloc[sloc] = 0
            count_time[time] += 1
            count_sloc[sloc] += 1
        max_time = max(count_time.values())
        max_sloc = max(count_sloc.values())
        return max_time == 1 and max_sloc == 1

    if input_order != 'auto':
        return input_order
    extended_tags = {}
    found_tags = {}
    im: Instance = None
    for sloc in sorted_dataset_dict.keys():
        for im in sorted_dataset_dict[sloc]:
            for order in input_options['auto_sort']:
                try:
                    tag = im.get_tag(order, input_options)
                    if tag is None:
                        continue
                    if order not in found_tags:
                        found_tags[order] = []
                        extended_tags[order] = []
                    if tag not in found_tags[order]:
                        found_tags[order].append(tag)
                        extended_tags[order].append((tag, sloc))
                except (KeyError, TypeError, CannotSort):
                    pass

    # Determine how to sort
    actual_order = None
    for order in found_tags:
        if len(found_tags[order]) > 1:
            if actual_order in ('time', 'triggertime') and order in ['b', 'te']:
                # DWI images will typically have varying time.
                # Let b values override time stamps.
                actual_order = order
            elif actual_order is None:
                actual_order = order
            else:
                raise CannotSort('Cannot auto-sort: {}\n'.format(extended_tags) +
                                 '  actual_order: {}, order: {},'.format(actual_order, order) +
                                 ' Series #{}: {}'.format(im.SeriesNumber, im.SeriesDescription)
                                 )
    if actual_order is None:
        actual_order = INPUT_ORDER_NONE
    elif actual_order in (INPUT_ORDER_TIME, INPUT_ORDER_TRIGGERTIME) and \
            _single_slice_over_time(extended_tags[actual_order]):
        actual_order = INPUT_ORDER_NONE
    return actual_order


def scan_tags(sorted_dataset: SortedDatasetList, input_order: str, input_options: dict):
    def _minimum_tag(tags: list, tag: int) -> Any:
        minimum = 1e9
        for _, _tag in tags:
            if _tag[tag] < minimum:
                minimum = _tag[tag]
        return minimum

    _name: str = '{}.{}'.format(__name__, scan_tags.__name__)
    input_order_list = input_order.split(sep=',')
    tag_values = {}
    shape = ()
    faulty = 0
    tag_list = []
    previous_shape = None
    # for im in image_dict[seriesUID]:
    for sloc in sorted(sorted_dataset.keys()):
        for im in sorted_dataset[sloc]:
            tag_list.append((im, im.get_tag_tuple(faulty, input_order, input_options)))
            faulty += 1
        if INPUT_ORDER_TIME in input_order_list:
            # Replace time with
            minimum_tag = []
            time_tag = None
            for i, sort_key in enumerate(input_order_list):
                if sort_key == INPUT_ORDER_TIME:
                    time_tag = i
                minimum_tag.append(_minimum_tag(tag_list, i))
            wanted_time_values, fixed_mask = _get_fixed_tags(tag_list, fixed=minimum_tag, lookup=time_tag)
            for im in sorted_dataset[sloc]:
                tag = im.get_tag_tuple(faulty, input_order, input_options)
                tag_list = im.replace_tag(tag_list, wanted_time_values, tag, time_tag)
        # Count values per dimension
        tag_values = {}
        shape = ()
        for i, sort_key in enumerate(input_order):
            values = []
            for im, tag in tag_list:
                values.append(tag[i])
            values = sorted(set(values))
            shape += (len(values),)
            tag_values[sort_key] = values
        # Verify exact same shape per slice
        if previous_shape is not None:
            if previous_shape != shape:
                raise CannotSort(f'{_name}: Shape differ in each slice')
    return tag_values, shape


def verify_consistent_slices(series: SortedDatasetList, message: str, opts: dict = None) -> Counter:
    _name: str = '{}.{}'.format(__name__, verify_consistent_slices.__name__)
    # Verify same number of images for each slice
    slice_count = Counter()
    last_sloc = None
    for islice, sloc in enumerate(series):
        slice_count[islice] = len(series[sloc])
        last_sloc = sloc
    logger.debug("{}: tags per slice: {}".format(_name, slice_count))
    accept_uneven_slices = False
    if 'accept_uneven_slices' in opts and opts['accept_uneven_slices']:
        accept_uneven_slices = True
    min_slice_count = min(slice_count.values())
    max_slice_count = max(slice_count.values())
    if min_slice_count != max_slice_count and not accept_uneven_slices:
        logger.error("{}: tags per slice: {}".format(message, slice_count))
        raise CannotSort(
            "{}: ".format(message) +
            "Different number of images in each slice. Tags per slice:\n{}".format(slice_count) +
            "\nLast file: {}".format(series[last_sloc][0].filename) +
            "\nCould try 'split_acquisitions=True' or 'split_echo_numbers=True'."
        )
    return slice_count

