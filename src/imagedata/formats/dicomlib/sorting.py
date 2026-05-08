from typing import Any, Union
import logging
import numpy as np
from collections import Counter, namedtuple

from ...axis import VariableAxis
from ...header import Header
from ...formats import CannotSort, INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_TRIGGERTIME
from .datatypes import (DatasetDict, DatasetList, SortedData, SortedDatasetList, SortedDataDict,
                        SortedHeaderDict)
from .instance import Instance


logger = logging.getLogger(__name__)


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
            return -1 if np.all(t1 < t2) else 1
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
            elif t2[i].size == 0:
                return 1
            elif np.allclose(t1[i], t2[i], rtol=1e-3, atol=1e-2):
                continue
            else:
                return -1 if np.linalg.norm(t2[i] - t1[i]) < 0 else 1
                # return -1 if np.all(t1[i] < t2[i]) else 1
        elif isinstance(t1[i], tuple):
            _ = compare_tuples(t1[i], t2[i])
            if _ != 0:
                return _
        elif t1[i] == t2[i]:
            continue
        else:
            return -1 if t1[i] < t2[i] else 1
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


def _get_unique_tag_values(tag_list: list, lookup: int) -> list:
    def _value_in_list(tag_list, value) -> bool:
        for v in tag_list:
            found = True
            for tag in range(len(value)):
                if issubclass(type(value[tag]), np.ndarray):
                    if v[tag].size != value[tag].size:
                        continue
                    found = found and np.allclose(v[tag], value[tag], rtol=1e-3, atol=1e-2)
                else:
                    found = found and (v[tag] == value[tag])
            if found:
                return True
        return False

    tag_values = []
    for _, tag in tag_list:
        tag_values.append(tag[lookup])

    if issubclass(type(tag_values[0]), tuple):
        vlist = []
        for values in tag_values:
            if not _value_in_list(vlist, values):
                vlist.append(values)
        return vlist
    elif issubclass(type(tag_values[0]), np.ndarray):
        vlist = [tag_values[0]]
        for v in tag_values[1:]:
            found = False
            for u in vlist:
                if u.size != v.size:
                    continue
                if np.allclose(u, v, rtol=1e-3, atol=1e-2):
                    found = True
                    break
            if not found:
                vlist.append(v)
        return vlist
    else:
        return sorted(set(tag_values))


def _get_fixed_tags(tag_list: list, fixed: list, lookup: int|None) -> tuple[list, list]:
    tags = len(fixed)
    values = []
    fixed_mask = [False for _ in range(tags)]
    for _, tag in tag_list:
        found = True
        for i in range(tags):
            if i != lookup:
                fixed_mask[i] = True
                if fixed is not None and tag[i] != fixed[i]:
                    found = False
        if found and tag[lookup] not in values:
            values.append(tag[lookup])
    return sorted(values), fixed_mask


def _determine_tag_position(axes: list[list],
                            tag: tuple,
                            tag_list: list[tuple],
                            time_tag: int|None) -> tuple[int]:

    def _determine_tag_in_list(tag_list: list, tag: tuple) -> int:
        for i, _tag in enumerate(tag_list):
            if compare_tuples(tag, _tag) == 0:
                return i
        raise IndexError('Tag not found')

    _name: str = '{}.{}'.format(__name__, _determine_tag_position.__name__)
    new_tag = ()
    for i, _tag in enumerate(tag):
        if time_tag is not None and i == time_tag:
            _times, _mask = _get_fixed_tags(tag_list, tag, time_tag)
            what_tag = _times.index(_tag)
        else:
            try:
                what_tag = _determine_tag_in_list(axes[i], _tag)
            except IndexError:
                what_tag = i
        new_tag += (what_tag,)
    return new_tag


def scan_tags(sorted_dataset_list: SortedDatasetList, input_order: str, input_options: dict) -> tuple[tuple, dict]:
    def _minimum_tag(tags: list, tag: int) -> Any:
        minimum = 1e9
        for _, _tag in tags:
            if _tag[tag] < minimum:
                minimum = _tag[tag]
        return minimum

    _name: str = '{}.{}'.format(__name__, scan_tags.__name__)
    accept_duplicate_tag = 'accept_duplicate_tag' in input_options and input_options['accept_duplicate_tag']
    input_order_list = input_order.split(sep=',')
    if input_order_list == [INPUT_ORDER_NONE]:
        input_order_list = []
    shape = ()
    faulty = 0
    previous_shape = None
    tags = {}
    _all_axes = {}
    for sloc in sorted(sorted_dataset_list.keys()):
        new_input_order_list = input_order_list.copy()
        tag_list = []
        for im in sorted_dataset_list[sloc]:
            tag_list.append((im, im.get_tag_tuple(faulty, input_order, input_options)))
            faulty += 1
        _axes = []
        time_tag = None
        for i, in_order in enumerate(input_order_list):
            if in_order == INPUT_ORDER_TIME:
                # Replace variable time with minimum time
                minimum_tag = []
                time_tag = None
                for i, sort_key in enumerate(input_order_list):
                    if sort_key == INPUT_ORDER_TIME:
                        time_tag = i
                    minimum_tag.append(_minimum_tag(tag_list, i))
                wanted_time_values, fixed_mask = _get_fixed_tags(tag_list, fixed=minimum_tag, lookup=time_tag)
                _axes.append(wanted_time_values)
            else:
                _axes.append(_get_unique_tag_values(tag_list, lookup=i))
        _done = {}
        duplicate_tags = False
        for im in sorted_dataset_list[sloc]:
            tag = im.get_tag_tuple(faulty, input_order, input_options)
            idx = _determine_tag_position(_axes, tag, tag_list, time_tag)
            if im.tag_index is None:
                im.set_tags(tag)
                im.set_tag_index(idx)
            else:
                raise CannotSort('Tag index should not already exist')
            if idx in _done:
                duplicate_tags = True
            _done[idx] = True
        # Count values per dimension
        tag_values = {}
        shape = ()
        for i, sort_key in enumerate(input_order_list):
            values = _get_unique_tag_values(tag_list, lookup=i)
            shape += (len(_axes[i]),)
            tag_values[sort_key] = values

        if duplicate_tags and accept_duplicate_tag:
            _axes = scan_duplicate_tags(sorted_dataset_list[sloc], _axes)
            shape = ()
            for _axis in _axes:
                shape += (len(_axis),)

        # Verify exact same shape per slice
        if previous_shape is None:
            previous_shape = shape
        else:
            if previous_shape != shape:
                raise CannotSort(f'{_name}: Shape differ in each slice')

        _all_axes[sloc] = _axes

    try:
        _ref_axes = _all_axes[next(iter(_all_axes))][0]
        for sloc in sorted(sorted_dataset_list.keys()):
            for _ in range(len(_ref_axes)):
                if not np.allclose(_ref_axes[_], _all_axes[sloc][_], rtol=1e-3, atol=1e-2):
                    raise CannotSort(f'{_name}: Axes differ for input order {new_input_order_list[_]} in each slice')
    except IndexError:
        pass

    for _slice, sloc in enumerate(sorted(sorted_dataset_list.keys())):
        _done = {}
        if len(shape):
            tags[_slice] = np.empty(shape, dtype=tuple)
        else:
            tags[_slice] = np.empty(1, dtype=tuple)
        for im in sorted_dataset_list[sloc]:
            _idx = im.tag_index
            if _idx not in _done:
                tags[_slice][_idx] = im.tags
                _done[_idx] = True
            else:
                raise CannotSort('Duplicate tag')

    # Construct Axes namedtuple
    _new_axes = []
    for i, sort_key in enumerate(new_input_order_list):
        _new_axes.append(VariableAxis(sort_key, _axes[i]))
    Axes = namedtuple('Axes', new_input_order_list)
    axes = Axes._make(_new_axes)

    return axes, tags


def scan_duplicate_tags(dataset_list: DatasetList, axes: list) -> list:

    # Investigate duplicate items
    _done = {}
    for im in dataset_list:
        _idx = im.tag_index
        if _idx not in _done:
            _done[_idx] = [im]
        else:
            _done[_idx].append(im)

    # Move duplicate items
    _new_axis = []
    _count = {}
    _prefix_idx = None  # next(iter(_done))[:-1]
    for _idx in _done:
        if _idx[:-1] != _prefix_idx:
            _prefix_idx = _idx[:-1]
        if not _prefix_idx in _count:
            _count[_prefix_idx] = 0  # Reset count for next prefix index
        for im in _done[_idx]:
            _new_idx = _idx[:-1] + (_count[_prefix_idx],)
            im.set_tag_index(_new_idx)
            _new_axis.append(im.tags[-1])
            _count[_prefix_idx] += 1

    try:
        axes[-1] = _new_axis
    except IndexError:
        axes = [_new_axis]

    # Verify
    # _keys = sorted([_im.tag_index for _im in dataset_list])
    # for _key in _keys:
    #     for im in dataset_list:
    #         if  im.tag_index == _key:
    #             t = im.get_acquisition_time()
    #             te = im.get_echo_time()
    #             print(_key, t, te)

    return axes


def verify_consistent_slices(sorted_dataset_list: SortedDatasetList, message: str, opts: dict = None) -> Counter:
    _name: str = '{}.{}'.format(__name__, verify_consistent_slices.__name__)
    """Verify same number of images for each slice.
    """
    slice_count = Counter()
    last_sloc = None
    for islice, sloc in enumerate(sorted_dataset_list):
        slice_count[islice] = len(sorted_dataset_list[sloc])
        last_sloc = sloc
    logger.debug("{}: tags per slice: {}".format(_name, slice_count))
    accept_uneven_slices = 'accept_uneven_slices' in opts and opts['accept_uneven_slices']
    min_slice_count = min(slice_count.values())
    max_slice_count = max(slice_count.values())
    if min_slice_count != max_slice_count and not accept_uneven_slices:
        logger.error("{}: tags per slice: {}".format(message, slice_count))
        raise CannotSort(
            "{}: ".format(message) +
            "Different number of images in each slice. Tags per slice:\n{}".format(slice_count) +
            "\nLast file: {}".format(sorted_dataset_list[last_sloc][0].filename) +
            "\nCould try 'split_acquisitions=True' or 'split_echo_numbers=True'."
        )
    return slice_count


def combine_data_and_header(dataset_dict: DatasetDict, header_dict: SortedHeaderDict, opts: dict) -> SortedDataDict:
    sorted_data_dict: SortedDataDict = SortedDataDict()
    for seriesUID in header_dict:
        header: Header = header_dict[seriesUID]
        dataset_list: DatasetList = dataset_dict[seriesUID]
        sorted_data_dict[seriesUID] = SortedData((dataset_list, header))
        pass
    return sorted_data_dict