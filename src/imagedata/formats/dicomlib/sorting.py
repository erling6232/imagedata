from typing import Any, Union
import logging
import numpy as np
from numbers import Number
from collections import Counter, defaultdict, namedtuple
from functools import cmp_to_key
from operator import itemgetter
from collections.abc import Iterable
from pydicom.dataset import Dataset

from ...axis import VariableAxis, UniformLengthAxis
from ...header import Header
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


def scan_tags(sorted_dataset_list: SortedDatasetList, input_order: str, input_options: dict):
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
    for sloc in sorted(sorted_dataset_list.keys()):
        for im in sorted_dataset_list[sloc]:
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
            for im in sorted_dataset_list[sloc]:
                tag = im.get_tag_tuple(faulty, input_order, input_options)
                tag_list = im.replace_tag(tag_list, wanted_time_values, tag, time_tag)
        # Count values per dimension
        tag_values = {}
        shape = ()
        for i, sort_key in enumerate(input_order_list):
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

def sort_tags(hdr: Header,
              sorted_dataset_list: SortedDatasetList,
              input_order: str,
              input_options: dict,
              slice_count: Counter,
              opts: dict
              ) -> None:

    def collect_tags(sorted_data: list[Instance]) -> list[tuple]:
        """Collect tags from sorted data"""
        tag_list = []
        for im in sorted_data:
            tag_list.append(im.tags)
        return tag_list

    def calculate_shape(tag_list: list[tuple]) -> tuple[tuple[int], tuple[list]]:
        s = ()
        axes = ()
        if len(tag_list) == 0:
            return s, axes
        tags = len(tag_list[0])
        for i in range(tags):
            values = []
            for tag in tag_list:
                values.append(tag[i])
            if i == tags - 1 and accept_duplicate_tag:  # Accept duplicate along last axis
                axes += (values,)
            elif issubclass(type(values[0]), np.ndarray):
                vlist = [values[0]]
                for v in values[1:]:
                    found = False
                    for u in vlist:
                        if u.size != v.size:
                            continue
                        if np.allclose(u, v, rtol=1e-3, atol=1e-2):
                            found = True
                            break
                    if not found:
                        vlist.append(v)
                axes += (vlist,)
            else:
                axes += (list(dict.fromkeys(values)),)
            s += (len(axes[-1]),)
        return s, axes

    def calculate_shape_with_duplicates(sorted_data: list[Instance]) -> (
            tuple)[tuple[int], tuple[list]]:

        def _find_closest(tag_db: list, value: Union[Number, np.ndarray]) -> (
                tuple)[Union[int | None], Union[float | None]]:
            min_distance = np.inf
            min_index = None
            if issubclass(type(value), np.ndarray):
                for i in range(len(tag_db)):
                    if tag_db[i].size == value.size == 0:
                        min_distance = 0.0
                        min_index = i
                        break
                    if tag_db[i].size != value.size:
                        continue
                    if np.allclose(value, tag_db[i], rtol=1e-3, atol=1e-2):
                        min_distance = 0.0
                        min_index = i
                        break
                    else:
                        distance = np.linalg.norm(abs(value - tag_db[i]))
                        if distance < min_distance:
                            min_distance = distance
                            min_index = i
            else:
                try:
                    min_index = tag_db.index(value)
                    min_distance = abs(value - tag_db[min_index])
                except ValueError:
                    pass
            return min_index, min_distance

        _name: str = '{}.{}'.format(__name__, calculate_shape_with_duplicates.__name__)
        s = ()
        axes = ()
        tag_db = {}
        if len(sorted_data) == 0:
            return s, axes

        # Calculate tag shape
        im0 = sorted_data[0]
        tags = len(im0.tags)
        idx = [-1 for _ in range(tags)]
        previous_tag = tuple(None for _ in range(tags))
        for _ in range(tags):
            tag_db[_] = []

        for im in sorted_data:
            tag = im.tags
            add_tag = {}
            for t in reversed(range(tags)):
                cmp = compare_tag_values(previous_tag[t], tag[t])
                if cmp > 0:
                    for _ in range(t, tags):
                        min_index, min_distance = _find_closest(tag_db[_], tag[_])
                        if min_index is not None and min_distance < 1e-3:
                            idx[_] = min_index
                        else:
                            idx[_] = len(tag_db[_])
                            add_tag[_] = idx[_]
                elif cmp < 0:
                    min_index, min_distance = _find_closest(tag_db[t], tag[t])
                    if min_index is not None and min_distance < 1e-3:
                        idx[t] = min_index
                    else:
                        raise IndexError(f"{_name}: Cannot sort tags. Images should already be sorted.")
                elif t == tags - 1:
                    idx[t] += 1
                    add_tag[t] = idx[t]
            im.set_tag_index(tuple(idx))
            for t in add_tag:
                tag_db[t].insert(add_tag[t], tag[t])
            previous_tag = tag
        for _ in range(tags):
            s += (len(tag_db[_]),)
            axes += (tag_db[_],)
        return s, axes

    def locate_image(im: Instance) -> tuple[int]:
        """Locate image in sorted data"""
        s = ()
        _slice = im.slice_index
        axis = _axes[_slice]
        for i in range(len(im.tags)):
            # Find tag in axes
            if issubclass(type(im.tags[i]), np.ndarray):
                min_distance = np.inf
                min_index = None
                for j, v in enumerate(axis[i]):
                    if v.size != im.tags[i].size:
                        continue
                    if np.allclose(v, im.tags[i], rtol=1e-3, atol=1e-2):
                        min_distance = 0
                        min_index = j
                        break
                    else:
                        distance = np.linalg.norm(abs(v - im.tags[i]))
                        if distance < min_distance:
                            min_distance = distance
                            min_index = j
                s += (min_index,)
            else:
                s += (axis[i].index(im.tags[i]),)
        return s

    def place_images() -> dict[np.ndarray]:
        """Place images in sorted data"""
        tags = {}
        for _slice, sloc in enumerate(sorted(sorted_dataset_list)):
            tags[_slice] = np.empty(shape, dtype=tuple)
            for im in sorted_dataset_list[sloc]:
                _idx = locate_image(im)
                im.set_tag_index(_idx)
                tags[_slice][_idx] = im.tags
        return tags

    def place_images_with_duplicates() -> dict[np.ndarray]:
        """Place images in sorted data, allow duplicate tags along last axis"""
        tags = {}
        for _slice, sloc in enumerate(sorted(sorted_dataset_list)):
            tags[_slice] = np.empty(shape, dtype=tuple)
            for im in sorted_dataset_list[sloc]:
                _idx = im.tag_index
                # Is this index already taken?
                if tags[_slice][_idx] is not None:
                    raise CannotSort("{}: duplicate tag ({}): {}".format(_name, input_order, hdr.tags[_slice][_idx]))
                tags[_slice][_idx] = im.tags
        return tags

    _name: str = '{}.{}'.format(__name__, sort_tags.__name__)

    accept_duplicate_tag = 'accept_duplicate_tag' in opts and opts['accept_duplicate_tag']
    tag_list = defaultdict(list)
    sorted_data = defaultdict(list)
    faulty = 0
    sloc: float
    _shapes = []
    _axes = []
    for _slice, sloc in enumerate(sorted(sorted_dataset_list)):
        im: Instance
        for im in sorted_dataset_list[sloc]:
            im.set_slice_index(_slice)
            im.set_tags(im.get_tag_tuple(faulty, input_order, input_options))
            faulty += 1
        sorted_data[_slice] = sorted(sorted_dataset_list[sloc], key=cmp_to_key(compare_tags))
        if accept_duplicate_tag:
            s, axis = calculate_shape_with_duplicates(sorted_data[_slice])
        else:
            tag_list[_slice] = collect_tags(sorted_data[_slice])
            s, axis = calculate_shape(tag_list[_slice])
        _shapes.append(s)
        _axes.append(axis)

    # Find maximum shape in slices
    shape = ()
    for i in range(len(_shapes[0])):
        shape += (max(_shapes, key=itemgetter(i))[i],)

    # Place each image on the proper tag index
    if accept_duplicate_tag:
        hdr.tags = place_images_with_duplicates()
    else:
        hdr.tags = place_images()

    # Get image dimensions and SOPInstanceUIDs from header
    SOPInstanceUIDs = {}
    frames = None
    rows = columns = 0
    for _slice, sloc in enumerate(sorted(sorted_dataset_list)):
        im: Instance
        for im in sorted_dataset_list[sloc]:
            rows = max(rows, im.Rows)
            columns = max(columns, im.Columns)
            if 'NumberOfFrames' in im:
                frames = im.NumberOfFrames
            _idx = im.tag_index
            SOPInstanceUIDs[_idx + (_slice,)] = im.SOPInstanceUID

    # Simplify shape dimension
    while len(shape) and shape[0] == 1:
        shape = shape[1:]
        # _axes = _axes[1:]
    hdr.dicomTemplate = sorted_dataset_list[next(iter(sorted_dataset_list))][0]
    hdr.SOPInstanceUIDs = SOPInstanceUIDs
    nz = len(sorted_dataset_list)
    if frames is not None and frames > 1:
        nz = frames
    try:
        ipp = hdr.dicomTemplate.get_image_position_patient()
    except ValueError:
        ipp = np.array([0, 0, 0])
    hdr.spacing = sorted_dataset_list.spacing
    slice_axis = UniformLengthAxis('slice', ipp[0], nz, hdr.spacing[0])
    row_axis = UniformLengthAxis('row', ipp[1], rows, hdr.spacing[1])
    column_axis = UniformLengthAxis('column', ipp[2], columns, hdr.spacing[2])
    if len(shape):
        tag_axes = []
        for i, order in enumerate(input_order.split(sep=',')):
            tag_axes.append(
                VariableAxis(order, _axes[0][i])
            )
        axis_names = input_order.split(sep=',')
        axis_names.extend(['slice', 'row', 'column'])
        Axes = namedtuple('Axes', axis_names)
        axes = Axes(*tag_axes, slice_axis, row_axis, column_axis)
    elif nz > 1:
        Axes = namedtuple('Axes', [
            'slice', 'row', 'column'
        ])
        axes = Axes(slice_axis, row_axis, column_axis)
    else:
        Axes = namedtuple('Axes', [
            'row', 'column'
        ])
        axes = Axes(row_axis, column_axis)
    hdr.color = False
    if 'SamplesPerPixel' in hdr.dicomTemplate and hdr.dicomTemplate.SamplesPerPixel == 3:
        hdr.color = True
    hdr.axes = axes
