"""Image series

The Series class is a subclassed Numpy.ndarray enhancing the array with relevant medical image
methods and attributes.

  Typical example usage:

  si = Series('input')

"""

import collections.abc
from typing import Tuple
import copy
import numbers
import argparse
from collections import namedtuple
import numpy as np
import logging
from pathlib import PurePath
import pydicom.dataset
import pydicom.datadict

from .axis import UniformAxis, UniformLengthAxis, to_namedtuple
from .formats import INPUT_ORDER_NONE, INPUT_ORDER_TIME, INPUT_ORDER_B
from .formats import shape_to_str, input_order_set, sort_on_set
from .formats.dicomlib.uid import get_uid_for_storage_class
from .readdata import read as r_read, write as r_write
from .header import Header

# from ._methods import (max, nanmax, min, nanmin, __sub__, multiply,
#                        __mul__, __imul__, __rmul__, __rmatmul__, __matmul__, __truediv__, rint)

logger = logging.getLogger(__name__)

HANDLED_FUNCTIONS = {}  # Functions handled by __array_function__()


def implements(numpy_function):
    """Register an __array_function__ implementation for Series objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


class DoNotSetSlicesError(Exception):
    pass


class MultipleSeriesError(Exception):
    pass


class Series(np.ndarray):
    """Series -- a multidimensional array of medical imaging pixels.

    The Series class is a subclassed numpy.ndarray enhancing the ndarray
    with relevant medical image methods and attributes.

    Examples:

        Read the contents of an input directory

        >>> image = Series('directory/')

        Make a Series instance from a Numpy.ndarray

        >>> a = np.eye(128)
        >>> image = Series(a)

    Args:
        data (array_like or URL): Input data, either explicit as np.ndarray, np.uint16, np.float32,
            or by URL to input data.

        input_order (str or tuple[str]): How to sort the input data. Typical values are:

            - 'auto' : auto-detect sort criteria (default).
            - 'none' : 3D volume or 2D slice.
            - 'time' : Time-resolved data.
            - 'b' : Diffusion data with variable b values.
            - 'bvector' : Diffusion data with variable gradient directions
            - 'rsi' : Diffusion data with variable b values and gradient directions
            - 'te' : Varying echo times.
            - 'fa' : Varying flip angles.

        opts (argparse.Namespace or dict): Dict of input options,
            mostly for format specific plugins.

        input_format (str): Specify a particular input format. Default: None (auto-detect).
        shape (tuple of ints): Specifying shape of input data.
        dtype (numpy.dtype): Numpy data type. Default: inferred from the input data.
        template (Series, array_like or URL): Input data to use as template for DICOM header.
        geometry (Series, array_like or URL): Input data to use as template for geometry.
        axes (namedtuple or iterable of Axis): Set axes for new instance.
        order: Row-major (C-style) or column-major (Fortran-style) order, {'C', 'F'}, optional

    Returns:
        Series: Series instance
    """

    name = "Series"
    description = "Image series"
    authors = "Erling Andersen"
    version = "1.4.0"
    url = "www.helse-bergen.no"

    viewer = None
    latest_roi_parameters = None

    def __new__(cls, data, input_order='auto', opts=None,
                input_format=None, shape=(0,), dtype=None, buffer=None, offset=0,
                strides=None, order=None,
                template=None, geometry=None, axes=None,
                **kwargs):

        _name: str = '{}.{}'.format(__name__, cls.__new__.__name__)

        if opts is None:
            opts = {}
        elif issubclass(type(opts), argparse.Namespace):
            opts = vars(opts)
        for key, value in kwargs.items():
            opts[key] = value
        if 'input_options' in opts:
            for key, value in opts['input_options'].items():
                opts[key] = value
        if axes is not None and type(axes) != 'Axes':
            axes = to_namedtuple(axes)

        if issubclass(type(template), Series):
            template = template.header
        if issubclass(type(geometry), Series):
            geometry = geometry.header
        if issubclass(type(data), np.ndarray):
            logger.debug('{}: data ({}) is subclass of np.ndarray'.format(_name, type(data)))
            obj = np.asarray(data, dtype).view(cls)
            # Initialize attributes to defaults
            # cls.__init_attributes(cls, obj)
            # obj.header = Header() # Already set in __array_finalize__

            # set the new 'input_order' attribute to the value passed
            if input_order == 'auto':
                obj.header.input_order = 'none'
            else:
                obj.header.input_order = input_order
            if obj.header.input_order != 'none':
                try:
                    getattr(obj.header.axes, input_order)
                except AttributeError:
                    # Set first axis from input_order
                    _fields = list(obj.header.axes._fields)
                    _values = list(obj.header.axes)
                    for i, order in enumerate(obj.header.input_order.split(',')):
                        _fields[i] = order
                        _values[i].name = order
                    Axes = namedtuple('Axes', _fields)
                    obj.header.axes = Axes._make(_values)


            if issubclass(type(data), Series):
                # Copy attributes from existing Series to newly created obj
                obj.header = copy.copy(data.header)  # carry forward attributes
                obj.input_order = data.input_order
                obj.header.add_template(data.header)
                obj.header.add_geometry(data.header)
            else:
                obj.header.set_default_values(obj.axes if axes is None else axes)

            # obj.header.set_default_values() # Already done in __array_finalize__
            if axes is not None:
                obj.header.axes = copy.copy(axes)
            obj.header.add_template(template)
            obj.header.add_geometry(geometry)
            return obj
        logger.debug('{}: data is NOT subclass of Series, type {}'.format(_name, type(data)))

        # Assuming data is url to input data
        if isinstance(data, str) or issubclass(type(data), PurePath):
            urls = data
        elif isinstance(data, list):
            urls = data
        else:
            if np.ndim(data) == 0:
                obj = np.asarray([data], dtype).view(cls)
            else:
                obj = np.asarray(data, dtype).view(cls)
            # cls.__init_attributes(cls, obj)
            obj.header = Header()
            if input_order == 'auto':
                obj.header.input_order = 'none'
            else:
                obj.header.input_order = input_order
            obj.header.input_format = type(data)
            if np.ndim(data) == 0:
                Axes = namedtuple('Axes', 'number')
                obj.header.axes = Axes(UniformAxis('number', 0, 1))
            obj.header.set_default_values(obj.axes if axes is None else axes)
            obj.header.add_template(template)
            obj.header.add_geometry(geometry)
            return obj

        # Read input, hdr is dict of attributes
        hdr, si = r_read(urls, input_order, opts, input_format)
        if len(hdr) > 1:
            raise MultipleSeriesError('Multiple (n={}) series found in Series'.format(len(hdr)))
        hdr = hdr[next(iter(hdr))]
        if 'headers_only' in opts and opts['headers_only']:
            si = None
        elif len(si):
            si = si[next(iter(si))]
        else:
            si = None
        obj = np.asarray(si, dtype).view(cls)
        assert obj.header, "No Header found in obj.header"

        # Copy attributes from hdr dict to newly created obj
        logger.debug('{}: Copy attributes from hdr dict to newly created obj'.format(_name))
        if axes is not None:
            obj.axes = copy.copy(axes)
        elif hdr.axes is not None:
            obj.axes = hdr.axes
        obj.header.set_default_values(obj.axes)
        obj.header.add_template(hdr)
        obj.header.add_geometry(hdr)
        # for attr in __attributes(hdr):
        #     __set_attribute(obj.header, attr, __get_attribute(template, attr))
        #     setattr(obj.header, attr, hdr[attr])
        # Store any template and geometry headers,
        obj.header.add_template(template)
        obj.header.add_geometry(geometry)
        # set the new 'input_order' attribute to the value passed
        obj.header.input_order = hdr.input_order
        obj.header.input_format = hdr.input_format
        obj.header.windowCenter = hdr.windowCenter
        obj.header.windowWidth = hdr.windowWidth
        # Finally, we must return the newly created object
        return obj

    def __array_finalize__(self, obj) -> None:
        # logger.debug("Series.__array_finalize__: entered obj: {}".format(type(obj)))
        # ``self`` is a new object resulting from
        # ndarray.__new__(Series, ...), therefore it only has
        # attributes that the ndarray.__new__ constructor gave it -
        # i.e. those of a standard ndarray.
        #
        # We could have got to the ndarray.__new__ call in 3 ways:
        # From an explicit constructor - e.g. Series():
        #    obj is None
        #    (we're in the middle of the Series.__new__
        #    constructor, and self.__input_order will be set when we return to
        #    Series.__new__)
        if obj is None:
            return
        # From view casting - e.g arr.view(Series):
        #    obj is arr
        #    (type(obj) can be Series)
        # From new-from-template - e.g Series[:3]
        #    type(obj) is Series
        #
        # Note that it is here, rather than in the __new__ method,
        # that we set the default value for 'input_order', because this
        # method sees all creation of default objects - with the
        # Series.__new__ constructor, but also with arr.view(Series).
        # logger.debug("Series.__array_finalize__: obj: {}".format(type(obj)))

        # if issubclass(type(obj), Series):
        if hasattr(obj, 'header') and issubclass(type(obj.header), Header):
            # Copy attributes from obj to newly created self
            # logger.debug('Series.__array_finalize__: Copy attributes from {}'.format(type(obj)))
            # self.__dict__ = obj.__dict__.copy()  # carry forward attributes
            self.header = copy.copy(obj.header)  # carry forward attributes
        else:
            self.header = Header()
            self.header.set_default_values(self.axes)
            self.windowCenter = None
            self.windowWidth = None

        # We do not need to return anything

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        if 'where' in kwargs:
            if issubclass(type(kwargs['where']), Series):
                kwargs['where'] = kwargs['where'].view(np.ndarray)
        for i, input_ in enumerate(inputs):
            if issubclass(type(input_), np.ndarray) and input_.dtype.fields is not None:
                # Delegate structured dtypes to __array_ufunc_struct__
                return self.__array_ufunc_struct__(ufunc, method, *inputs, **kwargs)
            if issubclass(type(input_), Series):
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if issubclass(type(output), Series):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        results = super(Series, self).__array_ufunc__(ufunc, method,
                                                      *args, **kwargs)
        # results = getattr(ufunc, method)(*inputs, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            # if issubclass(type(inputs[0]), Series):
            #     inputs[0].header = info
            return

        if ufunc.nout == 1:
            if np.isscalar(results):
                return results  # Do not pack scalar results as Series object
            results = (results,)

        results = tuple((np.asarray(result).view(Series)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        if results and issubclass(type(results[0]), Series):
            results[0].header = self._unify_headers(inputs)
            try:
                results[0].header.windowCenter = None
                results[0].header.windowWidth = None
            except AttributeError:
                pass

        return results[0] if len(results) == 1 else results

    def __array_ufunc_struct__(self, ufunc, method, *inputs, **kwargs):
        struct_dtype = None
        field_names = None
        if 'where' in kwargs:
            if issubclass(type(kwargs['where']), Series):
                kwargs['where'] = kwargs['where'].view(np.ndarray)
        for i, input_ in enumerate(inputs):
            if not (not issubclass(type(input_), np.ndarray) or not (
                    input_.dtype.fields is not None)) and struct_dtype is None:
                struct_dtype = input_.dtype
                field_names = input_.dtype.names
            if issubclass(type(input_), Series):
                if input_.dtype.names != field_names:
                    # Require same field names, not necessarily same datatypes
                    raise IndexError('Structured dtype differ: {} vs {}'.format(
                        field_names, input_.dtype.names
                    ))

        outputs = kwargs.pop('out', None)
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if output.dtype.fields is not None:
                    raise IndexError('Output struct dtype not implemented')
                if issubclass(type(output), Series):
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        _results = {}
        # Assume each channel of dtype struct can be calculated independently
        results_dtype = []
        for field in field_names:
            args = []
            for input_ in inputs:
                if issubclass(type(input_), Series):
                    if input_.dtype.fields is not None:
                        args.append(input_[field].view(np.ndarray))
                    else:
                        args.append(input_.view(np.ndarray))
                else:
                    args.append(input_)
            # _results[_channel] = super(Series, self).__array_ufunc__(ufunc, method,
            #                                                        *args, **kwargs)
            results = self.__array_ufunc__(ufunc, method,
                                           *args, **kwargs)
            if results is NotImplemented:
                return NotImplemented
            if method == 'at':
                return
            if not np.isscalar(results):
                results.header = self._unify_headers(inputs)
                results.header.windowCenter = None
                results.header.windowWidth = None
                results_dtype.append((field, results.dtype))
            else:
                results_dtype.append((field, type(results)))
            _results[field] = results
        if ufunc.nout == 1:
            if np.isscalar(results):
                _list = []
                for field in field_names:
                    _list.append(_results[field])
                return tuple(tuple(_list))
        #         raise ValueError("ufunc on _results")
        #         return results  # Do not pack scalar results as Series object
        #     # results = (results,)
        # else:
        if ufunc.nout != 1:
            raise ValueError('What to do with multiple color results?')
        # results = _results['R'].to_channels(
        results = self.to_channels(
            [_results[field] for field in field_names],
            field_names
        )
        return results if outputs[0] is None else outputs[0]
        # results = tuple((result
        #                  if output is None else output)
        #                 for result, output in zip(results, outputs))
        # return results[0] if len(results) == 1 else results
        #
        # if results and issubclass(type(results[0]), Series):
        #     results[0].header = self._unify_headers(inputs)
        #     if results[0].header is not None:
        #         _level, _width = results[0].__calculate_window()
        #         results[0].header.windowCenter = _level
        #         results[0].header.windowWidth = _width
        #
        # return results[0] if len(results) == 1 else results

    def __array_function__(self, func, types, args, kwargs):
        if func not in HANDLED_FUNCTIONS:
            # Let NumPy handle the function, possibly returning an NdArray, not Series
            return super().__array_function__(func, types, args, kwargs)
            # return NotImplemented
        # Note: this allows subclasses that don't override
        # __array_function__ to handle Series objects
        if not all(issubclass(t, Series) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    @staticmethod
    def _unify_headers(inputs: tuple) -> Header:
        """Unify the headers of the inputs.

        Typical usage is in expressions like c = a + b where at least
        one of the arguments is a Series instance.
        This function will provide a header for the result of
        the expression.

        Args:
            inputs (ndarray or Series): a tuple of arguments
        Returns:
            Header: Unified header.
        """

        header = None
        # logger.debug('Series._unify_headers: inputs {}'.format(len(inputs)))
        for i, input_ in enumerate(inputs):
            # logger.debug('Series._unify_headers: input {}: {}'.format(i, type(input_)))
            if issubclass(type(input_), Series):
                if input_.header is None:
                    # logger.debug('Series._unify_headers: new header')
                    header = Header()
                    header.input_order = INPUT_ORDER_NONE
                else:
                    # logger.debug('Series._unify_headers: copy header')
                    header = copy.copy(input_.header)
                    header.input_order = input_.input_order
                    header.set_default_values(input_.axes)
                    header.add_template(input_.header)
                    header.add_geometry(input_.header)

                # Here we could have compared the headers of
                # the arguments and resolved discrepancies.
                # The simplest resolution, however, is to take the
                # header of the first argument.
        return header

    def __getitem__(self, item):
        """__getitem__(self, item)

        Called to implement evaluation of self[item]. The
        accepted items should be integers and slice objects. Note that the
        special interpretation of negative indexes (if the class wishes to
        emulate a sequence type) is up to the __getitem__() method. If item is
        of an inappropriate type, TypeError may be raised; if of a value
        outside the set of indexes for the sequence (after any special
        interpretation of negative values), IndexError should be raised.
        Note: for loops expect that an IndexError will be
        raised for illegal indexes to allow proper detection of the end of the
        sequence.
        """

        def _set_geometry(_ret, _todo):
            # Ensure 'axes' is set first
            for i in range(len(_todo)):
                attr, value = _todo[i]
                if attr == 'axes':
                    setattr(_ret, attr, value)
                    _todo.pop(i)
                    break
            for attr, value in _todo:
                try:
                    setattr(_ret, attr, value)
                except (AttributeError, ValueError):
                    pass

        def _number_of_ellipses(_items):
            n = 0  # type: int
            for _item in _items:
                if _item is Ellipsis:
                    n += 1
            return n

        def _calculate_spec(obj, _items):
            _slicing = False
            _spec = {}
            if issubclass(type(obj), Series):
                # Calculate slice range
                try:
                    for _dim in range(obj.ndim):  # Loop over actual array shape
                        # Initial start,stop,step,axis
                        _spec[_dim] = (0, obj.shape[_dim], 1, obj.axes[_dim])
                except (AttributeError, NameError):
                    raise ValueError('No header in _calculate_spec')
                except IndexError:
                    # Probably an obj without axes property
                    return False, _spec

                # Determine how to loop over slice spec and items
                if _number_of_ellipses(_items) > 1:
                    raise IndexError('Multiple ellipses are not allowed.')

                # Assume Ellipsis anywhere in items
                index_spec = []
                index_item = []
                for _item in range(len(_items)):
                    if _items[_item] is not Ellipsis:
                        index_spec.append(_item)
                        index_item.append(_item)
                    else:  # Ellipsis
                        remaining_items = len(_items) - _item - 1
                        index_spec += range(len(_spec) - remaining_items, len(_spec))
                        index_item += range(len(_items) - remaining_items, len(_items))
                        break
                if len(index_spec) != len(index_item):
                    raise IndexError('Index problem: spec length %d, items length %d' %
                                     (len(index_spec), len(index_item)))

                for _dim, _item in zip(index_spec, index_item):
                    if isinstance(_items[_item], slice):
                        _start = _items[_item].start or _spec[_dim][0]
                        _stop = _items[_item].stop or _spec[_dim][1]
                        _step = _items[_item].step or _spec[_dim][2]
                        if _start < 0:
                            _start = obj.shape[_dim] + _start
                        if _stop < 0:
                            _stop = obj.shape[_dim] + _stop
                        _spec[_dim] = (_start, _stop, _step, obj.axes[_dim])
                        _slicing = True
                    elif issubclass(type(_items[_item]), str):
                        _spec[_dim] = (_items[_item], obj.axes[_dim])
                        _slicing = True
                    elif isinstance(_items[_item], (np.ndarray, Series)):
                        continue
                    elif isinstance(_items[_item], collections.abc.Sequence):
                        # The item is an iterable like tuple or list
                        if isinstance(_items[_item], tuple) and len(_items[_item]) == 1:
                            _it = int(_items[_item][0])
                            _start = _it or _spec[_dim][0]
                            if _start < 0:
                                _start = obj.shape[_dim] + _start
                            _stop = _start + 1
                            _step = 1
                            _spec[_dim] = (_start, _stop, _step, obj.axes[_dim])
                        else:
                            _slice = [_ for _ in _items[_item]]
                            _spec[_dim] = (_slice, obj.axes[_dim])
                        _slicing = True
                    elif np.issubdtype(type(_items[_item]), np.integer):
                        _it = int(_items[_item])
                        _start = _it or _spec[_dim][0]
                        if _start < 0:
                            _start = obj.shape[_dim] + _start
                        _stop = _start + 1
                        _step = 1
                        _spec[_dim] = (_start, _stop, _step, obj.axes[_dim])
                        _slicing = True
                    elif _items[_item] is None:
                        try:
                            _spec[_dim] = (None, None, None, obj.axes[_dim])
                        except IndexError:
                            _spec[_dim] = (None, None, None, None)
                        _slicing = True
                    else:
                        # The item is an iterable like tuple or list
                        _spec[_dim] = (_items[_item], obj.axes[_dim])
                        _slicing = True
            return _slicing, _spec

        if getattr(self, 'header', None) is None:
            return super(Series, self).__getitem__(item)
        if isinstance(item, tuple):
            items = item
        else:
            items = (item,)
        slicing, spec = _calculate_spec(self, items)

        todo = []  # Collect header changes, apply after ndarray slicing
        new_axes_names = []
        if slicing:
            # Here we slice the header information
            new_axes = []
            for i in range(len(spec)):
                # Slice along axis i
                try:
                    start, stop, step, axis = spec[i]
                    _slice = slice(start, stop, step)
                    if axis.name not in ['slice', 'row', 'column'] and len(axis.values[_slice]) <= 1:
                        continue
                except ValueError:
                    _slice, axis = spec[i]
                    if len(_slice) <= 1:
                        continue
                except AttributeError:
                    _slice = slice(0, 1, 0)
                    axis = UniformLengthAxis('unknown', 0, 1)

                new_axes.append(axis[_slice])
                new_axes_names.append(axis.name)

                if axis.name == 'slice':
                    # Select slice of imagePositions
                    sl = self.sliceLocations[_slice]
                    todo.append(('sliceLocations', sl))
                    try:
                        ipp = self.__get_imagePositions(spec[i])
                        todo.append(('imagePositions', None))  # Wipe existing positions
                        if ipp is not None:
                            todo.append(('imagePositions', ipp))
                    except KeyError:
                        pass
                elif axis.name in self.input_order.split(','):
                    # Select slice of tags
                    tags = self.__get_tags(spec)
                    todo.append(('tags', tags))
            Axes = namedtuple('Axes', new_axes_names)
            new_axes = Axes._make(new_axes)

        # Slicing the ndarray is done here
        ret = super(Series, self).__getitem__(item)
        # if slicing and issubclass(type(ret), Series):
        if issubclass(type(ret), Series):
            # noinspection PyUnboundLocalVariable
            if slicing:
                todo.append(('axes', new_axes))
                todo.append(('seriesInstanceUID', self.header.seriesInstanceUID))
            else:
                new_uid = ret.header.new_uid()
                ret.seriesInstanceUID = new_uid
            if ret.ndim < self.ndim:
                # Must copy the ret object before modifying. Otherwise, ret is a view to self.
                ret.header = copy.copy(ret.header)
                ret.input_order = INPUT_ORDER_NONE
                _names = []
                for _name in new_axes_names:
                    if _name not in ['slice', 'row', 'column']:
                        _names.append(_name)
                if len(_names) > 0:
                    ret.input_order = ','.join(_names)
            _set_geometry(ret, todo)
            ret.axes = ret.axes[-ret.ndim:]
        elif isinstance(ret, np.void):
            ret = tuple(ret)
        return ret

    def __get_sliceLocations(self, spec):
        try:
            sl = self.sliceLocations
        except ValueError:
            return None
        start, stop, step = 0, self.slices, 1
        if spec[0] is not None:
            start = spec[0]
        if spec[1] is not None:
            stop = spec[1]
        if spec[2] is not None:
            step = spec[2]
        sl = np.array(sl[start:stop:step])
        return sl

    def __get_imagePositions(self, spec):
        # logger.debug('__get_imagePositions: enter')
        try:
            ipp = self.imagePositions
        except ValueError:
            return None
        if len(spec) == 2:
            _values = {}
            for i, _idx in enumerate(spec[0]):
                _values[i] = ipp[_idx]
            return _values
        start, stop, step = 0, self.slices, 1
        if spec[0] is not None:
            start = spec[0]
        if spec[1] is not None:
            stop = spec[1]
        if spec[2] is not None:
            step = spec[2]
        ippdict = {}
        j = 0
        for i in range(start, stop, step):
            if i < 0:
                raise ValueError('i < 0')
            ippdict[j] = ipp[i]
            j += 1
        return ippdict

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        try:
            patientID = self.patientID
        except ValueError:
            patientID = ''
        try:
            patientName = self.patientName
        except ValueError:
            patientName = ''
        try:
            modality = self.getDicomAttribute('Modality')
        except ValueError:
            modality = ''
        try:
            seriesDescription = self.seriesDescription
        except ValueError:
            seriesDescription = ''
        try:
            seriesNumber = self.seriesNumber
        except ValueError:
            seriesNumber = 0
        return "Patient: {} {}\n".format(patientID, patientName) + \
            "  Study  Time: {} {}\n".format(
                self.getDicomAttribute('StudyDate'),
                self.getDicomAttribute('StudyTime')
                ) + \
            "  Series Time: {} {}\n".format(
                self.getDicomAttribute('SeriesDate'),
                self.getDicomAttribute('SeriesTime')
                ) + \
            "  Series #{} {}: {}\n".format(seriesNumber, modality, seriesDescription) + \
            "  Shape: {}, dtype: {}, input order: {}".format(
                shape_to_str(self.shape), self.dtype,
                self.input_order
                )

    @staticmethod
    def __find_tag_in_hdr(hdr_list, find_tag):
        for tag, filename, hdr in hdr_list:
            if tag == find_tag:
                return tag, filename, hdr
        return None

    def __get_tags(self, specs):
        tags, slices = self.header.get_tags_and_slices()
        try:
            tmpl_tags = self.tags
            tags = self.tags[0].shape
        except ValueError:
            return None
        slice_spec = [_ for _ in range(0, self.slices, 1)]
        tag_spec = tuple()
        for d in specs:
            try:
                start, stop, step, axis = specs[d]
                _indices = [_ for _ in range(start, stop, step)]
            except ValueError:
                start, stop, step = None, None, None
                _indices, axis = specs[d]
            if axis.name == "slice":
                slice_spec = _indices
            elif axis.name in self.input_order.split(','):
                if start is None:
                    tag_spec += (_indices,)
                else:
                    tag_spec += (slice(start, stop, step),)
        new_tags = {}
        for i, s in enumerate(slice_spec):
            new_tags[i] = self.tags[s][tag_spec]
        return new_tags

    def __calculate_window(self):
        if np.issubdtype(self.dtype, np.integer):
            _min_value = np.nanmin(self)
            _max_value = np.nanmax(self)
            _width = np.float32(_max_value) - np.float32(_min_value)
            _level = (np.float32(_min_value) + np.float32(_max_value)) / 2
            if abs(_width) > 2:
                _width = round(_width)
            if abs(_level) > 2:
                _level = round(_level)
        elif self.dtype == np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
            # RGB image
            _level = 127
            _width = 256
        elif self.dtype.fields is not None:
            # Structured dtype
            _min_value = np.inf
            _max_value = -np.inf
            for field in self.dtype.fields:
                _min_value = min(_min_value, np.float32(np.nanmin(self[field])))
                _max_value = max(_max_value, np.float32(np.nanmax(self[field])))
            _width = _max_value - _min_value
            _level = np.float32((_min_value + _max_value) / 2)
        else:
            _min_value = np.float32(np.nanmin(self))
            _max_value = np.float32(np.nanmax(self))
            _width = _max_value - _min_value
            _level = np.float32((_min_value + _max_value) / 2)
        return _level, _width

    def write(self, url, opts=None, formats=None, **kwargs):
        """Write Series image

        Args:
            url (str): Output URL.
            opts (argparse.Namespace or dict): Output options.
            formats (list or str): list of output formats, overriding opts.output_format.

        DICOMPlugin accept these opts:
            - "keep_uid": whether we will keep existing SOP Instance UID (bool).

            - "window": "original" will keep window center/level DICOM attributes,
              not recalculate from present data (str).

            - "output_sort": Which tag will sort the output images (int).
              Values: SORT_ON_SLICE, SORT_ON_TAG. Default: SORT_ON_SLICE.

            - "output_dir": Store all images in a single or multiple directories (str).
              Values: "single", "multi". Default: "single"

            - input_order: DICOM tag for given input_order (str).
        """
        _name: str = '{}.{}'.format(__name__, self.write.__name__)

        logger.debug('{}: url    : {}'.format(_name, url))
        logger.debug('{}: formats: {}'.format(_name, formats))
        logger.debug('{}: opts   : {}'.format(_name, opts))

        if opts is None:
            opts = {}
        elif issubclass(type(opts), argparse.Namespace):
            opts = vars(opts)
        for key, value in kwargs.items():
            opts[key] = value
        r_write(self, url, formats=formats, opts=opts)

    @property
    def input_order(self):
        """str: Input order

        How to sort input files:
            * INPUT_ORDER_NONE ('none'): No sorting.
            * INPUT_ORDER_TIME ('time'): Sort on image time (acquisition time or trigger time).
            * INPUT_ORDER_B ('b'): Sort on b value.
            * INPUT_ORDER_FA ('fa'): Sort on flip angle.
            * INPUT_ORDER_TE ('te'): Sort on echo time.
            * INPUT_ORDER_FAULTY ('faulty'): Correct erroneous attributes.

        Raises:
            ValueError: when order is illegal.
        """
        return self.header.input_order

    @input_order.setter
    def input_order(self, order):
        for component in order.split(sep=','):
            if component not in input_order_set:
                raise ValueError("Unknown input order: {}".format(order))
        self.header.input_order = order

    @property
    def input_format(self):
        """str: Input format

        Possible input formats depend on the available `formats` plugins,
        and include `'dicom'`, `'itk'` and `'nifti'`.
        """
        return self.header.input_format

    @input_format.setter
    def input_format(self, fmt):
        self.header.input_format = fmt

    @property
    def input_sort(self):
        """int: Input order

        How to sort output files:
            * SORT_ON_SLICE: Run over slices first
            * SORT_ON_TAG  : Run over input order first, then slices

        Raises:
            ValueError: when input order is not defined.
        """
        try:
            if self.header.input_sort is not None:
                return self.header.input_sort
        except AttributeError:
            pass
        raise ValueError("Input sort order not set.")

    @input_sort.setter
    def input_sort(self, order):
        if order is None or order in sort_on_set:
            self.header.input_sort = order
        else:
            raise ValueError("Unknown sort order: {}".format(order))

    @property
    def sort_on(self):
        """int: Output order

        How to sort output files:
            * SORT_ON_SLICE: Run over slices first
            * SORT_ON_TAG  : Run over input order first, then slices

        Raises:
            ValueError: when output order is not defined.
        """
        try:
            if self.header.sort_on is not None:
                return self.header.sort_on
        except AttributeError:
            pass
        raise ValueError("Output sort order not set.")

    @sort_on.setter
    def sort_on(self, order):
        if order in sort_on_set:
            self.header.sort_on = order
        else:
            raise ValueError("Unknown sort order: {}".format(order))

    @property
    def shape(self):
        """tuple of ints: Matrix shape.

        Raises:
            IndexError: always when set (do not set shape). Should set axes instead.
        """
        return super(Series, self).shape

    @shape.setter
    def shape(self, s):
        raise IndexError('Should set axes instead of shape.')

    @property
    def rows(self):
        """int: Number of rows.

        Raises:
            ValueError: when number of rows is not defined.
        """
        try:
            return len(self.axes.row)
        except AttributeError:
            if self.ndim < 2:
                raise ValueError("{}D dataset has no rows".format(self.ndim))
            return self.shape[-2]

    @property
    def columns(self):
        """int: Number of columns.

        Raises:
            ValueError: when number of columns is not defined.
        """
        try:
            return len(self.axes.column)
        except AttributeError:
            if self.ndim < 1:
                raise ValueError("Dataset has no columns")
            return self.shape[-1]

    @property
    def slices(self):
        """int: Number of slices.

        Raises:
            ValueError: when number of slices is not defined.
            DoNotSetSlicesError: Always (do not set slices)
        """
        try:
            return len(self.axes.slice)
        except AttributeError:
            if self.ndim < 3:
                # logger.debug("Series.slices: {}D dataset has no slices".format(self.ndim))
                # raise ValueError("{}D dataset has no slices".format(self.ndim))
                return 1
            # logger.debug("Series.slices: {}D dataset slice from shape ({}) {}".format(
            #     self.ndim, self.shape, self.shape[-3 - _color]))
            return self.shape[-3]

    @slices.setter
    def slices(self, nslices):
        raise DoNotSetSlicesError('Do not set slices=%d explicitly. Slices are inferred '
                                  'from the shape.' % nslices)

    @property
    def sliceLocations(self):
        """numpy.ndarray: Slice locations.

        Sorted numpy array of slice locations, in mm.

        Raises:
            ValueError: When no slice locations are defined.
        """
        _name: str = '{}.{}'.format(__name__, 'sliceLocations')

        if self.header.sliceLocations is not None:
            return self.header.sliceLocations
        try:
            # Some image formats do not provide slice locations.
            # If orientation and imagePositions are set, slice locations can
            # be calculated.
            if self.header.orientation is not None and self.header.imagePositions is not None:
                logger.debug('{}: '
                             'sliceLocations: calculate {} slice from orientation and '
                             'imagePositions'.format(_name, self.slices))
                loc = np.empty(self.slices)
                normal = self.transformationMatrix[0, :3]
                for _slice in range(self.slices):
                    loc[_slice] = np.inner(normal, self.imagePositions[_slice].flatten())
                self.header.sliceLocations = loc
                return self.header.sliceLocations
        except AttributeError:
            pass
        except Exception:
            pass
        raise ValueError("Slice locations are not defined.")

    @sliceLocations.setter
    def sliceLocations(self, loc):

        # def _is_uniform_spacing(loc):
        #     # sort slice locations
        #     _locations = {}
        #     _location0 = loc[0]
        #     for _location in loc[1:]:
        #         _locations[_location - _location0] = True
        #         _location0 = _location
        #     return len(_locations) == 1

        if loc is None or len(loc) < 1:
            raise ValueError('Cannot set slice locations to empty list')
        if len(loc) != self.slices:
            raise ValueError('Cannot set {} slice locations for {} slices'.format(
                len(loc), self.slices
            ))
        self.header.sliceLocations = loc
        return
        # if _is_uniform_spacing(loc):
        #     if len(loc) > 1:
        #         ds = loc[1] - loc[0]
        #     else:
        #         ds = self.spacing[0]
        #     _axis = UniformLengthAxis('slice', loc[0], len(loc), ds)
        # else:
        #     _axis = VariableAxis('slice', loc)
        # for i, axis in enumerate(self.axes):
        #     if axis.name == 'slice':
        #         # Replace axis
        #         self.axes[i] = _axis
        #         return

    def get_slice_axis(self):
        """Get the slice axis instance.
        """
        try:
            return self.axes.slice
        except AttributeError:
            return None

    def get_tag_axis(self):
        """Get the tax axis instance.
        """
        try:
            return getattr(self.axes, self.input_order)
        except AttributeError:
            return None

    @property
    def tags(self):
        """dict[slice] of numpy.ndarray(tags): Image tags for each slice

        Image tags can be an array of:
            - time points
            - diffusion weightings (b values)
            - flip angles

        Image tags will be tuples when image dimension > 4.

        tags is a dict with (slice keys, tag array)

        dict[slice] is np.ndarray(tags)

        Examples:

            >>>self.tags[slice][tag]

        Raises:
            ValueError: when tags are not set.
        """
        try:
            if self.header.tags is not None:
                return self.header.tags
        except AttributeError:
            pass
        return None
        if self.axes[0].name in ('slice', 'row', 'column'):
            return {0: np.array(0)}
        try:
            values = self.axes[0].values
            tags = {}
            for _slice in range(self.slices):
                tags[_slice] = np.array(values)
            return tags
        except AttributeError:
            pass
        return None

    @tags.setter
    def tags(self, tags):
        self.header.tags = {}
        # hdr = {}
        for s in tags.keys():
            self.header.tags[s] = np.array(tags[s])

    @property
    def axes(self):
        """list of Axis: axes objects, sorted like shape indices.

        Raises:
            ValueError: when the axes are not set.
        """
        try:
            if self.header.axes is not None:
                return self.header.axes
        except AttributeError:
            pass
        # Calculate axes from image shape
        shape = super(Series, self).shape
        if len(shape) < 1:
            return None
        _max_known_shape = min(3, len(shape))
        _labels = self.input_order.split(sep=',')
        _labels += ['slice', 'row', 'column'][-_max_known_shape:]
        _labels = _labels[-_max_known_shape:]
        i = 0
        while len(_labels) < self.ndim:
            _labels.insert(0, 'unknown{}'.format(i))
            i += 1

        _axes = []
        for i, d in enumerate(super(Series, self).shape):
            _axes.append(
                UniformLengthAxis(
                    _labels[i], 0, d, 1
                )
            )
        Axes = namedtuple('Axes', _labels)
        self.header.axes = Axes._make(_axes)
        return self.header.axes

    @axes.setter
    def axes(self, ax):
        # Verify that axes shape match ndarray shape
        # Verify that axis names are used once only
        used_name = {}
        field_names = []
        for axis in ax:
            # if len(axis) != self.shape[i]:
            #     raise IndexError("Axis length {}  must match array shape {}".format(
            #         [len(x) for x in ax], self.shape))
            if axis.name in used_name:
                raise ValueError("Axis name {} is used multiple times.".format(axis.name))
            used_name[axis.name] = True
            field_names.append(axis.name)
        Axes = namedtuple('Axes', field_names)
        self.header.axes = Axes._make(ax)

        # Update spacing from new axes
        try:
            spacing = self.spacing
        except ValueError:
            spacing = np.array((1.0, 1.0, 1.0))
        for i, direction in enumerate(['slice', 'row', 'column']):
            try:
                axis = getattr(self.axes, direction)
                spacing[i] = axis.step
            except AttributeError:
                pass
        self.header.spacing = spacing

    @property
    def spacing(self):
        """numpy.array([ds,dr,dc]): spacing in mm.

        Given as ds,dr,dc in mm.
        2D image will return ds=1.

        Raises:
            ValueError: when spacing is not set.
            ValueError: when spacing is not a tuple of 3 coordinates

        Usage:
            >>> si = Series(np.eye(128))
            >>> ds, dr, dc = si.spacing
            >>> si.spacing = ds, dr, dc
        """
        try:
            # if self.header.spacing is not None:
            #    return self.header.spacing
            slice_axis = self.axes.slice
            ds = slice_axis.step
        except (AttributeError, ValueError) as e:
            if self.header.spacing is not None:
                ds = self.header.spacing[0]
            else:
                raise ValueError("Spacing is unknown: {}".format(e))
        try:
            return np.array((ds, self.axes.row.step, self.axes.column.step))
        except AttributeError as e:
            raise ValueError("Spacing is unknown: {}".format(e))

    @spacing.setter
    def spacing(self, *args):
        _name: str = '{}.{}'.format(__name__, 'spacing')

        if args[0] is None:
            return
        logger.debug("{}: {} {}".format(_name, len(args), args))
        for arg in args:
            logger.debug("{}: arg {} {}".format(_name, len(arg), arg))
        # Invalidate existing transformation matrix
        self.header.transformationMatrix = None
        # Handle both tuple and component spacings
        if len(args) == 3:
            spacing = np.array(args)
        elif len(args) == 1:
            arg = args[0]
            if len(arg) == 3:
                spacing = np.array(arg)
            elif len(arg) == 1:
                arg0 = arg[0]
                if len(arg0) == 3:
                    spacing = np.array(arg0)
                else:
                    raise ValueError("Length of spacing in setSpacing(): %d" % len(arg0))
            else:
                raise ValueError("Length of spacing in setSpacing(): %d" % len(arg))
        else:
            raise ValueError("Length of spacing in setSpacing(): %d" % len(args))
        try:
            self.axes.slice.step = spacing[0]
        except (AttributeError, ValueError):
            # Assume 2D image with no slice dimension
            pass
        try:
            self.axes.row.step = spacing[1]
            self.axes.column.step = spacing[2]
            self.header.spacing = np.array(spacing)
        except ValueError as e:
            raise ValueError("Spacing cannot be set: {}".format(e))

    @property
    def imagePositions(self):
        """dict of numpy.array([z,y,x]): The [z,y,x] coordinates of the upper left
        hand corner (first pixel) of each slice.

        dict(imagePositions[s]) of [z,y,x] in mm, as numpy array

        When setting, the position list is added to existing imagePositions.
        Overlapping dict keys will replace exisiting imagePosition for given slice s.

        Examples:
            >>> from imagedata import Series
            >>> import numpy as np
            >>> si = Series(np.eye(128))
            >>> z,y,x = si.imagePositions[0]

        Examples:
            >>> from imagedata import Series
            >>> import numpy as np
            >>> si = Series(np.zeros((16, 128, 128)))
            >>> for s in range(si.slices):
            ...     si.imagePositions = { s: si.getPositionForVoxel(np.array([s, 0, 0])) }

        Raises:
            ValueError: when imagePositions are not set.
            AssertionError: when positions have wrong shape or datatype.
        """
        _name: str = '{}.{}'.format(__name__, 'imagePositions')

        try:
            if self.header.imagePositions is not None:
                if len(self.header.imagePositions) > self.slices:
                    # Truncate imagePositions to actual number of slices.
                    # Could be the result of a slicing operation.
                    ipp = {}
                    for z in range(self.slices):
                        ipp[z] = \
                            self.header.imagePositions[z]
                    self.header.imagePositions = ipp
                elif len(self.header.imagePositions) < self.slices and \
                        len(self.header.imagePositions) == 1:
                    # Some image formats define one imagePosition only.
                    # Could calculate the missing imagePositions from origin and
                    # orientation.
                    # Set imagePositions for additional slices
                    logger.debug('{}: '
                                 'Series.imagePositions.get: 1 positions only.  Calculating the other {} '
                                 'positions'.format(_name, self.slices - 1))
                    m = self.transformationMatrix
                    for _slice in range(1, self.slices):
                        self.header.imagePositions[_slice] = \
                            self.getPositionForVoxel(np.array([_slice, 0, 0]),
                                                     transformation=m)
                return self.header.imagePositions
        except AttributeError:
            pass
        raise ValueError("No imagePositions set.")

    @imagePositions.setter
    def imagePositions(self, poslist):
        if poslist is None:
            self.header.imagePositions = None
            return
        assert isinstance(poslist, dict), "ImagePositions is not dict() (%s)" % type(poslist)
        # Invalidate existing transformation matrix
        # self.header.transformationMatrix = None
        try:
            if self.header.imagePositions is None:
                self.header.imagePositions = dict()
        except AttributeError:
            self.header.imagePositions = dict()
        # logger.debug("imagePositions set for keys {}".format(poslist.keys()))
        for _slice in poslist.keys():
            pos = poslist[_slice]
            # logger.debug("imagePositions set _slice {} to {}".format(_slice,pos))
            assert isinstance(pos, np.ndarray), "Wrong datatype of position (%s)" % type(pos)
            assert len(pos) == 3, "Wrong size of pos (is %d, should be 3)" % len(pos)
            self.header.imagePositions[_slice] = np.array(pos)

    @property
    def orientation(self):
        """numpy.array: The direction cosines of the first row and the first column with
        respect to the patient.

        These attributes shall be provided as a pair.
        Row value (column index) for the z,y,x axes respectively,
        followed by the column value (row index) for the z,y,x axes respectively.

        Raises:
            ValueError: when orientation is not set.
            AssertionError: when len(orient) != 6
        """
        try:
            if self.header.orientation is not None:
                return self.header.orientation
        except AttributeError:
            pass
        raise ValueError("No orientation set.")

    @orientation.setter
    def orientation(self, orient):
        if orient is None:
            self.header.transformationMatrix = None
            self.header.orientation = None
            return
        assert len(orient) == 6, "Wrong size of orientation"
        # Invalidate existing transformation matrix
        # self.header.transformationMatrix = None
        self.header.orientation = np.array(orient)

    @property
    def dicomTemplate(self):
        """pydicom.dataset.Dataset: DICOM data set

        Raises:
        ValueError: when modality is not set.
        ValueError: when modality cannot be converted to str.
        """
        try:
            if self.header.dicomTemplate is not None:
                return self.header.dicomTemplate
        except AttributeError:
            pass
        raise ValueError("No DICOM template set.")

    @dicomTemplate.setter
    def dicomTemplate(self, ds):
        if ds is None:
            self.header.dicomTemplate = None
            return
        if issubclass(type(ds), pydicom.dataset.Dataset):
            self.header.dicomTemplate = str(ds)
        else:
            raise ValueError("Dataset is not a pydicom.dataset.Dataset.")

    @property
    def modality(self):
        """str: Imaging modality.

        Raises:
            ValueError: when modality is not set.
            ValueError: when modality cannot be converted to str.
        """
        try:
            if self.header.modality is not None:
                return self.header.modality
        except AttributeError:
            pass
        raise ValueError("No modality set.")

    @modality.setter
    def modality(self, mod):
        if mod is None:
            self.header.modality = None
            return
        try:
            self.header.modality = str(mod)
        except AttributeError:
            raise ValueError("Cannot convert modality to string")

    @property
    def laterality(self):
        """str: Imaging laterality.

        Raises:
            ValueError: when laterality is not set.
            ValueError: when laterality cannot be converted to str.
        """
        try:
            if self.header.laterality is not None:
                return self.header.laterality
        except AttributeError:
            pass
        raise ValueError("No laterality set.")

    @laterality.setter
    def laterality(self, lat):
        if lat is None:
            self.header.laterality = None
            return
        try:
            self.header.laterality = str(lat)
        except AttributeError:
            raise ValueError("Cannot convert laterality to string")

    @property
    def bodyPartExamined(self):
        """str: Body Part Examined.

        Raises:
            ValueError: when Body Part Examined is not set.
            ValueError: when Body Part Examined cannot be converted to str.
        """
        try:
            if self.header.bodyPartExamined is not None:
                return self.header.bodyPartExamined
        except AttributeError:
            pass
        raise ValueError("No Body Part Examined set.")

    @bodyPartExamined.setter
    def bodyPartExamined(self, part):
        if part is None:
            self.header.bodyPartExamined = None
            return
        try:
            self.header.bodyPartExamined = str(part)
        except AttributeError:
            raise ValueError("Cannot convert Body Part Examined to string")

    @property
    def patientPosition(self):
        """str: Patient Position.

        Raises:
            ValueError: when Patient Position is not set.
            ValueError: when Patient Position cannot be converted to str.
        """
        try:
            if self.header.patientPosition is not None:
                return self.header.patientPosition
        except AttributeError:
            pass
        raise ValueError("No Patient Position set.")

    @patientPosition.setter
    def patientPosition(self, pos):
        if pos is None:
            self.header.patientPosition = None
            return
        try:
            self.header.patientPosition = str(pos)
        except AttributeError:
            raise ValueError("Cannot convert Patient Position to string")

    @property
    def protocolName(self):
        """str: Imaging Protocol Name.

        Raises:
            ValueError: when protocolName is not set.
            ValueError: when protocolName cannot be converted to str.
        """
        try:
            if self.header.protocolName is not None:
                return self.header.protocolName
        except AttributeError:
            pass
        raise ValueError("No protocolName set.")

    @protocolName.setter
    def protocolName(self, protocol):
        if protocol is None:
            self.header.protocolName = None
            return
        try:
            self.header.protocolName = str(protocol)
        except AttributeError:
            raise ValueError("Cannot convert Protocol Name to string")

    @property
    def seriesDate(self):
        """str: Imaging Series Date.

        Raises:
            ValueError: when seriesDate is not set.
            ValueError: when seriesDate cannot be converted to str.
        """
        try:
            if self.header.seriesDate is not None:
                return self.header.seriesDate
        except AttributeError:
            pass
        raise ValueError("No Series Date set.")

    @seriesDate.setter
    def seriesDate(self, date):
        if date is None:
            self.header.seriesDate = None
            return
        try:
            self.header.seriesDate = str(date)
        except AttributeError:
            raise ValueError("Cannot convert Series Date to string")

    @property
    def seriesTime(self):
        """str: Imaging Series Time.

        Raises:
            ValueError: when seriesTime is not set.
            ValueError: when seriesTime cannot be converted to str.
        """
        try:
            if self.header.seriesTime is not None:
                return self.header.seriesTime
        except AttributeError:
            pass
        raise ValueError("No Series Time set.")

    @seriesTime.setter
    def seriesTime(self, time):
        if time is None:
            self.header.seriesTime = None
            return
        try:
            self.header.seriesTime = str(time)
        except AttributeError:
            raise ValueError("Cannot convert Series Time to string")

    @property
    def seriesNumber(self):
        """int: DICOM series number.

        Raises:
            ValueError: when series number is not set.
            ValueError: when series number cannot be converted to int.
        """
        try:
            if self.header.seriesNumber is not None:
                return self.header.seriesNumber
        except AttributeError:
            pass
        raise ValueError("No series number set.")

    @seriesNumber.setter
    def seriesNumber(self, sernum):
        if sernum is None:
            self.header.seriesNumber = None
            return
        try:
            self.header.seriesNumber = int(sernum)
        except AttributeError:
            raise ValueError("Cannot convert series number to integer")

    @property
    def seriesDescription(self):
        """str: DICOM series description.
        """
        try:
            if self.header.seriesDescription is not None:
                return self.header.seriesDescription
        except AttributeError:
            pass
        return ''

    @seriesDescription.setter
    def seriesDescription(self, descr):
        if descr is None:
            self.header.seriesDescription = None
            return
        assert isinstance(descr, str), "Given series description is not str"
        self.header.seriesDescription = descr

    @property
    def imageType(self):
        """list of str: DICOM image type(s).

        Raises:
            ValueError: when image type is not set.
            TypeError: When imagetype is not printable.
        """
        try:
            if self.header.imageType is not None:
                return self.header.imageType
        except AttributeError:
            pass
        raise ValueError("No image type set.")

    @imageType.setter
    def imageType(self, imagetype):
        if imagetype is None:
            self.header.imageType = None
            return
        self.header.imageType = list()
        try:
            for s in imagetype:
                self.header.imageType.append(str(s))
        except AttributeError:
            raise TypeError("Given image type is not printable (is %s)" % type(imagetype))

    @property
    def studyInstanceUID(self):
        """str: DICOM Study instance UID

        Raises:
            ValueError: when study instance UID is not set.
            TypeError: When uid is not printable.
        """
        try:
            if self.header.studyInstanceUID is not None:
                return self.header.studyInstanceUID
        except AttributeError:
            pass
        raise ValueError("No study instance UID set.")

    @studyInstanceUID.setter
    def studyInstanceUID(self, uid):
        if uid is None:
            self.header.studyInstanceUID = None
            return
        try:
            self.header.studyInstanceUID = str(uid)
        except AttributeError:
            raise TypeError("Given study instance UID is not printable")

    @property
    def studyID(self):
        """str: DICOM study ID

        Raises:
            ValueError: when study ID is not set.
            TypeError: When id is not printable
        """
        try:
            if self.header.studyID is not None:
                return self.header.studyID
        except AttributeError:
            pass
        raise ValueError("No study ID set.")

    @studyID.setter
    def studyID(self, id):
        if id is None:
            self.header.studyID = None
            return
        try:
            self.header.studyID = str(id)
        except AttributeError:
            raise TypeError("Given study ID is not printable")

    @property
    def seriesInstanceUID(self):
        """str: DICOM series instance UID

        Raises:
            ValueError: when series instance UID is not set
            TypeError: When uid is not printable
        """
        try:
            if self.header.seriesInstanceUID is not None:
                return self.header.seriesInstanceUID
        except AttributeError:
            pass
        raise ValueError("No series instance UID set.")

    @seriesInstanceUID.setter
    def seriesInstanceUID(self, uid):
        if uid is None:
            self.header.seriesInstanceUID = None
            return
        try:
            if isinstance(uid, str):
                self.header.seriesInstanceUID = uid
            else:
                self.header.seriesInstanceUID = str(uid)
        except AttributeError:
            raise TypeError("Given series instance UID is not printable")

    @property
    def frameOfReferenceUID(self):
        """str: DICOM frame of reference UID

        Raises:
            ValueError: when frame of reference UID is not set
            TypeError: When uid is not printable
        """
        try:
            if self.header.frameOfReferenceUID is not None:
                return self.header.frameOfReferenceUID
        except AttributeError:
            pass
        raise ValueError("No frame of reference UID set.")

    @frameOfReferenceUID.setter
    def frameOfReferenceUID(self, uid):
        if uid is None:
            self.header.frameOfReferenceUID = None
            return
        try:
            self.header.frameOfReferenceUID = str(uid)
        except AttributeError:
            raise TypeError("Given frame of reference UID is not printable")

    @property
    def SOPClassUID(self):
        """str: DICOM SOP Class UID

        Raises:
            ValueError: when SOP Class UID is not set
        """
        try:
            if self.header.SOPClassUID is not None:
                return self.header.SOPClassUID
        except AttributeError:
            pass
        raise ValueError("No SOP Class UID set.")

    @SOPClassUID.setter
    def SOPClassUID(self, uid):
        if uid is None:
            self.header.SOPClassUID = None
            return
        try:
            self.header.SOPClassUID = get_uid_for_storage_class(uid)
        except ValueError:
            raise

    @property
    def SOPInstanceUIDs(self):
        """str: DICOM SOP Instance UIDs

        Raises:
            ValueError: when SOP Instance UIDs is not set
        """
        try:
            if self.header.SOPInstanceUIDs is not None:
                return self.header.SOPInstanceUIDs
        except AttributeError:
            pass
        raise ValueError("No SOP Instance UIDs set.")

    @SOPInstanceUIDs.setter
    def SOPInstanceUIDs(self, uids):
        self.header.SOPInstanceUIDs = uids

    @property
    def accessionNumber(self):
        """str: DICOM accession number

        Raises:
            ValueError: when accession number is not set
            TypeError: When accno is not printable
        """
        try:
            if self.header.accessionNumber is not None:
                return self.header.accessionNumber
        except AttributeError:
            pass
        raise ValueError("No accession number set.")

    @accessionNumber.setter
    def accessionNumber(self, accno):
        if accno is None:
            self.header.accessionNumber = None
            return
        try:
            self.header.accessionNumber = str(accno)
        except AttributeError:
            raise TypeError("Given accession number is not printable")

    @property
    def patientName(self):
        """str: Patient name

        Raises:
            ValueError: when patient name is not set
            TypeError: When patnam is not printable
        """
        try:
            if self.header.patientName is not None:
                return self.header.patientName
        except AttributeError:
            pass
        raise ValueError("No patient name set.")

    @patientName.setter
    def patientName(self, patnam):
        if patnam is None:
            self.header.patientName = None
            return
        try:
            self.header.patientName = str(patnam)
        except AttributeError:
            raise TypeError("Given patient name is not printable")

    @property
    def patientID(self):
        """str: Patient ID

        Raises:
            ValueError: when patient ID is not set
            TypeError: When patID is not printable
        """
        try:
            if self.header.patientID is not None:
                return self.header.patientID
        except AttributeError:
            pass
        raise ValueError("No patient ID set.")

    @patientID.setter
    def patientID(self, patid):
        if patid is None:
            self.header.patientID = None
            return
        try:
            self.header.patientID = str(patid)
        except AttributeError:
            raise TypeError("Given patient ID is not printable")

    @property
    def patientBirthDate(self):
        """str: Patient birth date

        Raises:
            ValueError: when patient birth date is not set.
            TypeError: When patient birth date is not printable.
        """
        try:
            if self.header.patientBirthDate is not None:
                return self.header.patientBirthDate
        except AttributeError:
            pass
        raise ValueError("No patient birthdate set.")

    @patientBirthDate.setter
    def patientBirthDate(self, patbirdat):
        if patbirdat is None:
            self.header.patientBirthDate = None
            return
        try:
            self.header.patientBirthDate = str(patbirdat)
        except AttributeError:
            raise TypeError("Given patient birth date is not printable")

    @property
    def color(self):
        """bool: Color interpretation.

        Whether the array stores a color image, and the
        last index represents the color components

        Raises:
            ValueError: when color interpretation is not set
            TypeError: When color is not bool
        """
        return self.dtype == np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        # return self.dtype == rgb_dtype
        # try:
        #     if self.header.axes is not None and len(self.header.axes):
        #         return self.header.axes[-1].name == 'rgb'
        # except AttributeError:
        #     pass
        # raise ValueError("No Color Interpretation is set.")

    @color.setter
    def color(self, color):
        raise ValueError("Do not set color. Set dtype.")

    @property
    def colormap(self):
        try:
            if self.header.colormap is not None:
                return self.header.colormap
        except AttributeError:
            pass
        return None

    @colormap.setter
    def colormap(self, map):
        if map is None:
            self.header.colormap = None
            return
        try:
            self.header.colormap = map
        except AttributeError:
            raise TypeError("Given colormap is not usable")

    @property
    def colormap_norm(self):
        try:
            if self.header.colormap_norm is not None:
                return self.header.colormap_norm
        except AttributeError:
            pass
        return None

    @colormap_norm.setter
    def colormap_norm(self, norm):
        if norm is None:
            self.header.colormap_norm = None
            return
        try:
            self.header.colormap_norm = norm
        except AttributeError:
            raise TypeError("Given colormap_norm is not usable")

    @property
    def colormap_label(self):
        """str: Colormap label

        Raises:
            ValueError: when colormap label is not set
            TypeError: When colormap label is not printable
        """
        try:
            if self.header.colormap_label is not None:
                return self.header.colormap_label
        except AttributeError:
            pass
        raise ValueError("No Colormap Label is set.")

    @colormap_label.setter
    def colormap_label(self, string):
        if string is None:
            self.header.colormap_label = None
            return
        try:
            self.header.colormap_label = str(string)
        except AttributeError:
            raise TypeError("Given colormap_label is not printable")

    @property
    def photometricInterpretation(self):
        """str: Photometric Interpretation.

        Raises:
            ValueError: when photometric interpretation is not set
            TypeError: When photometric interpretation is not printable
        """
        try:
            if self.header.photometricInterpretation is not None:
                return self.header.photometricInterpretation
        except AttributeError:
            pass
        raise ValueError("No Photometric Interpretation is set.")

    @photometricInterpretation.setter
    def photometricInterpretation(self, string):
        if string is None:
            self.header.photometricInterpretation = None
            return
        try:
            self.header.photometricInterpretation = str(string)
        except AttributeError:
            raise TypeError("Given photometric interpretation is not printable")

    @property
    def windowCenter(self):
        """number: Window Center

        Raises:
            ValueError: when window center is not set
            TypeError: When window center is not a number
        """
        try:
            if self.header.windowCenter is None:
                self.header.windowCenter, self.header.windowWidth = self.__calculate_window()
                # level = (np.float32(np.nanmax(self)) + np.float32(np.nanmin(self))) / 2
                # if np.isnan(level):
                #     level = 1
                # if abs(level) > 2:
                #     level = round(level)
                # self.header.windowCenter = level
            return self.header.windowCenter
        except AttributeError:
            pass
        raise ValueError("No Window Center is set.")

    @windowCenter.setter
    def windowCenter(self, value):
        if value is None:
            try:
                self.header.windowCenter = None
            except AttributeError:
                pass
            return
        if isinstance(value, numbers.Number):
            try:
                self.header.windowCenter = value
            except AttributeError:
                pass
        else:
            raise TypeError("Given window center is not a number.")

    @property
    def windowWidth(self):
        """number: Window Width

        Raises:
            ValueError: when window width is not set
            TypeError: When window width is not a number
        """
        try:
            if self.header.windowWidth is None:
                self.header.windowCenter, self.header.windowWidth = self.__calculate_window()
                # window = np.float32(np.nanmax(self)) - np.float32(np.nanmin(self))
                # if np.isnan(window):
                #     window = 1
                # if abs(window) > 2:
                #     window = round(window)
                # self.header.windowWidth = window
            return self.header.windowWidth
        except AttributeError:
            pass
        raise ValueError("No Window Width is set.")

    @windowWidth.setter
    def windowWidth(self, value):
        if value is None:
            try:
                self.header.windowWidth = None
            except AttributeError:
                pass
            return
        if isinstance(value, numbers.Number):
            try:
                self.header.windowWidth = value
            except AttributeError:
                pass
        else:
            raise TypeError("Given window width is not a number.")

    # noinspection PyPep8Naming
    @property
    def transformationMatrix(self):
        """numpy.array: Transformation matrix.

        If the transformation matrix is not set, an attempt will be made to calculate it
        from spacing, imagePositions and orientation.

        When setting the transformation matrix, spacing and slices must be set in advance.
        A new transformation matrix will also impact orientation and  imagePositions.

        Raises:
            ValueError: Transformation matrix cannot be constructed.
        """

        # def normalize(v):
        #     """Normalize a vector
        #
        #     https://stackoverflow.com/questions/21030391/how-to-normalize-an-array-in-numpy
        #
        #     :param v: 3D vector
        #     :return: normalized 3D vector
        #     """
        #     norm=np.linalg.norm(v, ord=1)
        #     if norm==0:
        #         norm=np.finfo(v.dtype).eps
        #     return v/norm

        _name: str = '{}.{}'.format(__name__, 'transformationMatrix')
        debug = None
        # debug = True

        try:
            if self.header.transformationMatrix is not None:
                logger.debug('{}: return existing matrix'.format(_name))
                return self.header.transformationMatrix

            # Calculate transformation matrix
            logger.debug('{}: Calculate transformation matrix'.format(_name))
            logger.debug('{}: self {} {}'.format(_name, self.dtype, self.shape))
            ds, dr, dc = self.spacing
            logger.debug('{}: ds {}, dr {}, dc {}'.format(_name, ds, dr, dc))
            slices = len(self.header.imagePositions)
            logger.debug('{}: slices {}'.format(_name, slices))
            T0 = self.header.imagePositions[0].reshape(3, 1)  # z,y,x
            Tn = self.header.imagePositions[slices - 1].reshape(3, 1)
            orient = self.orientation

            colr = np.array(orient[3:]).reshape(3, 1)
            colc = np.array(orient[:3]).reshape(3, 1)
            if slices > 1:
                logger.debug('{}: multiple slices case (slices={})'
                             ''.format(_name, slices))
                # Calculating normal vector based on first and last slice should be
                # the correct method.
                k = (T0 - Tn) / (1 - slices)
                # Will just calculate normal to row and column to match other software.
                # k = np.cross(colr, colc, axis=0) * ds
            else:
                logger.debug('{}: single slice case'.format(_name))
                k = np.cross(colr, colc, axis=0)
                # k = normalize(k) * ds
                k = k * ds
            logger.debug('{}: k={}'.format(_name, k.T))
            # logger.debug("q: k {} colc {} colr {} T0 {}".format(k.shape,
            #    colc.shape, colr.shape, T0.shape))
            A = np.eye(4)
            A[:3, :4] = np.hstack([
                k,
                colr * dr,
                colc * dc,
                T0])
            if debug:
                logger.debug("{}: A:\n{}".format(_name, A))
            self.header.transformationMatrix = A
            return self.header.transformationMatrix
        except AttributeError:
            logger.debug('{}: AttributeError'.format(_name))
            pass
        raise ValueError('Transformation matrix cannot be constructed.')

    @transformationMatrix.setter
    def transformationMatrix(self, m):
        self.header.transformationMatrix = m
        # if M is not None:
        #    ds,dr,dc = self.spacing
        #    # Set imagePositions for first slice
        #    z,y,x = M[0:3,3]
        #    self.imagePositions = ({0: np.array([z,y,x])})
        #    # Set slice orientation
        #    orient = []
        #    orient.append(M[2,2]/dr)
        #    orient.append(M[1,2]/dr)
        #    orient.append(M[0,2]/dr)
        #    orient.append(M[2,1]/dc)
        #    orient.append(M[1,1]/dc)
        #    orient.append(M[0,1]/dc)
        #    self.orientation = orient
        #    # Set imagePositions for additional slices
        #    for _slice in range(1,self.slices):
        #        self.imagePositions = {
        #            _slice: self.getPositionForVoxel(np.array([_slice,0,0]),
        #                transformation=M)
        #        }

    def get_transformation_components_xyz(self):
        """Get origin and direction from transformation matrix in xyz convention.

        Returns:
            tuple: Tuple of
                - Origin (np.array size 3): Origin.
                - Orientation (np.array size 6): (row, then column directional cosines)
                  (DICOM convention).
                - Normal vector (np.array size 3): Slice direction.
        """
        m = self.transformationMatrix
        ds, dr, dc = self.spacing

        # origin
        try:
            ipp = self.imagePositions
            if len(ipp) > 0:
                ipp = ipp[0]
            else:
                ipp = np.array([0, 0, 0])
        except ValueError:
            ipp = np.array([0, 0, 0])
        if ipp.shape == (3, 1):
            ipp.shape = (3,)
        z, y, x = ipp[:]
        origin = np.array([x, y, z])

        # orientation
        # Reverse orientation vectors from zyx to xyz
        try:
            orientation = [
                self.orientation[2], self.orientation[1], self.orientation[0],
                self.orientation[5], self.orientation[4], self.orientation[3]]
        except ValueError:
            orientation = [1, 0, 0, 0, 1, 0]

        n = m[:3, 0][::-1].reshape(3)
        if self.slices == 1:
            n = n / ds

        return origin, np.array(orientation), n

    @property
    def timeline(self):
        """numpy.array: Timeline in seconds, as numpy array of floats.

        Delta time is given as seconds. First image is t=0.
        Length of array is length of time axis.

        Raises:
            ValueError: when there is no time axis
        """

        timeline = [0.0]
        for axis in self.axes:
            if axis.name in ['time', 'triggertime']:
                try:
                    values = axis.values
                    for t in range(1, len(values)):
                        timeline.append(values[t] - values[0])
                    return np.array(timeline)
                except ValueError:
                    raise ValueError("No time axis is defined. Input order: {}".format(
                        self.input_order))
        raise ValueError("No time axis is defined. Input order: {}".format(self.input_order))

    @property
    def bvalues(self):
        """numpy.array: b-values in s/mm2, as numpy array of floats.

        Length of array is length of b axis.

        Raises:
            ValueError: when there is no diffusion (b) axis
        """
        try:
            return np.array(self.axes.b.values)
        except ValueError:
            raise ValueError("No b-value axis is defined. Input order: {}".format(
                self.input_order))

    def getDicomAttribute(self, keyword):
        """Get named DICOM attribute.

        Args:
            keyword (str): name or dicom tag

        Returns:
            DICOM attribute
        """

        try:
            if issubclass(type(keyword), str):
                _tag = pydicom.datadict.tag_for_keyword(keyword)
            else:
                _tag = keyword
            if _tag is None:
                return None
            if _tag in self.dicomTemplate:
                return self.dicomTemplate[_tag].value
            else:
                return None
        except ValueError:
            return None

    def setDicomAttribute(self, keyword, value, slice=None, tag=None):
        """Set named DICOM attribute.

        Args:
            keyword (str): name or dicom tag.
            value: new value for DICOM attribute.
            slice (int): optional slice to set attribute for. Default: all.
            tag (int): optional tag to set attribute for. Default: all.
        Raises:
            ValueError: When no DICOM tag is set.
        """

        if issubclass(type(keyword), str):
            _tag = pydicom.datadict.tag_for_keyword(keyword)
        else:
            _tag = keyword
        if _tag is None:
            raise ValueError('No DICOM tag set')
        self.header.dicomToDo.append(
            (_tag, value, slice, tag)
        )

    def getPositionForVoxel(self, r, transformation=None):
        """Get patient position for center of given voxel r.

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel

        Args:
            r (numpy.array): (s,r,c) of voxel in voxel coordinates
            transformation (numpy.array, optional): transformation matrix when different
                from self.transformationMatrix

        Returns:
            numpy.array((z,y,x)): position of voxel in world coordinates (mm)
        """

        _name: str = '{}.{}'.format(__name__, self.getPositionForVoxel.__name__)

        if transformation is None:
            logger.debug('{}: use existing transformationMatrix {}'.format(
                _name, self.transformationMatrix.shape
            ))
            transformation = self.transformationMatrix
        else:
            logger.debug('{}: user-provided transformationMatrix'
                         '{}'.format(_name, transformation.shape)
                         )
        # q = self.getTransformationMatrix()

        # V = np.array([[r[2]], [r[1]], [r[0]], [1]])  # V is [x,y,z,1]
        # if len(r) == 3 or (len(r) == 1 and len(r[0] == 3)):
        #     p = np.vstack((r.reshape((3, 1)), [1]))
        # elif len(r) == 4:
        #     p = r.reshape((4, 1))
        try:
            p = r.reshape((4, 1))
        except ValueError:
            p = np.vstack((r.reshape((3, 1)), [1]))

        newposition = np.dot(transformation, p)

        return newposition[:3]  # z,y,x

    def getVoxelForPosition(self, p, transformation=None):
        """ Get voxel for given patient position p

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel

        Args:
            p (numpy.array): (z,y,x) of voxel in world coordinates (mm)
            transformation (numpy.array, optional): transformation matrix when different
                from self.transformationMatrix

        Returns:
            numpy.array((s,r,c)): of voxel in voxel coordinates
        """

        if transformation is None:
            transformation = self.transformationMatrix
        # q = self.getTransformationMatrix()

        # V = np.array([[p[2]], [p[1]], [p[0]], [1]])    # V is [x,y,z,1]

        try:
            pt = p.reshape((4, 1))
        except ValueError:
            pt = np.vstack((p.reshape((3, 1)), [1]))  # type: np.ndarray

        qinv = np.linalg.inv(transformation)
        r = np.dot(qinv, pt)

        # z,y,x
        return (r + 0.5).astype(int)[:3]

    def align(moving, reference, interpolation='linear', force=False, fill_value=0):
        """Align moving series (self) to reference.
        The moving series is resampled on the grid of the reference series.
        In effect the moving series is reformatted to the slices of the reference series.
        The aligned image is rounded to nearest integer when the moving image is integer.

        Examples:

            Align vibe series (moving) with reference dce series

            >>> moving = Series('vibe')
            >>> reference = Series('dce', 'time')
            >>> img = moving.align(reference)

        Args:
            moving (Series): The moving series which will be aligned to the reference series.
            reference (Series): Reference series.
            interpolation (str): Method of interpolation.
                See scipy.interpolate.RegularGridInterpolator for possible value.
                Default: 'linear'.

            force (bool): Override check on FrameOfReferenceUID when True. Default: False.
            fill_value (Numeric): value to fill voxels outside field-of-view. Default: 0.
        Returns:
            Series: Aligned series.
        Raises:
            ValueError: When FrameOfReference or TransformationMatrix is missing for either series.
        """

        from scipy.interpolate import RegularGridInterpolator
        
        def homogenize(matrix):
            """Convert 3x4 matrix to 4x4 homogeneous matrix if needed."""
            if matrix.shape == (3, 4):
                return np.vstack([matrix, [0, 0, 0, 1]])
            elif matrix.shape == (4, 4):
                return matrix
            else:
                raise ValueError("Transformation matrix must be 3x4 or 4x4.")


        if moving.color or reference.color:
            raise ValueError('Aligning color images not implemented.')

        if not force:
            if moving.frameOfReferenceUID != reference.frameOfReferenceUID:
                raise ValueError('FrameOfReferenceUID differ. Use force=True to override')

        # Create voxel grid in reference space
        cs, cr, cc = np.meshgrid(
            np.arange(reference.slices),
            np.arange(reference.rows),
            np.arange(reference.columns),
            indexing='ij'
        )
        flat_voxels = np.vstack([
            cs.flatten(),
            cr.flatten(),
            cc.flatten(),
            np.ones(cs.size)
        ])  # shape: (4, N)

        # Convert reference voxel coordinates to world coordinates
        T_ref = homogenize(reference.transformationMatrix)
        x_world = T_ref @ flat_voxels  # shape: (4, N)

        # Map world coordinates to moving voxel coordinates
        T_mov = homogenize(moving.transformationMatrix)
        T_mov_inv = np.linalg.inv(T_mov)
        moving_coords = T_mov_inv @ x_world  # shape: (4, N)
        interp_coords = moving_coords[:3, :].T  # shape: (N, 3)

        # Prepare output array
        output_shape = (reference.slices, reference.rows, reference.columns)

        if moving.ndim == 3:
            # Regular 3D volume
            fnc = RegularGridInterpolator(
                (np.arange(moving.slices),
                 np.arange(moving.rows),
                 np.arange(moving.columns)),
                moving,
                method=interpolation,
                bounds_error=False,
                fill_value=fill_value
            )
            interpolated = fnc(interp_coords)
            if np.issubdtype(moving.dtype, np.integer):
                interpolated = np.rint(interpolated)

            result = interpolated.reshape(output_shape).astype(moving.dtype)

            return Series(
                result,
                template=moving,
                geometry=reference,
                axes=[
                    reference.axes.slice,
                    reference.axes.row,
                    reference.axes.column
                ]
            )

        elif moving.ndim == 4:
            # Handle 4D (e.g. time or tags)
            n_tags = moving.shape[0]
            result = np.zeros((n_tags, *output_shape), dtype=moving.dtype)

            for i in range(n_tags):
                fnc = RegularGridInterpolator(
                    (np.arange(moving.slices),
                     np.arange(moving.rows),
                     np.arange(moving.columns)),
                    moving[i],
                    method=interpolation,
                    bounds_error=False,
                    fill_value=fill_value
                )
                interpolated = fnc(interp_coords)
                if np.issubdtype(moving.dtype, np.integer):
                    interpolated = np.rint(interpolated)
                result[i] = interpolated.reshape(output_shape)

            return Series(
                result,
                input_order=moving.input_order,
                template=moving,
                geometry=reference,
                axes=[
                    moving.axes[0],  # tag/time axis
                    reference.axes.slice,
                    reference.axes.row,
                    reference.axes.column
                ]
            )

        else:
            raise ValueError("Only 3D or 4D images are supported.")


    def to_channels(self, channels, labels):
        """Create a Series object with channeled data.

        Examples:

            >>> from imagedata import Series
            >>> channel0 = Series(...)
            >>> channel1 = Series(...)
            >>> channel2 = Series(...)
            >>> T2 = Series(...)
            >>> T2_channels = T2.to_channels([channel0, channel1, channel2], ['0', '1', '2'])

        Args:
            channels (list): List of data for each channel.
            labels (list): List of labels for each channel.

        Returns:
            Series: Channeled Series object
        """

        # if self.color:
        #     return self
        #
        # import matplotlib.pyplot as plt
        # import matplotlib.colors
        # from .viewer import get_window_level

        ch_shape = None
        ch_dtype = []
        for label, channel in zip(labels, channels):
            if issubclass(type(channel), np.ndarray):
                if ch_shape is None:
                    ch_shape = channel.shape
                if channel.shape != ch_shape:
                    raise IndexError('Shape of channel {} differ: {} vs {}'.format(
                        label, channel.shape, self.shape
                    ))
                ch_dtype.append(channel.dtype)
            else:
                ch_dtype.append(np.dtype(type(channel)))
        if ch_shape is None:
            ch_shape = (len(channels),)
        dtype = np.dtype([(label, dt) for label, dt in zip(labels, ch_dtype)])
        data = np.empty(ch_shape, dtype)
        for label, channel in zip(labels, channels):
            data[label][:] = channel
        data = np.asarray(data).view(Series)
        si = Series(
            data,
            input_order=self.input_order,
            geometry=self
        )
        # si.header.photometricInterpretation = 'RGB'
        si.header.add_template(self.header)
        return si

    def to_rgb(self, colormap='Greys_r', lut=None, norm='linear',
               clip='window', probs=(0.01, 0.999)):
        """Create an RGB color image of self.

        Examples:

            >>> T2 = Series(...)
            >>> T2_rgb = T2.to_rgb()

        Args:
            colormap (str): Matplotlib colormap name. Defaults: 'Greys_r'.
            lut (int): Number of rgb quantization levels.
                Default: None, lut is calculated from the voxel values.
            norm (str or matplotlib.colors.Normalize): Normalization method. Either linear/log,
                or the `.Normalize` instance used to scale scalar data to the [0, 1] range before
                mapping to colors using colormap.
            clip (str): How to clip the data values.
                Default: 'window', clip data to window center and width.
                'hist': clip data at histogram probabilities.
            probs (tuple): Minimum and maximum probabilities when clipping using histogram method.

        Returns:
            Series: RGB Series object
        """

        if self.color:
            return self

        import matplotlib.pyplot as plt
        import matplotlib.colors
        from .viewer import get_window_level

        if lut is None:
            lut = 256 if self.dtype.kind == 'f' else (np.nanmax(self).item()) + 1
        if lut == 1:
            lut = 2
        if isinstance(norm, str):
            if norm == 'linear':
                norm = matplotlib.colors.Normalize
            elif norm == 'log':
                norm = matplotlib.colors.LogNorm
            # elif norm == 'centered':
            #     norm = matplotlib.colors.CenteredNorm
            else:
                raise ValueError('Unknown normalization function: {}'.format(norm))
        if not issubclass(type(colormap), matplotlib.colors.Colormap):
            colormap = plt.get_cmap(colormap, lut)
        colormap.set_bad(color='k')  # Important for log display of non-positive values
        colormap.set_under(color='k')
        colormap.set_over(color='w')
        if type(norm) is type:
            if clip == 'window':
                window, level, vmin, vmax = get_window_level(self, norm, window=None, level=None)
            elif clip == 'hist':
                vmin, vmax = self.calculate_clip_range(probs, lut)
            else:
                raise ValueError('Unknow clip method: {}'.format(clip))
            norm = norm(vmin=vmin, vmax=vmax, clip=True)
        data = norm(self)
        color_data = colormap(data, bytes=True)[..., :3]  # Strip off alpha color
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        rgb = Series(
            color_data.copy().view(dtype=rgb_dtype).reshape(color_data.shape[:-1]),
            input_order=self.input_order,
            geometry=self
        )
        # if self.dtype.kind == 'f':
        #     rgb = Series(
        #         colormap(data, bytes=True)[..., :3],  # Strip off alpha color
        #         input_order=self.input_order,
        #         geometry=self,
        #         # axes=self.axes + [VariableAxis('rgb', ['r', 'g', 'b'])]
        #     )
        # else:
        #     rgb = Series(
        #         colormap(data, bytes=True)[..., :3],  # Strip off alpha color
        #         input_order=self.input_order,
        #         geometry=self,
        #         # axes=self.axes + [VariableAxis('rgb', ['r', 'g', 'b'])]
        #     )

        rgb.header.photometricInterpretation = 'RGB'
        rgb.header.add_template(self.header)
        # rgb.header.colormap = mpl.colorbar.ColorbarBase(
        #
        # )
        rgb.header.colormap = copy.copy(colormap)
        rgb.header.colormap_norm = copy.copy(norm)
        rgb.header.color = True
        return rgb

    def fuse_mask(self, mask, alpha=0.7, blend=False,
                  colormap='Greys_r', lut=None, norm='linear',
                  clip='window', probs=(0.01, 0.999),
                  maskmap='magma', maskrange=None):

        """Color fusion of mask

        Create an RGB image of self, overlaying/enhancing the mask area.

        When the mask is binary, it is gaussian filtered to disperse edges.

        With ideas from Hauke Bartsch and Sathiesh Kaliyugarasan (2023).

        Examples:

            >>> img = Series(...)
            >>> mask = Series(...)
            >>> overlayed_img = img.fuse_mask(mask, clip='hist')
            >>> overlayed_img.show()

        Args:
            mask (Series or np.ndarray): Mask image
            alpha (float): Alpha blending for each channel. Default: 0.7
            blend (bool): Whether the self image will be blended using alpha. Default: False
            colormap (str): Matplotlib colormap name for image. Defaults: 'Greys_r'.
            maskmap (str): Matplotlib colormap name for mask. Defaults: 'magma'.
            lut (int): Number of rgb quantization levels.
                Default: None, lut is calculated from the voxel values.
            norm (str or matplotlib.colors.Normalize): Normalization method. Either linear/log,
                or the `.Normalize` instance used to scale scalar data to the [0, 1] range before
                mapping to colors using colormap.
            clip (str): How to clip the data values.
                Default: 'window', clip data to window center and width.
                'hist': clip data at histogram probabilities.
            probs (tuple): Minimum and maximum probabilities when clipping using histogram method.
            maskrange (tuple): Range of mask colormap. Defaults: None: Use full mask range.
        Returns:
            Series: RGB Series object
        Raises:
            IndexError: When the mask does not match the image
        """

        from scipy.ndimage import gaussian_filter

        def _is_binary_mask(mask):
            if mask.dtype.kind == 'b':
                return True
            if mask.dtype.kind == 'i' or mask.dtype.kind == 'u':
                return np.nanmin(mask) == 0 and np.nanmax(mask) <= 1
            return False

        def _float_to_rgb_series(im):
            im8 = np.empty(
                im['R'].shape,
                dtype=np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
            )
            for _color in ['R', 'G', 'B']:
                im8[_color] = np.rint(im[_color])
            rgb = Series(
                im8,
                input_order=self.input_order,
                geometry=self,
                axes=self.axes
            )
            rgb.header.photometricInterpretation = 'RGB'
            rgb.header.add_template(self.header)
            return rgb

        if self.color:
            background = self / 255.  # [0, 1]
        else:
            background = self.to_rgb(colormap=colormap, lut=lut, norm=norm,
                                     clip=clip, probs=probs) / 255.
        if issubclass(type(mask), Series):
            if mask.color:
                raise ValueError('Mask cannot be a color image')
        if background.ndim == 2 and mask.ndim != 2:  # 2D case
            raise IndexError('Mask should be 2D')
        elif background.ndim > 2 and mask.ndim != 3:  # >= 3D case
            raise IndexError('Mask should be 3D')
        if mask.ndim == 2:
            if mask.shape != background.shape[-2:]:
                raise IndexError('Shape of mask does not match image')
        elif mask.ndim == 3:
            if mask.shape != background.shape[-3:]:
                raise IndexError('Shape of mask does not match image')

        if maskrange is None:
            maskrange = (np.nanmin(mask), np.nanmax(mask))
        if maskrange[0] == maskrange[1]:
            maskrange = (maskrange[0], maskrange[0] + 1)

        # Now smooth the colors channel
        if mask.ndim == 2:
            # mask_filter = np.zeros_like(mask, dtype=np.float32)
            # if np.nanmax(mask) > 0:
            #     mask_filter = mask.astype(np.float32) / np.nanmax(mask)  # [0, 1]
            mask_filter = mask.astype(np.float32)
            if _is_binary_mask(mask):
                mask_filter = gaussian_filter(mask_filter, sigma=1.5)
        elif mask.ndim == 3:
            # Smooth for each slice independently
            mask_filter = np.zeros_like(mask, dtype=np.float32)
            for _slice in range(mask.shape[0]):
                # _max_in_slice = np.nanmax(mask[_slice])
                # if _max_in_slice > 0:
                #     mask_filter[_slice] = \
                #         mask[_slice].astype(np.float32) / _max_in_slice  # [0, 1]
                mask_filter[_slice] = mask[_slice].astype(np.float32)
                if _is_binary_mask(mask):
                    mask_filter[_slice] = gaussian_filter(mask_filter[_slice], sigma=1.5)
        else:
            raise ValueError('Cannot fuse mask of dimension {}'.format(mask.ndim))

        if _is_binary_mask(mask):
            overlay = Series(np.zeros(mask.shape,
                                      dtype=np.dtype([
                                          ('R', np.float32),
                                          ('G', np.float32),
                                          ('B', np.float32)
                                      ])
                                      )
                             )
            overlay['R'] = mask_filter  # Red channel
            overlay_colormap = None
        else:
            overlay = Series(mask_filter)
            overlay.windowCenter = (maskrange[0] + maskrange[1]) / 2.
            overlay.windowWidth = abs(np.float64(maskrange[1]) - np.float64(maskrange[0]))
            overlay = overlay.to_rgb(colormap=maskmap)
            overlay_norm = overlay.colormap_norm
            overlay_colormap = overlay.colormap
            overlay = overlay / 255
        img = np.empty_like(overlay)
        for _color in ['R', 'G', 'B']:
            if blend:
                img[_color] = (1.0 - alpha) * overlay[_color] + alpha * background[_color]
            else:
                img[_color] = (1.0 - alpha) * overlay[_color] + background[_color]
            img[_color] = np.clip(img[_color], a_min=0, a_max=1)
            img[_color] *= 255

        img = _float_to_rgb_series(img)
        if overlay_colormap is not None:
            img.colormap = copy.copy(overlay_colormap)
            img.colormap_norm = copy.copy(overlay_norm)
            try:
                img.colormap_label = mask.seriesDescription
            except ValueError:
                pass
        return img

    def show(self, im2=None, fig=None, ax=None, colormap='Greys_r', norm='linear', colorbar=None,
             window=None, level=None, link=False):
        """Interactive display of series on screen.

        Allows interactive scrolling and adjustment of window center and level.

        With ideas borrowed from Erlend Hodneland (2021).

        Examples:

            >>> img = Series(...)
            >>> img.show()

        Args:
            im2 (Series or list of Series): Series or list of Series which will be displayed in
                addition to self.
            fig (matplotlib.plt.Figure, optional): if already exist
            ax (matplotlib.plt.Axes, optional): if already exist
            colormap (str): color map for display. Default: 'Greys_r'
            norm (str or matplotlib.colors.Normalize): Normalization method. Either linear/log, or
                the `.Normalize` instance used to scale scalar data to the [0, 1] range before
                mapping to colors using colormap.
            colorbar (bool): Display colorbar with image.
                Default: None: determine colorbar based on colormap and norm.
            window (number): window width of signal intensities. Default is DICOM Window Width.
            level (number): window level of signal intensities. Default is DICOM Window Center.
            link (bool): whether scrolling is linked between displayed images. Default: False

        Raises:
            ValueError: when image is not a subclass of Series, or too many viewports
                are requested.
            IndexError: when there is a mismatch with images and viewports.
        """
        from .viewer import Viewer, default_layout
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        def _get_ax(ax):
            try:
                _ax = ax[0][0]
            except (TypeError, IndexError):
                try:
                    _ax = ax[0]
                except (TypeError, IndexError):
                    _ax = ax
            return _ax

        notebook = mpl.get_backend() in ['nbagg', 'widget']

        # im2 can be single image or list of images
        images = list()
        images.append(self)
        if im2 is not None:
            if issubclass(type(im2), list):
                images += im2  # Join lists of Series
            else:
                images.append(im2)  # Append single Series instance

        # Create or connect to canvas
        if ax is not None:
            _ax = _get_ax(ax)
            fig = _ax.get_figure()
            # axes = np.array(_ax).reshape((1, 1))
            axes = ax
        else:
            if fig is None:
                fig = plt.figure()
            axes = default_layout(fig, len(images))

        try:
            self.viewer = Viewer(images, fig=fig, ax=axes,
                                 colormap=colormap, norm=norm, colorbar=colorbar,
                                 window=window, level=level, link=link)
        except AssertionError:
            raise
        self.viewer.connect()
        plt.tight_layout()
        plt.show(block=True)
        if not notebook:
            self.viewer.disconnect()

    def get_roi(self, roi=None, color='r', follow=False, vertices=False, im2=None, fig=None,
                ax=None,
                colormap='Greys_r', norm='linear', colorbar=None, window=None, level=None,
                link=False, single=False, onselect=None):
        """Let user draw ROI on image.

        Examples:

            >>> img = Series(...)
            >>> mask = img.get_roi()

            >>> mask, vertices = img.get_roi(vertices=True)

        Args:
            roi: Either predefined vertices (optional). Dict of slices, index as [tag,slice] or
                [slice], each is list of (x,y) pairs. Or a binary Series grid.
            color (str): Color of polygon ROI. Default: 'r'.
            follow: (bool) Copy ROI to next tag. Default: False.
            vertices (bool): Return both grid mask and dictionary of vertices. Default: False.
            im2 (Series or list of Series): Series or list of Series which will be displayed in
                addition to self.
            fig (matplotlib.plt.Figure, optional) if already exist
            ax (matplotlib.plt.Figure, optional) if already exist
            colormap (str): colour map for display. Default: 'Greys_r'
            norm (str or matplotlib.colors.Normalize): Normalization method. Either linear/log, or
                the `.Normalize` instance used to scale scalar data to the [0, 1] range before
                mapping to colors using colormap.
            colorbar (bool): Display colorbar with image.
                Default: None: determine colorbar based on colormap and norm.
            window (number): window width of signal intensities. Default is DICOM Window Width.
            level (number): window level of signal intensities. Default is DICOM Window Center.
            link (bool): whether scrolling is linked between displayed objects. Default: False.
            single (bool): draw ROI in single slice per tag. Default: False.
            onselect (function): call function when roi change. Default: None.
                When a polygon is completed or modified after completion,
                the *onselect* function is called and passed a list of the vertices as
                ``(xdata, ydata)`` tuples.

        Returns:
            If vertices: tuple of grid mask and vertices_dict. Otherwise, grid mask only.
                - grid mask: Series object with voxel=1 inside ROI.
                  Series object with shape (nz,ny,nx) (3D or more) or
                  (ny, nx) (2D) from original image,
                  dtype ubyte. Voxel inside ROI is 1, 0 outside.
                - vertices_dict: if vertices: Dictionary of vertices.

            If running from a notebook (widget/nbagg driver), no ROI is returned.
            Call get_roi_mask() afterwards to get the mask.

        Raises:
            ValueError: when image is not a subclass of Series, or too many viewports are
                        requested.
            IndexError: when there is a mismatch with images and viewports.
        """
        from .viewer import Viewer, default_layout
        import matplotlib.pyplot as plt
        import matplotlib as mpl

        notebook = mpl.get_backend() in ['nbagg', 'widget']

        # im2 can be single image or list of images
        images = list()
        images.append(self)
        if im2 is not None:
            if issubclass(type(im2), list):
                images += im2  # Join lists of Series
            else:
                images.append(im2)  # Append single Series instance

        # Create or connect to canvas
        if ax is not None:
            try:
                _ax = ax[0][0]
            except (TypeError, IndexError):
                try:
                    _ax = ax[0]
                except (TypeError, IndexError):
                    _ax = ax
            fig = _ax.get_figure()
            axes = np.array(_ax).reshape((1, 1))
        else:
            if fig is None:
                fig = plt.figure()
            axes = default_layout(fig, len(images))

        if roi is not None and issubclass(type(roi), Series):
            # Convert roi mask to roi vertices
            roi = self.vertices_from_grid(roi)

        try:
            self.viewer = Viewer(images, fig=fig, ax=axes, follow=follow,
                                 colormap=colormap, norm=norm, colorbar=colorbar,
                                 window=window, level=level, link=link, onselect=onselect)
        except AssertionError:
            raise
        self.viewer.connect_draw(roi=roi, color=color)
        plt.tight_layout()
        plt.show()
        # vertices = viewer.get_roi()
        self.latest_roi_parameters = (follow, vertices, single)
        if notebook:
            # Leave early without waiting for drawn mask
            if vertices:
                return None, None
            else:
                return None

        self.viewer.disconnect_draw()
        return self.get_roi_mask()

    def get_roi_mask(self):
        """Get mask drawn by user in get_roi().

        Examples:
            When used in Jupyter Notebook:

            >>> img = Series(...)
            >>> img.get_roi()
            >>> mask = img.get_roi_mask()

        Returns:
            If vertices: tuple of grid mask and vertices_dict. Otherwise, grid mask only.
                - grid mask: Series object with voxel=1 inside ROI.
                  Series object with shape (nz,ny,nx) from original image,
                  dtype ubyte. If original image is 2D, the mask
                  will be shape (ny,nx) from original image.
                  Voxel inside ROI is 1, 0 outside.
                - vertices_dict: if vertices: Dictionary of vertices.

        Raises:
            ValueError: when get_roi() has not produced a mask up front.
        """
        from .viewer import grid_from_roi

        if self.latest_roi_parameters is None:
            raise ValueError('Cannot get ROI mask until a successful call to get_roi().')

        follow, vertices, single = self.latest_roi_parameters
        if follow:
            input_order = self.input_order
        else:
            input_order = 'none'
        try:
            new_grid = grid_from_roi(self, self.viewer.get_roi(), single=single)
        except IndexError:
            if follow:
                new_grid = np.zeros_like(self, dtype=np.ubyte)
            else:
                if self.ndim == 2:
                    new_grid = np.zeros((self.rows, self.columns), dtype=np.ubyte)
                else:
                    new_grid = np.zeros((self.slices, self.rows, self.columns), dtype=np.ubyte)
            new_grid = Series(new_grid, input_order=input_order, template=self, geometry=self)
        new_grid.seriesDescription = 'ROI'
        new_grid.windowCenter = .5
        new_grid.windowWidth = 1
        if vertices:
            return new_grid, self.viewer.get_roi()  # Return grid and vertices
        else:
            return new_grid  # Return grid only

    def calculate_clip_range(self, probs: Tuple, bins: int = None):
        assert len(probs) == 2, "Wrong format of histogram probabilities"

        # Calculate cumulative counts and bin edges of the image
        if bins is None:
            bins = 1024 if self.dtype.kind == 'f' else (np.nanmax(self).item()) + 1
        cumcounts, bin_edges = np.histogram(self, bins=bins)
        # Normalize cumulative counts
        cumcounts = cumcounts.cumsum() / cumcounts.sum()

        # Find the indices of the bins that correspond to the given probabilities
        min_bin, max_bin = np.searchsorted(cumcounts, probs)
        # Get the intensity values at the min and max bins
        return bin_edges[min_bin], bin_edges[max_bin]

    def vertices_from_grid(self, grid, align: bool = True) -> dict:
        """Convert roi grid to roi vertices
        Assume there is a single, connected roi in the grid
        https://docs.opencv.org/4.x/d4/d73/tutorial_py_contours_begin.html

        Args:
            grid
            align

        Returns:
            dict
        """
        import cv2 as cv

        def _threshold(im):
            vertices = []
            ret, thresh = cv.threshold(im, 0.5, 1, cv.THRESH_BINARY)
            # method = cv.CHAIN_APPROX_SIMPLE
            method = cv.CHAIN_APPROX_TC89_L1
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, method)
            for contour in contours:
                for vertex in contour:
                    for point in vertex:
                        vertices.append((point[0], point[1]))
            return vertices

        # Ensure grid is aligned with image
        if align:
            grid = grid.align(self)

        vertices = {}
        if grid.axes[0].name == 'row':
            # 2D grid
            vertices[0] = _threshold(grid)
        elif grid.axes[0].name == 'slice':
            # 3D grid
            for _slice in range(grid.slices):
                contour = _threshold(grid[_slice])
                if len(contour) > 0:
                    vertices[_slice] = contour
        else:
            # 4D grid
            for _tag in range(self.tags[0]):
                for _slice in range(grid.slices):
                    contour = _threshold(grid[_tag, _slice])
                    if len(contour) > 0:
                        vertices[_tag, _slice] = contour

        return vertices


# -----------------------------------------------------------------------------
#
# Implementations of NumPy functions on Series instances
#
# -----------------------------------------------------------------------------


def implements(numpy_function):
    """Register an __array_function__ implementation for Series objects."""

    def decorator(func):
        HANDLED_FUNCTIONS[numpy_function] = func
        return func

    return decorator


def _delegate_to_numpy(func, arrays, **kwargs):
    if issubclass(type(arrays), tuple):
        arr0 = arrays[0]
        ndarrays = [arr0.view(np.ndarray)]
        for arr in arrays[1:]:
            ndarrays.append(arr.view(np.ndarray))
    else:
        arr0 = arrays
        ndarrays = [arr0.view(np.ndarray)]

    # Delegate function to numpy
    s = func(ndarrays, **kwargs)
    # View concatenated ndarray as Series object
    obj = s.view(Series)
    obj.header = copy.copy(arr0.header)

    return obj

def _delegate_args_to_numpy(func, *arrays, **kwargs):
    series_template = None
    a = []
    for item in arrays:
        if issubclass(type(item), Series):
            a.append(item.view(np.ndarray))
            series_template = item
        else:
            a.append(item)

    # Delegate function to numpy
    s = func(*a, **kwargs)
    if series_template is not None and issubclass(type(s), np.ndarray):
        # View concatenated ndarray as Series object
        obj = s.view(Series)
        obj.header = copy.copy(series_template.header)
        return obj
    return s

def _delegate_a_to_numpy(func, a, **kwargs):
    """Delegate function on single array to NumPy."""
    if issubclass(type(a), Series) and a.dtype.fields is not None:
        return _delegate_struct_to_numpy(func, a, **kwargs)
    else:
        ndarray = a.view(np.ndarray)
        s = func(ndarray, **kwargs)
        if issubclass(type(s), np.ndarray):
            obj = s.view(Series)
            obj.input_order = a.input_order
            obj.header.add_template(a.header)
            obj.header.add_geometry(a.header)
            if obj.axes[0].name[:7] == 'unknown' or obj.axes[0].name[:4] == 'none':
                new_keys = [obj.input_order] + list(obj.axes._fields[1:])
                values = list(obj.axes)
                values[0].name = obj.input_order
                new_axes = namedtuple('Axes', new_keys)
                obj.axes = new_axes._make(values)
        else:
            obj = s
    return obj


def _delegate_struct_to_numpy(func, a, **kwargs):
    field_names = a.dtype.names
    obj = {}
    for field in field_names:
        ndarray = a[field].view(np.ndarray)
        s = func(ndarray, **kwargs)
        if issubclass(type(s), np.ndarray):
            obj[field] = s.view(Series)
            obj[field].header = copy.copy(a.header)
        else:
            obj[field] = s
    if np.isscalar(obj[field_names[0]]):
        _list = []
        for field in field_names:
            _list.append(obj[field])
        return tuple(_list)
    return a.to_channels(
        [obj[field] for field in field_names],
        field_names
    )

def _delegate_a_to_numpy_out(func, a, **kwargs):
    """Delegate function on single array to NumPy.
    Return NumPy output unmodified."""
    ndarray = a.view(np.ndarray)
    return func(ndarray, **kwargs)

@implements(np.minimum)
def _minimum(a, **kwargs):
    return _delegate_a_to_numpy(np.minimum, a, **kwargs)

@implements(np.maximum)
def _maximum(a, **kwargs):
    return _delegate_a_to_numpy(np.maximum, a, **kwargs)

@implements(np.min)
def _min(a, **kwargs):
    return _delegate_a_to_numpy(np.min, a, **kwargs)

@implements(np.nanmin)
def _min(a, **kwargs):
    return _delegate_a_to_numpy(np.nanmin, a, **kwargs)

@implements(np.max)
def _max(a, **kwargs):
    return _delegate_a_to_numpy(np.max, a, **kwargs)

@implements(np.nanmax)
def _max(a, **kwargs):
    return _delegate_a_to_numpy(np.nanmax, a, **kwargs)

@implements(np.nan_to_num)
def _nan_to_num(a, **kwargs):
    return _delegate_a_to_numpy(np.nan_to_num, a, **kwargs)
@implements(np.min_scalar_type)
def _min_scalar_type(a):
    return _delegate_a_to_numpy(np.min_scalar_type, a)

@implements(np.result_type)
def _result_type(*a, **kwargs):
    return _delegate_args_to_numpy(np.result_type, *a, **kwargs)

@implements(np.rint)
def _rint(a, **kwargs):
    print('_rint: a:', type(a))
    return _delegate_a_to_numpy(np.rint, a, **kwargs)

@implements(np.zeros_like)
def _zeros_like(a, **kwargs):
    return _delegate_a_to_numpy(np.zeros_like, a, **kwargs)

@implements(np.empty_like)
def _empty_like(a, **kwargs):
    return _delegate_a_to_numpy(np.empty_like, a, **kwargs)

@implements(np.sum)
def _sum(a, **kwargs):
    return _delegate_a_to_numpy(np.sum, a, **kwargs)

@implements(np.mean)
def _mean(a, **kwargs):
    return _delegate_a_to_numpy(np.mean, a, **kwargs)

@implements(np.clip)
def _clip(a, **kwargs):
    return _delegate_a_to_numpy(np.clip, a, **kwargs)

@implements(np.count_nonzero)
def _count_nonzero(a, **kwargs):
    return _delegate_a_to_numpy(np.count_nonzero, a, **kwargs)

@implements(np.concatenate)
def concatenate(arrays, axis=0, out=None):
    # implementation of concatenate for Series objects
    # Compare axes, except for concatenation axis
    arr0 = arrays[0]
    axis_name = arr0.axes[axis].name
    ndarrays = [arr0.view(np.ndarray)]
    for arr in arrays[1:]:
        ndarrays.append(arr.view(np.ndarray))
        for i, axs in enumerate(zip(arr0.axes, arr.axes)):
            if i == axis:
                continue  # Skip concatenation axis
            if axs[0] != axs[1]:
                return ValueError('Axis {} {} differ'.format(i, axs[0].name))

    # Concatenate the ndarrays
    s = np.concatenate(ndarrays, axis=axis, out=out)
    # View concatenated ndarray as Series object
    obj = s.view(Series)
    obj.header = copy.copy(arr0.header)

    # Concatenate axes
    new_axis = arrays[0].axes[axis].copy()
    for arr in arrays[1:]:
        new_axis.append(arr.axes[axis])
    obj.axes = obj.axes._replace(**{axis_name: new_axis})

    # Concatenate tags
    if obj.axes[axis].name == 'slice':
        for arr in arrays[1:]:
            for _slice in range(len(obj.tags), len(obj.tags)+arr.slices):
                obj.tags[_slice] = obj.tags[0]
            obj.sliceLocations = np.append(obj.sliceLocations, arr.sliceLocations)
    elif obj.axes[axis].name in ('row', 'column'):
        pass
    else:
        for _slice in range(obj.slices):
            for arr in arrays[1:]:
                obj.tags[_slice] = np.append(obj.tags[_slice], arr.tags[_slice])
    return obj
