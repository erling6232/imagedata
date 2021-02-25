"""Image series

The Series class is a subclassed Numpy.ndarray enhancing the array with relevant medical image methods and attributes.

  Typical example usage:

  si = Series('input')

"""

import copy
import numpy as np
from numpy.compat import basestring
import logging
import pydicom.dataset
import pydicom.datadict
# import numpy.core._multiarray_umath

import imagedata.axis
import imagedata.formats
import imagedata.readdata as readdata
import imagedata.formats.dicomlib.uid
from imagedata.header import Header, add_template, add_geometry

logging.getLogger(__name__).addHandler(logging.NullHandler())

# Verify that pathlib.Path module is available
# Set Path variable to reflect state
try:
    from pathlib import Path
except ImportError:
    Path = None


class DoNotSetSlicesError(Exception):
    pass


def is_pathlib_path(obj):
    """
    Check whether obj is a pathlib.Path object.
    """
    return Path is not None and isinstance(obj, Path)


class Series(np.ndarray):
    """Series -- a multidimensional array of medical imaging pixels.

    The Series class is a subclassed Numpy.ndarray enhancing the ndarray
    with relevant medical image methods and attributes.

    Series(data, input_order='none', opts=None, shape=(0,), dtype=float, order=None, template=None, geometry=None)

    Attributes:
        shape: shape of array

    Examples:

        Read the contents of an input directory

        >>> image = Series('directory/')

        Make a Series instance from a Numpy.ndarray

        >>> a = np.eye(128)
        >>> image = Series(a)

    Args:
        data: (array_like or URL)
            Input data, either explicit as np.ndarray, np.uint16, np.float32, or by URL to input data.
        input_order: (str)
            How to sort the input data. Typical values are:
                * 'none' : 3D volume or 2D slice (default).
                * 'time' : Time-resolved data.
                * 'b' : Diffusion data with variable b values.
                * 'te' : Varying echo times.
                * 'fa' : Varying flip angles.
        opts: (argparse.Namespace or dict)
            Dict of input options, mostly for format specific plugins.
        shape: Tuple of ints, specifying shape of input data.
        dtype: Numpy data type.
            Default: float
        template: (Series, array_like or URL)
            Input data to use as template for DICOM header.
        geometry: (Series, array_like or URL)
            Input data to use as template for geometry.
        order: {'C', 'F'}, optional
            Row-major (C-style) or column-major (Fortran-style) order.

    Returns:
        Series instance
    """

    name = "Series"
    description = "Image series"
    authors = "Erling Andersen"
    version = "1.1.1"
    url = "www.helse-bergen.no"

    def __new__(cls, data, input_order='none',
                opts=None, shape=(0,), dtype=float, buffer=None, offset=0,
                strides=None, order=None,
                template=None, geometry=None):

        if issubclass(type(template), Series):
            template = template.header
        if issubclass(type(geometry), Series):
            geometry = geometry.header
        if issubclass(type(data), np.ndarray):
            logging.debug('Series.__new__: data ({}) is subclass of np.ndarray'.format(type(data)))
            obj = np.asarray(data).view(cls)
            # Initialize attributes to defaults
            # cls.__init_attributes(cls, obj)
            # obj.header = Header() # Already set in __array_finalize__

            if issubclass(type(data), Series):
                # Copy attributes from existing Series to newly created obj
                # obj.__dict__ = data.__dict__.copy()  # carry forward attributes
                obj.header = copy.copy(data.header)  # carry forward attributes

            # set the new 'input_order' attribute to the value passed
            obj.header.input_order = input_order
            # obj.header.set_default_values() # Already done in __array_finalize__
            add_template(obj.header, template)
            add_geometry(obj.header, geometry)
            return obj
        logging.debug('Series.__new__: data is NOT subclass of Series, type {}'.format(type(data)))

        if issubclass(type(data), np.uint16) or issubclass(type(data), np.float32):
            obj = np.asarray(data).view(cls)
            # cls.__init_attributes(cls, obj)
            obj.header = Header()
            obj.header.input_order = input_order
            obj.header.set_default_values(obj.axes)
            add_template(obj.header, template)
            add_geometry(obj.header, geometry)
            return obj

        # Assuming data is url to input data
        if isinstance(data, basestring):
            urls = data
        elif isinstance(data, list):
            urls = data
        elif is_pathlib_path(data):
            urls = data.resolve()
        else:
            # raise ValueError("Input data could not be resolved: ({}) {}".format(type(data),data))
            obj = np.asarray(data).view(cls)
            # cls.__init_attributes(cls, obj)
            obj.header = Header()
            obj.header.input_order = input_order
            obj.header.set_default_values(obj.axes)
            add_template(obj.header, template)
            add_geometry(obj.header, geometry)
            return obj

        # Read input, hdr is dict of attributes
        hdr, si = readdata.read(urls, input_order, opts)

        obj = np.asarray(si).view(cls)
        assert obj.header, "No Header found in obj.header"

        # set the new 'input_order' attribute to the value passed
        obj.header.input_order = input_order
        # Copy attributes from hdr dict to newly created obj
        logging.debug('Series.__new__: Copy attributes from hdr dict to newly created obj')
        if obj.axes is None and 'axes' in hdr:
            axes = hdr['axes']
        else:
            axes = obj.axes
        obj.header.set_default_values(axes)
        for attr in hdr.keys():
            setattr(obj.header, attr, hdr[attr])
        # Store any template and geometry headers,
        add_template(obj.header, template)
        add_geometry(obj.header, geometry)
        # Finally, we must return the newly created object
        return obj

    def __array_finalize__(self, obj) -> None:
        # logging.debug("Series.__array_finalize__: entered obj: {}".format(type(obj)))
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
        # logging.debug("Series.__array_finalize__: obj: {}".format(type(obj)))

        if issubclass(type(obj), Series):
            # Copy attributes from obj to newly created self
            # logging.debug('Series.__array_finalize__: Copy attributes from {}'.format(type(obj)))
            # self.__dict__ = obj.__dict__.copy()  # carry forward attributes
            self.header = copy.copy(obj.header)  # carry forward attributes
        else:
            self.header = Header()
            self.header.set_default_values(self.axes)

        # We do not need to return anything

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        # logging.debug('Series.__array_ufunc__ method: %s' % method)
        args = []
        in_no = []
        # logging.debug('Series.__array_ufunc__ inputs: %s %d' % (
        #    type(inputs), len(inputs)))
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Series):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
                # logging.debug('                       input %d: Series' % i)
                # logging.debug('                       input %s: ' % type(input_))
                # logging.debug('                       input spacing {} '.format(
                #        input_.spacing))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        out_no = []
        # logging.debug('Series.__array_ufunc__ inputs: %d' % len(inputs))
        if outputs:
            out_args = []
            for j, output in enumerate(outputs):
                if isinstance(output, Series):
                    out_no.append(j)
                    out_args.append(output.view(np.ndarray))
                else:
                    out_args.append(output)
            kwargs['out'] = tuple(out_args)
        else:
            outputs = (None,) * ufunc.nout

        info = {}
        if in_no:
            info['inputs'] = in_no
        if out_no:
            info['outputs'] = out_no

        results = super(Series, self).__array_ufunc__(ufunc, method,
                                                      *args, **kwargs)
        if results is NotImplemented:
            return NotImplemented

        if method == 'at':
            if isinstance(inputs[0], Series):
                inputs[0].header = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(Series)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        if results and isinstance(results[0], Series):
            # logging.debug('Series.__array_ufunc__ add info to results:\n{}'.format(info))
            results[0].header = self._unify_headers(inputs)

        return results[0] if len(results) == 1 else results

    @staticmethod
    def _unify_headers(inputs: tuple) -> Header:
        """Unify the headers of the inputs.

        Typical usage is in expressions like c = a + b where at least
        one of the arguments is a Series instance.
        This function will provide a header for the result of
        the expression.

        Inputs:
        - inputs: a tuple of arguments (ndarray or Series)
        Return:
        - header: a Header() class
        """

        header = None
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Series):
                if header is None:
                    header = input_.header
                    # logging.debug('Series._unify_headers: copy header')
                # else:
                # Here we could have compared the headers of
                # the arguments and resolved discrepancies.
                # The simplest resolution, however, is to take the 
                # header of the first argument.
        return header

    def __getitem__(self, item):
        """__getitem__(self, item)
        
        Called to implement evaluation of self[item]. For sequence types, the
        accepted items should be integers and slice objects. Note that the
        special interpretation of negative indexes (if the class wishes to
        emulate a sequence type) is up to the __getitem__() method. If item is
        of an inappropriate type, TypeError may be raised; if of a value
        outside the set of indexes for the sequence (after any special
        interpretation of negative values), IndexError should be raised. For
        mapping types, if item is missing (not in the container), KeyError
        should be raised. Note: for loops expect that an IndexError will be
        raised for illegal indexes to allow proper detection of the end of the
        sequence.
        """

        def _set_geometry(_ret, _todo):
            for attr, value in _todo:
                try:
                    setattr(_ret, attr, value)
                except AttributeError:
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
            if isinstance(obj, Series):
                # Calculate slice range
                for _dim in range(obj.ndim):  # Loop over actual array shape
                    _spec[_dim] = (0, obj.shape[_dim], 1, obj.axes[_dim])  # Initial start,stop,step,axis

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

                for i_item in index_item:
                    # If any item is of unknown type, we will not slice the data
                    if not isinstance(_items[i_item], slice) and not isinstance(_items[i_item], int):
                        return _slicing, _spec
                for i_spec, i_item in zip(index_spec, index_item):
                    if isinstance(_items[i_item], slice):
                        _start = _items[i_item].start or _spec[i_spec][0]
                        _stop = _items[i_item].stop or _spec[i_spec][1]
                        _step = _items[i_item].step or _spec[i_spec][2]
                        _spec[i_spec] = (_start, _stop, _step, obj.axes[i_spec])
                        _slicing = True
                    elif isinstance(_items[i_item], int):
                        _start = _items[i_item] or _spec[i_spec][0]
                        _stop = _start + 1
                        _step = 1
                        _spec[i_spec] = (_start, _stop, _step, obj.axes[i_spec])
                        _slicing = True
            return _slicing, _spec

        if isinstance(item, tuple):
            items = item
        else:
            items = (item,)
        slicing, spec = _calculate_spec(self, items)

        todo = []  # Collect header changes, apply after ndarray slicing
        if slicing:
            # Here we slice the header information
            new_axes = []
            for i in range(self.ndim):
                # Slice dimension i
                start, stop, step, axis = spec[i]
                new_axes.append(axis[start:stop:step])

                if axis.name == 'slice':
                    # Select slice of sliceLocations
                    sloc = self.__get_sliceLocations(spec[i])
                    todo.append(('sliceLocations', sloc))
                    # Select slice of imagePositions
                    ipp = self.__get_imagePositions(spec[i])
                    todo.append(('imagePositions', None))  # Wipe existing positions
                    if ipp is not None:
                        todo.append(('imagePositions', ipp))
                elif axis.name == imagedata.formats.input_order_to_dirname_str(self.input_order):
                    # Select slice of tags
                    tags = self.__get_tags(spec)
                    todo.append(('tags', tags))
            # Select slice of DicomHeaderDict
            hdr = self.__get_DicomHeaderDict(spec)
            todo.append(('DicomHeaderDict', hdr))

        # Slicing the ndarray is done here
        ret = super(Series, self).__getitem__(item)
        if slicing and isinstance(ret, Series):
            # noinspection PyUnboundLocalVariable
            todo.append(('axes', new_axes[-ret.ndim:]))
            _set_geometry(ret, todo)
        return ret

    def __get_sliceLocations(self, spec):
        # logging.debug('__get_sliceLocations: enter')
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
        # logging.debug('__get_sliceLocations: exit')
        return sl

    def __get_imagePositions(self, spec):
        # logging.debug('__get_imagePositions: enter')
        try:
            ipp = self.imagePositions
        except ValueError:
            return None
        start, stop, step = 0, self.slices, 1
        if spec[0] is not None:
            start = spec[0]
        if spec[1] is not None:
            stop = spec[1]
        if spec[2] is not None:
            step = spec[2]
        ippdict = {}
        j = 0
        # logging.debug('__get_imagePositions: start,stop={},{}'.format(spec[0], stop))
        for i in range(start, stop, step):
            ippdict[j] = ipp[i]
            j += 1
        # logging.debug('__get_imagePositions: exit')
        return ippdict

    def __get_DicomHeaderDict(self, specs):
        try:
            slices = len(self.DicomHeaderDict)
        except ValueError:
            return None
        slice_spec = slice(0, slices, 1)
        assert len(self.tags) > 0, "No tags defined"
        tags = len(self.tags[0])
        tag_spec = slice(0, tags, 1)
        for d in specs:
            start, stop, step, axis = specs[d]
            if axis.name == 'slice':
                slice_spec = slice(start, stop, step)
            elif axis.name == imagedata.formats.input_order_to_dirname_str(self.input_order):
                tag_spec = slice(start, stop, step)
        # DicomHeaderDict[slice].tuple(tagvalue, filename, dicomheader)
        hdr = {}
        j = 0
        for s in range(slice_spec.start, slice_spec.stop, slice_spec.step):
            _slice = min(s, slices - 1)
            hdr[j] = list()
            for t in range(tag_spec.start, tag_spec.stop, tag_spec.step):
                try:
                    tag = self.tags[_slice][t]
                except IndexError:
                    raise IndexError("Could not get tag for slice {}, tag {}".format(_slice, t))
                hdr[j].append(
                    self.__find_tag_in_hdr(self.DicomHeaderDict[_slice], tag)
                )
            j += 1

        return hdr

    def __repr__(self):
        return """Patient: {} {}\nStudy  Time: {} {}\nSeries Time: {} {}\nSeries #{}: {}\nShape: {}, dtype: {}, input order: {}""".format(
            self.patientID, self.patientName,
            self.getDicomAttribute('StudyDate'), self.getDicomAttribute('StudyTime'),
            self.getDicomAttribute('SeriesDate'), self.getDicomAttribute('SeriesTime'),
            self.seriesNumber, self.seriesDescription,
            imagedata.formats.shape_to_str(self.shape), self.dtype,
            imagedata.formats.input_order_to_dirname_str(self.input_order))

    @staticmethod
    def __find_tag_in_hdr(hdr_list, find_tag):
        for tag, filename, hdr in hdr_list:
            if tag == find_tag:
                return tag, filename, hdr
        return None

    def __get_tags(self, specs):
        try:
            tmpl_tags = self.tags
            last_slice = len(self.tags) - 1  # known_slices might be less than actual data shape
            tags = len(self.tags[0])
        except ValueError:
            return None
        slice_spec = slice(0, self.slices, 1)
        tag_spec = slice(0, tags, 1)
        for d in specs:
            start, stop, step, axis = specs[d]
            if axis.name == "slice":
                slice_spec = slice(start, stop, step)
            elif axis.name == imagedata.formats.input_order_to_dirname_str(self.input_order):
                tag_spec = slice(start, stop, step)
        # tags: dict[slice] is np.array(tags)
        new_tags = {}
        j = 0
        for s in range(slice_spec.start, slice_spec.stop, slice_spec.step):
            new_tags[j] = list()
            for t in range(tag_spec.start, tag_spec.stop, tag_spec.step):
                # Limit slices to the known slices. Duplicates last slice and/or tag if too few.
                try:
                    new_tags[j].append(tmpl_tags[s][t])
                except IndexError:
                    raise IndexError("Could not get tag for slice {}, tag {}".format(s, t))
            j += 1
        return new_tags

    def write(self, url, opts=None, formats=None):
        """Write Series image

        Args:
            self: Series array
            directory_name: directory name
            filename_template: template including %d for image number
            opts: Output options (argparse.Namespace or dict)
            formats: list of output formats, overriding opts.output_format (list or str)
        """
        logging.debug('Series.write: url    : {}'.format(url))
        logging.debug('Series.write: formats: {}'.format(formats))
        logging.debug('Series.write: opts   : {}'.format(opts))
        readdata.write(self, url, formats=formats, opts=opts)

    @property
    def input_order(self):
        """Input order

        How to sort input files:
        INPUT_ORDER_NONE: No sorting
        INPUT_ORDER_TIME: Sort on image time (acquisition time or trigger time)
        INPUT_ORDER_B: Sort on b value
        INPUT_ORDER_FA: Sort on flip angle
        INPUT_ORDER_FAULTY: Correct erroneous attributes

        Returns:
            Current input order.
        """
        return self.header.input_order

    @input_order.setter
    def input_order(self, order):
        """Set input order

        Args:
            order: new input_order

        Raises:
            ValueError: when order is illegal.
        """
        if order in imagedata.formats.input_order_set:
            self.header.input_order = order
        else:
            raise ValueError("Unknown input order: {}".format(order))

    @property
    def input_format(self):
        """Input format

        Returns:
            The input format (str).
        """
        return self.header.input_format

    @input_format.setter
    def input_format(self, fmt):
        """Set input format

        Args:
            fmt: new input_format
        """
        self.header.input_format = fmt

    @property
    def input_sort(self):
        """Input order

        How to sort output files:
        SORT_ON_SLICE: Run over slices first
        SORT_ON_TAG  : Run over input order first, then slices

        Returns:
            The input order.

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
        """Set input sort order

        Args:
            order: new input sort order

        Raises:
            ValueError: when order is illegal.
        """
        if order is None or order in imagedata.formats.sort_on_set:
            self.header.input_sort = order
        else:
            raise ValueError("Unknown sort order: {}".format(order))

    @property
    def sort_on(self):
        """Output order

        How to sort output files:
        SORT_ON_SLICE: Run over slices first
        SORT_ON_TAG  : Run over input order first, then slices

        Returns:
            Current output order.
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
        """Set output sort order

        Args:
            order: new output sort order
        Raises:
            ValueError: when order is illegal.
        """
        if order in imagedata.formats.sort_on_set:
            self.header.sort_on = order
        else:
            raise ValueError("Unknown sort order: {}".format(order))

    @property
    def shape(self):
        """Matrix shape

        Returns:
            Current matrix shape as tuple
        """
        return super(Series, self).shape

    @shape.setter
    def shape(self, s):
        """Set matrix shape

        Args:
            s: new shape tuple
        Raises:
            IndexError: always (do not set shape)
        """
        raise IndexError('Should set axes instead of shape.')
        # prev_shape = super(Series, self).shape
        # super(Series, self).resize(s)
        # if len(s) > len(prev_shape):
        #     # Adding dimension => add axis
        #     for i in range(len(s) - len(prev_shape)):
        #         logging.debug('Series.shape add empty axis 0')
        #         self.axes.insert(
        #             0,
        #             imagedata.axis.Axis('none')
        #         )
        # elif len(s) < len(prev_shape):
        #     # Reduce dimension => remove axis
        #     for i in range(len(prev_shape) - len(s)):
        #         logging.debug('Series.shape delete axis 0')
        #         del self.axes[0]

    @property
    def rows(self):
        """Number of rows

        Returns:
            Number of rows.
        Raises:
            ValueError: when number of rows is not defined.
        """
        try:
            row_axis = self.find_axis('row')
            return len(row_axis)
        except ValueError:
            _color = 0
            if self.color:
                _color = 1
            if self.ndim - _color < 2:
                raise ValueError("{}D dataset has no rows".format(self.ndim))
            return self.shape[-2 - _color]

    @property
    def columns(self):
        """Number of columns

        Returns:
            Number of columns.
        Raises:
            ValueError: when number of columns is not defined.
        """
        try:
            column_axis = self.find_axis('column')
            return len(column_axis)
        except ValueError:
            _color = 0
            if self.color:
                _color = 1
            if self.ndim - _color < 1:
                raise ValueError("Dataset has no columns")
            return self.shape[-1 - _color]

    @property
    def slices(self):
        """Number of slices

        Returns:
            Number of slices.
        Raises:
            ValueError: when number of slices is not defined.
        """
        try:
            slice_axis = self.find_axis('slice')
            # logging.debug("Series.slices: {}D dataset slice_axis {}".format(self.ndim, slice_axis))
            return len(slice_axis)
        except ValueError:
            _color = 0
            if self.color:
                _color = 1
            if self.ndim - _color < 3:
                logging.debug("Series.slices: {}D dataset has no slices".format(self.ndim))
                # raise ValueError("{}D dataset has no slices".format(self.ndim))
                return 1
            logging.debug("Series.slices: {}D dataset slice from shape ({}) {}".format(
                self.ndim, self.shape, self.shape[-3 - _color]))
            return self.shape[-3 - _color]

    @slices.setter
    def slices(self, nslices):
        """Set number of slices

        Args:
            nslices: Number of slices.
        Raises:
            DoNotSetSlicesError: Always (do not set slices)
        """
        # if nslices > 0:
        #    if self.ndim > 2 and nslices != self.shape[-3]:
        #        logging.warning('Setting {} slices does not match shape {}'.format(nslices, self.shape))
        #    self.header.slices = nslices
        # else:
        #    self.header.slices = None
        raise DoNotSetSlicesError('Do not set slices=%d explicitly. Slices are inferred from the shape.' %
                                  nslices)

    @property
    def sliceLocations(self):
        """Slice locations

        Sorted numpy array of slice locations, in mm.

        Returns:
            List of slice locations as numpy array
        Raises:
            ValueError: When no slice locations are defined.
        """
        try:
            if self.header.sliceLocations is not None:
                return self.header.sliceLocations
            # Some image formats do not provide slice locations.
            # If orientation and imagePositions are set, slice locations can
            # be calculated.
            if self.header.orientation is not None and self.header.imagePositions is not None:
                logging.debug(
                    'sliceLocations: calculate {} slice from orientation and imagePositions'.format(self.slices))
                loc = np.empty(self.slices)
                normal = self.transformationMatrix[0, :3]
                for _slice in range(self.slices):
                    loc[_slice] = np.inner(normal, self.imagePositions[_slice])
                self.header.sliceLocations = loc
                return self.header.sliceLocations
        except AttributeError:
            pass
        raise ValueError("Slice locations are not defined.")

    @sliceLocations.setter
    def sliceLocations(self, loc):
        """Set slice locations

        Args:
            loc: list or numpy array of slice locations, in mm.
        """
        if loc is not None:
            self.header.sliceLocations = np.sort(loc)
        else:
            self.header.sliceLocations = None

    @property
    def DicomHeaderDict(self):
        """DICOM header dictionary

        DicomHeaderDict[slice].tuple(tagvalue, filename, dicomheader)

        Returns:
            DicomHeaderDict instance.
        Raises:
            ValueError: when DICOM header is not set.
        Examples:
            Get values for slice=0:

            >>> tagvalue, filename, dicomheader = image.DicomHeaderDict()[0]
        """
        # logging.debug('Series.DicomHeaderDict: here')
        try:
            if self.header.DicomHeaderDict is not None:
                # logging.debug('Series.DicomHeaderDict: return')
                # logging.debug('Series.DicomHeaderDict: return {}'.format(type(self.header.DicomHeaderDict)))
                # logging.debug('Series.DicomHeaderDict: return {}'.format(self.header.DicomHeaderDict.keys()))
                return self.header.DicomHeaderDict
        except AttributeError:
            pass
        raise ValueError("Dicom Header Dict is not set.")

    @DicomHeaderDict.setter
    def DicomHeaderDict(self, dct):
        """Set DicomHeaderDict

        Args:
            dct: DicomHeaderDict instance
        """
        self.header.DicomHeaderDict = dct

    @property
    def tags(self):
        """Image tags for each slice
        
        Image tags can be an array of:
            - time points
            - diffusion weightings (b values)
            - flip angles
        
        tags is a dict with (slice keys, tag array)

        dict[slice] is np.array(tags)

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

    @tags.setter
    def tags(self, tags):
        """Set tag(s) for given slice(s).
        Includes adjusting tags for DicomHeaderDict.
        
        Args:
            tags: dict() of np.array(tags)
                tags.keys() are slice numbers (int)
        """
        self.header.tags = {}
        hdr = {}
        for s in tags.keys():
            self.header.tags[s] = np.array(tags[s])
            hdr[s] = list()
            max_t = len(self.header.DicomHeaderDict[s]) - 1
            for t in range(len(tags[s])):
                if t <= max_t:
                    wrongtag, filename, dcm = self.header.DicomHeaderDict[s][t]
                    hdr[s].append(
                        (tags[s][t], filename, dcm)
                    )
                else:
                    # Copy last DicomHeaderDict
                    wrongtag, filename, dcm = self.header.DicomHeaderDict[s][max_t]
                    hdr[s].append(
                        (tags[s][t], None, dcm)
                    )
        self.header.DicomHeaderDict = hdr

    @property
    def axes(self):
        """axes

        Return axis object for given dimension.

        Returns:
            List of Axis objects
        Raises:
            ValueError: when the axis is not set.
        """
        try:
            if self.header.axes is not None:
                return self.header.axes
        except AttributeError:
            pass
        # Calculate axes from image shape
        self.header.axes = []
        shape = super(Series, self).shape
        if len(shape) < 1:
            return None
        if shape[-1] == 3 and self.dtype == np.uint8:
            _mono_shape = shape[:-1]
        else:
            _mono_shape = shape
        _max_known_shape = min(3, len(_mono_shape))
        _labels = ['slice', 'row', 'column'][-_max_known_shape:]
        while len(_labels) < self.ndim:
            _labels.insert(0, 'unknown')

        i = 0
        for d in super(Series, self).shape:
            self.header.axes.append(
                imagedata.axis.UniformLengthAxis(
                    _labels[i], 0, d, 1
                )
            )
            i += 1
        return self.header.axes

    @axes.setter
    def axes(self, ax):
        """Set axes and shape

        Args:
            ax: list of axis objects
        """
        self.header.axes = ax
        # shape = [len(axis) for axis in ax]
        # try:
        #     super(Series, self).resize(tuple(shape))
        # except AttributeError:
        #     pass

    def find_axis(self, name):
        """Find axis with given name

        Args:
            name: Axis name to search for
        Returnes:
            axis object with given name
        Raises:
            ValueError: when no axis object has given name
        """
        for axis in self.axes:
            if axis.name == name:
                return axis
        raise ValueError("No axis object with name %s exist" % name)

    @property
    def spacing(self):
        """spacing

        Given as dz,dy,dx in mm.

        Returns:
            spacing: [dz,dy,dx] in mm, as numpy array
        Raises:
            ValueError: when spacing is not set.
        """
        try:
            if self.header.spacing is not None:
                return self.header.spacing
        except AttributeError:
            pass
        raise ValueError("Spacing is unknown")

    @spacing.setter
    def spacing(self, *args):
        """Set spacing

        Args:
            spacing: dz,dy,dx in mm, given as numpy array, list or separate arguments
        Raises:
            ValueError: when spacing is not a tuple of 3 coordinates
        """
        if args[0] is None:
            self.header.spacing = None
            return
        logging.debug("spacing.setter {} {}".format(len(args), args))
        for arg in args:
            logging.debug("spacing.setter arg {} {}".format(len(arg), arg))
        # Invalidate existing transformation matrix
        self.header.transformationMatrix = None
        # Handle both tuple and component spacings
        if len(args) == 3:
            self.header.spacing = np.array(args)
        elif len(args) == 1:
            arg = args[0]
            if len(arg) == 3:
                self.header.spacing = np.array(arg)
            elif len(arg) == 1:
                arg0 = arg[0]
                if len(arg0) == 3:
                    self.header.spacing = np.array(arg0)
                else:
                    raise ValueError("Length of spacing in setSpacing(): %d" % len(arg0))
            else:
                raise ValueError("Length of spacing in setSpacing(): %d" % len(arg))
        else:
            raise ValueError("Length of spacing in setSpacing(): %d" % len(args))

    @property
    def imagePositions(self):
        """imagePositions

        The [z,y,x] coordinates of the upper left hand corner (first pixel)
        of each slice.

        dict(imagePositions[slice]) of [z,y,x] in mm, as numpy array

        Examples:
            >>> z,y,x = self.imagePositions[slice]

        Returns:
            dict of imagePositions. dict.keys() are slice numbers (int)
        Raises:
            ValueError: when imagePositions are not set.
        """
        # logging.debug('Series.imagePositions.get:')
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
                    logging.debug(
                        'Series.imagePositions.get: 1 positions only.  Calculating the other {} positions'.format(
                            self.slices - 1))
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
        """Set imagePositions

        poslist is added to existing imagePositions.
        Overlapping dict keys will replace exisiting imagePosition for given slice.

        Examples:
            for slice in range(slices):
                self.imagePositions = {
                    slice: self.getPositionForVoxel(np.array([slice,0,0]))
                }

        Args:
            poslist: dict of imagePositions. dict.keys() are slice numbers (int)
        Raises:
            AssertionError: when poslist has wrong shape or datatype.
        """
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
        # logging.debug("imagePositions set for keys {}".format(poslist.keys()))
        for _slice in poslist.keys():
            pos = poslist[_slice]
            # logging.debug("imagePositions set slice {} to {}".format(slice,pos))
            assert isinstance(pos, np.ndarray), "Wrong datatype of position (%s)" % type(pos)
            assert len(pos) == 3, "Wrong size of pos (is %d, should be 3)" % len(pos)
            self.header.imagePositions[_slice] = np.array(pos)

    @property
    def orientation(self):
        """Orientation

        The direction cosines of the first row and the first column with respect
        to the patient.
        These attributes shall be provided as a pair.
        Row value (column index) for the z,y,x axes respectively,
        followed by the column value (row index) for the z,y,x axes respectively.

        Returns:
            orientation as np.array with 6 elements.
        Raises:
            ValueError: when orientation is not set.
        """
        try:
            if self.header.orientation is not None:
                return self.header.orientation
        except AttributeError:
            pass
        raise ValueError("No orientation set.")

    @orientation.setter
    def orientation(self, orient):
        """Set orientation

        Args:
            orient: np.array or list of 6 elements
        Raises:
            AssertionError: when len(orient) != 6
        """
        if orient is None:
            self.header.transformationMatrix = None
            self.header.orientation = None
            return
        assert len(orient) == 6, "Wrong size of orientation"
        # Invalidate existing transformation matrix
        # self.header.transformationMatrix = None
        self.header.orientation = np.array(orient)

    @property
    def seriesNumber(self):
        """Series number
        
        DICOM series number.

        Returns:
            series number (int)
        Raises:
            ValueError: when series number is not set.
        """
        try:
            if self.header.seriesNumber is not None:
                return self.header.seriesNumber
        except AttributeError:
            pass
        raise ValueError("No series number set.")

    @seriesNumber.setter
    def seriesNumber(self, sernum):
        """Set series number

        Args:
            sernum: numerical or string that can be converted to int
        Raises:
            ValueError: when sernum cannot be converted to int
        """
        if sernum is None:
            self.header.seriesNumber = None
            return
        try:
            self.header.seriesNumber = int(sernum)
        except AttributeError:
            raise ValueError("Cannot convert series number to integer")

    @property
    def seriesDescription(self):
        """Series description

        DICOM series description.
        
        Returns:
            series description (string)
        Raises:
            ValueError: When series description is not set.
        """
        try:
            if self.header.seriesDescription is not None:
                return self.header.seriesDescription
        except AttributeError:
            pass
        raise ValueError("No series description set.")

    @seriesDescription.setter
    def seriesDescription(self, descr):
        """Set series description

        Args:
            descr: series description (str)
        Raises:
            AssertionError: when series description is not str
        """
        if descr is None:
            self.header.seriesDescription = None
            return
        assert isinstance(descr, str), "Given series description is not str"
        self.header.seriesDescription = descr

    @property
    def imageType(self):
        """Image type

        DICOM image type
        
        Returns:
            image type, list of strings
        Raises:
            ValueError: when image type is not set.
        """
        try:
            if self.header.imageType is not None:
                return self.header.imageType
        except AttributeError:
            pass
        raise ValueError("No image type set.")

    @imageType.setter
    def imageType(self, imagetype):
        """Set image type

        Args:
            imagetype: list of strings
        Raises:
            TypeError: When imagetype is not printable.
        """
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
        """Study instance UID

        DICOM study instance UID

        Returns:
            uid type, study instance UID (str)
        Raises:
            ValueError: when study instance UID is not set.
        """
        try:
            if self.header.studyInstanceUID is not None:
                return self.header.studyInstanceUID
        except AttributeError:
            pass
        raise ValueError("No study instance UID set.")

    @studyInstanceUID.setter
    def studyInstanceUID(self, uid):
        """Set study instance UID

        Args:
            uid: study instance UID
        Raises:
            TypeError: When uid is not printable.
        """
        if uid is None:
            self.header.studyInstanceUID = None
            return
        try:
            self.header.studyInstanceUID = str(uid)
        except AttributeError:
            raise TypeError("Given study instance UID is not printable")

    @property
    def studyID(self):
        """Study ID

        DICOM study ID
        
        Returns:
            study ID (str)
        Raises:
            ValueError: when study ID is not set.
        """
        try:
            if self.header.studyID is not None:
                return self.header.studyID
        except AttributeError:
            pass
        raise ValueError("No study ID set.")

    @studyID.setter
    def studyID(self, id):
        """Set study ID

        Args:
            id: study ID
        Raises:
            TypeError: When id is not printable
        """
        if id is None:
            self.header.studyID = None
            return
        try:
            self.header.studyID = str(id)
        except AttributeError:
            raise TypeError("Given study ID is not printable")

    @property
    def seriesInstanceUID(self):
        """Series instance UID

        DICOM series instance UID
        
        Returns:
            uid type, series instance UID (str)
        Raises:
            ValueError: when series instance UID is not set
        """
        try:
            if self.header.seriesInstanceUID is not None:
                return self.header.seriesInstanceUID
        except AttributeError:
            pass
        raise ValueError("No series instance UID set.")

    @seriesInstanceUID.setter
    def seriesInstanceUID(self, uid):
        """Set series instance UID

        Args:
            uid: series instance UID
        Raises:
            TypeError: When uid is not printable
        """
        if uid is None:
            self.header.seriesInstanceUID = None
            return
        try:
            self.header.seriesInstanceUID = str(uid)
        except AttributeError:
            raise TypeError("Given series instance UID is not printable")

    @property
    def frameOfReferenceUID(self):
        """Frame of reference UID

        DICOM frame of reference UID
        
        Returns:
            uid type, frame of reference UID (str)
        Raises:
            ValueError: when frame of reference UID is not set
        """
        try:
            if self.header.frameOfReferenceUID is not None:
                return self.header.frameOfReferenceUID
        except AttributeError:
            pass
        raise ValueError("No frame of reference UID set.")

    @frameOfReferenceUID.setter
    def frameOfReferenceUID(self, uid):
        """Set frame of reference UID

        Args:
            uid: frame of reference UID
        Raises:
            TypeError: When uid is not printable
        """
        if uid is None:
            self.header.frameOfReferenceUID = None
            return
        try:
            self.header.frameOfReferenceUID = str(uid)
        except AttributeError:
            raise TypeError("Given frame of reference UID is not printable")

    @property
    def accessionNumber(self):
        """Accession number

        DICOM accession number

        Returns:
            accession number (str)
        Raises:
            ValueError: when accession number is not set
        """
        try:
            if self.header.accessionNumber is not None:
                return self.header.accessionNumber
        except AttributeError:
            pass
        raise ValueError("No accession number set.")

    @accessionNumber.setter
    def accessionNumber(self, accno):
        """Set accession number

        Args:
            accno: accession number
        Raises:
            TypeError: When accno is not printable
        """
        if accno is None:
            self.header.accessionNumber = None
            return
        try:
            self.header.accessionNumber = str(accno)
        except AttributeError:
            raise TypeError("Given accession number is not printable")

    @property
    def patientName(self):
        """Patient name

        DICOM patient name

        Returns:
            patient name (str)
        Raises:
            ValueError: when patient name is not set
        """
        try:
            if self.header.patientName is not None:
                return self.header.patientName
        except AttributeError:
            pass
        raise ValueError("No patient name set.")

    @patientName.setter
    def patientName(self, patnam):
        """Set patient name

        Args:
            patnam: patient name
        Raises:
            TypeError: When patnam is not printable
        """
        if patnam is None:
            self.header.patientName = None
            return
        try:
            self.header.patientName = str(patnam)
        except AttributeError:
            raise TypeError("Given patient name is not printable")

    @property
    def patientID(self):
        """Patient ID

        DICOM patient ID

        Returns:
            patient ID (str)
        Raises:
            ValueError: when patient ID is not set
        """
        try:
            if self.header.patientID is not None:
                return self.header.patientID
        except AttributeError:
            pass
        raise ValueError("No patient ID set.")

    @patientID.setter
    def patientID(self, patid):
        """Set patient ID

        Args:
            patID: patient ID
        Raises:
            TypeError: When patID is not printable
        """
        if patid is None:
            self.header.patientID = None
            return
        try:
            self.header.patientID = str(patid)
        except AttributeError:
            raise TypeError("Given patient ID is not printable")

    @property
    def patientBirthDate(self):
        """Patient birth date

        DICOM patient birth date

        Returns:
            patient birth date (str)
                None: when no birth date is set
        Raises:
            ValueError: when patient birth date is not set
        """
        try:
            if self.header.patientBirthDate is not None:
                return self.header.patientBirthDate
        except AttributeError:
            pass
        raise ValueError("No patient birthdate set.")

    @patientBirthDate.setter
    def patientBirthDate(self, patbirdat):
        """Set patient birth date

        Args:
            patbirdat: patient birth date
        Raises:
            TypeError: When patbirdat is not printable
        """
        if patbirdat is None:
            self.header.patientBirthDate = None
            return
        try:
            self.header.patientBirthDate = str(patbirdat)
        except AttributeError:
            raise TypeError("Given patient birth date is not printable")

    @property
    def color(self):
        """Color interpretation

        Whether the array stores a color image, and the
        last index represents the color components
        
        Returns:
            whether the array stores a color image (bool)
        Raises:
            ValueError: when color interpretation is not set
        """
        try:
            if self.header.color is not None:
                return self.header.color
        except AttributeError:
            pass
        raise ValueError("No Color Interpretation is set.")

    @color.setter
    def color(self, color):
        """Set color interpretation

        Args:
            color: color interpretation (bool)
        Raises:
            TypeError: When color is not bool
        """
        if color is None:
            self.header.color = False
            return
        try:
            self.header.color = bool(color)
        except AttributeError:
            raise TypeError("Given color is not a boolean.")

    @property
    def photometricInterpretation(self):
        """Photometric Interpretation

        DICOM Photometric Interpretation
        
        Returns:
            photometric interpretation (str)
        Raises:
            ValueError: when photometric interpretation is not set
        """
        try:
            if self.header.photometricInterpretation is not None:
                return self.header.photometricInterpretation
        except AttributeError:
            pass
        raise ValueError("No Photometric Interpretation is set.")

    @photometricInterpretation.setter
    def photometricInterpretation(self, string):
        """Set photometric interpretation

        Args:
            string: photometric interpretation
        Raises:
            TypeError: When str is not printable
        """
        if string is None:
            self.header.photometricInterpretation = None
            return
        try:
            self.header.photometricInterpretation = str(string)
        except AttributeError:
            raise TypeError("Given phometric interpretation is not printable")

    # noinspection PyPep8Naming
    @property
    def transformationMatrix(self):
        """Transformation matrix

        Args:
            self.spacing
            self.imagePositions
            self.orientation
        Returns:
            transformation matrix as numpy array
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

        debug = None
        # debug = True

        try:
            if self.header.transformationMatrix is not None:
                return self.header.transformationMatrix

            # Calculate transformation matrix
            logging.debug('Series.transformationMatrix: Calculate transformation matrix')
            ds, dr, dc = self.spacing
            slices = len(self.header.imagePositions)
            T0 = self.header.imagePositions[0].reshape(3, 1)  # z,y,x
            Tn = self.header.imagePositions[slices - 1].reshape(3, 1)
            orient = self.orientation

            colr = np.array(orient[3:]).reshape(3, 1)
            colc = np.array(orient[:3]).reshape(3, 1)
            if slices > 1:
                logging.debug('Series.transformationMatrix: multiple slices case (slices={})'.format(slices))
                # Calculating normal vector based on first and last slice should be the correct method.
                k = (T0 - Tn) / (1 - slices)
                # Will just calculate normal to row and column to match other software.
                # k = np.cross(colr, colc, axis=0)
                # ##k = np.cross(colc, colr, axis=0)
                # ##k = k * ds
            else:
                logging.debug('Series.transformationMatrix: single slice case')
                k = np.cross(colr, colc, axis=0)
                # k = normalize(k) * ds
                k = k * ds
            logging.debug('Series.transformationMatrix: k={}'.format(k.T))
            # logging.debug("q: k {} colc {} colr {} T0 {}".format(k.shape,
            #    colc.shape, colr.shape, T0.shape))
            A = np.eye(4)
            A[:3, :4] = np.hstack([
                k,
                colr * dr,
                colc * dc,
                T0])
            if debug:
                logging.debug("A:\n{}".format(A))
            self.header.transformationMatrix = A
            return self.header.transformationMatrix
        except AttributeError:
            pass
        raise ValueError('Transformation matrix cannot be constructed.')

    @transformationMatrix.setter
    def transformationMatrix(self, m):
        """Set transformation matrix

        Requirements:
            - self.spacing must be set before setting transformationMatrix
            - self.slices  must be set before setting transformationMatrix

        Args:
            M: new transformation matrix
        Returns:
            self.orientation
            self.imagePositions
            self.transformationMatrix
        """
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
        #    for slice in range(1,self.slices):
        #        self.imagePositions = {
        #            slice: self.getPositionForVoxel(np.array([slice,0,0]),
        #                transformation=M)
        #        }

    def get_transformation_components_xyz(self):
        """Get origin and direction from transformation matrix in xyz convention.

        Args:
            self.transformationMatrix
            self.spacing
        Returns:
            origin: np.array size 3
            orientation: np.array size 6 (row, then column directional cosines) (DICOM convention)
            normal vector: np.array size 3 (slice direction)
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
        """Get timeline
        
        Returns:
            timeline in seconds, as numpy array of floats
                Delta time is given as seconds. First image is t=0.
                Length of array is number of tags.
        Raises:
            ValueError: tags for dataset is not time tags
        """
        if self.input_order == imagedata.formats.INPUT_ORDER_TIME:
            timeline = [0.0]
            for t in range(1, len(self.tags[0])):
                timeline.append(self.tags[0][t] - self.tags[0][0])
            return np.array(timeline)
        else:
            raise ValueError("No timeline tags are available. Input order: {}".format(self.input_order))

    def getDicomAttribute(self, keyword, slice=0, tag=0):
        """Get named DICOM attribute.

        Args:
            keyword: name (str) or dicom tag
            slice: optional slice to get attribute from (default: 0)
            tag: optional tag to get attribute from (default: 0)
        Returns:
            DICOM attribute
        """

        if self.DicomHeaderDict is None:
            return None
        if issubclass(type(keyword), str):
            _tag = pydicom.datadict.tag_for_keyword(keyword)
        else:
            _tag = keyword
        if _tag is None:
            return None
        tg, fname, im = self.DicomHeaderDict[slice][tag]
        if _tag in im:
            return im[_tag].value
        else:
            return None

    def setDicomAttribute(self, keyword, value, slice=None, tag=None):
        """Set named DICOM attribute.

        Args:
            keyword: name (str) or dicom tag
            value: new value for DICOM attribute
            slice: optional slice to get attribute from (default: all)
            tag: optional tag to get attribute from (default: all)
        Raises:
            ValueError: When no DICOM tag is set.
        """

        if self.DicomHeaderDict is None:
            return None
        if issubclass(type(keyword), str):
            _tag = pydicom.datadict.tag_for_keyword(keyword)
        else:
            _tag = keyword
        if _tag is None:
            raise ValueError('No DICOM tag set')
        slices = [i for i in range(self.slices)]
        tags = [i for i in range(len(self.tags[0]))]
        if slice is not None:
            slices = [slice]
        if tag is not None:
            tags = [tag]
        for s in slices:
            for t in tags:
                tg, fname, im = self.DicomHeaderDict[s][t]
                if _tag in im:
                    im[_tag].value = value
                else:
                    VR = pydicom.datadict.dictionary_VR(_tag)
                    im.add_new(_tag, VR, value)

    def getPositionForVoxel(self, r, transformation=None):
        """Get patient position for center of given voxel r

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel
        Args:
            r: (s,r,c) of voxel in voxel coordinates as numpy.array
            transformation: transformation matrix when different from
                          self.transformationMatrix
        Returns:
            (z,y,x) of voxel in world coordinates (mm) as numpy.array
        """

        if transformation is None:
            transformation = self.transformationMatrix
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
            p: (z,y,x) of voxel in world coordinates (mm) as numpy.array
            transformation: transformation matrix when different from
                          self.transformationMatrix
        Returns:
            (s,r,c) of voxel in voxel coordinates as numpy.array
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
        # return np.array([int(r[2,0]+0.5),int(r[1,0]+0.5),int(r[0,0]+0.5)], dtype=int)
        # return int(r+0.5)[:3]
        return (r + 0.5).astype(int)[:3]

    def show(self, im2=None, fig=None, cmap='gray', window=None, level=None, link=False):
        """Show image
        Courtesy of Erlend Hodneland (2021)
        """
        from imagedata.viewer import Viewer, default_layout
        import matplotlib.pyplot as plt

        # im2 can be single image or list of images
        images = list()
        images.append(self)
        if im2 is not None:
            if issubclass(type(im2), list):
                images += im2  # Join lists of Series
            else:
                images.append(im2)  # Append single Series instance

        # Create or connect to canvas
        if fig is None:
            fig = plt.figure()

        axes = default_layout(fig, len(images))
        try:
            viewer = Viewer(images, fig=fig, ax=axes,
                            cmap=cmap, window=window, level=level, link=link)
        except AssertionError:
            raise
        v = viewer.connect()
        plt.tight_layout()
        plt.show()
