"""Image series
"""

import numpy as np
from numpy.compat import basestring
import logging
import imagedata.formats
import imagedata.readdata as readdata

logging.getLogger(__name__).addHandler(logging.NullHandler())

try:
    from pathlib import Path
except ImportError:
    Path = None

def is_pathlib_path(obj):
    """
    Check whether obj is a pathlib.Path object.
    """
    return Path is not None and isinstance(obj, Path)

class Series(np.ndarray):

    name = "Series"
    description = "Image series"
    authors = "Erling Andersen"
    version = "1.1.1"
    url = "www.helse-bergen.no"

    header_tags = ['input_format', 'slices', 'sliceLocations',
                   'DicomHeaderDict', 'tags', 'spacing',
                   'imagePositions', 'orientation', 'seriesNumber',
                   'seriesDescription', 'imageType', 'frameOfReferenceUID',
                   'input_sort', 'transformationMatrix']

    def __new__(cls, data, input_order=0,
            opts=None, shape=(0,), dtype=float, buffer=None, offset=0,
            strides=None, order=None):

        if issubclass(type(data), np.ndarray):
            logging.debug('Series.__new__: data ({}) is subclass of np.ndarray'.format(type(data)))
            obj = np.asarray(data).view(cls)
            # Initialize attributes to defaults
            cls.__init_attributes(cls, obj)

            if issubclass(type(data), Series):
                # Copy attributes from existing Series to newly created obj
                obj.__dict__ = data.__dict__.copy()  # carry forward attributes

            # set the new 'input_order' attribute to the value passed
            obj.__input_order          = input_order
            return obj
        logging.debug('Series.__new__: data is NOT subclass of Series, type {}'.format(type(data)))

        if issubclass(type(data), np.uint16) or issubclass(type(data), np.float32):
            obj = np.asarray(data).view(cls)
            cls.__init_attributes(cls, obj)
            obj.__input_order          = input_order
            return obj

        # Assuming data is url to input data
        if isinstance(data, basestring):
            urls = data
        elif isinstance(data, list):
            urls = data
        elif is_pathlib_path(data):
            urls = data.resolve()
        else:
            #raise ValueError("Input data could not be resolved: ({}) {}".format(type(data),data))
            obj = np.asarray(data).view(cls)
            cls.__init_attributes(cls, obj)
            obj.__input_order          = input_order
            return obj

        # Read input, hdr is dict of attributes
        hdr,si = readdata.read(urls, input_order, opts)

        obj = np.asarray(si).view(cls)
        obj.__init_attributes(obj)

        # set the new 'input_order' attribute to the value passed
        obj.__input_order          = input_order
        # Copy attributes from hdr dict to newly created obj
        logging.debug('Series.__new__: Copy attributes from hdr dict to newly created obj')
        for attr in cls.header_tags:
            if attr in hdr:
                #logging.debug('Series.__new__: Set {} to {}'.format(attr, hdr[attr]))
                setattr(obj, attr, hdr[attr])
            else:
                #logging.debug('Series.__new__: Set {} to None'.format(attr))
                setattr(obj, attr, None)
        # Finally, we must return the newly created object
        return obj

    def __array_finalize__(self, obj) -> None:
        logging.debug("Series.__array_finalize__: entered obj: {}".format(type(obj)))
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
        if obj is None: return
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
        logging.debug("Series.__array_finalize__: obj: {}".format(type(obj)))

        if issubclass(type(obj), Series):
            # Copy attributes from obj to newly created self
            logging.debug('Series.__array_finalize__: Copy attributes from {}'.format(type(obj)))
            self.__dict__ = obj.__dict__.copy()  # carry forward attributes
        else:
            logging.debug('Series.__array_finalize__: init attributes')
            self.__init_attributes(self)

        # We do not need to return anything

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        args = []
        in_no = []
        for i, input_ in enumerate(inputs):
            if isinstance(input_, Series):
                in_no.append(i)
                args.append(input_.view(np.ndarray))
            else:
                args.append(input_)

        outputs = kwargs.pop('out', None)
        out_no = []
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
                inputs[0].info = info
            return

        if ufunc.nout == 1:
            results = (results,)

        results = tuple((np.asarray(result).view(Series)
                         if output is None else output)
                        for result, output in zip(results, outputs))
        if results and isinstance(results[0], Series):
            results[0].info = info

        return results[0] if len(results) == 1 else results

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

        def _set_geometry(ret, todo):
            for attr,value in todo:
                try:
                    setattr(ret, attr, value)
                except:
                    pass

        logging.debug('Series.__getitem__: item type {}: {}'.format(type(item),item))
        slicing = False
        if isinstance(self, Series):
            # Calculate slice range
            spec = {}
            logging.debug('Series.__getitem__: self.shape {}'.format(self.shape))
            for i in range(4): # Loop over tag,z,y,x regardless of array shape
                spec[i] = (0, None, 1) # Initial start,stop,step
            for i in range(self.ndim-1,-1,-1): # Loop over actual array shape
                spec[4-self.ndim+i] = (0, self.shape[i], 1) # Initial start,stop,step
            logging.debug('Series.__getitem__: spec {}'.format(spec))
            if isinstance(item, tuple):
                items = item
            else:
                items = (item,)
            logging.debug('Series.__getitem__: items type {}: {}'.format(type(items),item))
            for i in range(min(self.ndim,len(items))):
                if type(items[i]) == Ellipsis: break
                if isinstance(items[i], slice):
                    start,stop,step = spec[i]
                    if items[i].start is not None: start = items[i].start
                    if items[i].stop  is not None: stop  = items[i].stop
                    if items[i].step  is not None: step  = items[i].step
                    if step != 1:
                        raise IndexError('Step size = {} not implemented. Must be one.'.format(step))
                    spec[i] = (start,stop,step)
                    slicing = True
            if slicing:
                logging.debug('Series.__getitem__: tag slice: {}'.format(spec[0]))
                logging.debug('Series.__getitem__: z   slice: {}'.format(spec[1]))
                logging.debug('Series.__getitem__: y   slice: {}'.format(spec[2]))
                logging.debug('Series.__getitem__: x   slice: {}'.format(spec[3]))

                todo = []
                # Select slice of sliceLocations
                sloc = self.__get_sliceLocations(spec[1])
                todo.append(('sliceLocations', sloc))
                # Select slice of imagePositions
                ipp = self.__get_imagePositions(spec[1])
                todo.append(('imagePositions', None)) # Wipe existing positions
                todo.append(('imagePositions', ipp))
                # Select slice of DicomHeaderDict
                hdr = self.__get_DicomHeaderDict(spec)
                todo.append(('DicomHeaderDict', hdr))
                # Select slice of tags
                tags = self.__get_tags(spec)
                todo.append(('tags', tags))

        ret = super(Series, self).__getitem__(item)
        if slicing:
            _set_geometry(ret, todo)
        return ret

    def __init_attributes(self, obj):
        """Initialize attributes to defaults

        Input:
        - obj : Series instance which will initialized
        """
        obj.__input_order          = imagedata.formats.INPUT_ORDER_NONE
        obj.__sort_on              = None
        for attr in self.header_tags:
            try:
                # Must set the hidden __attributes to avoid exceptions on None values
                _attr = '__' + attr
                setattr(obj, _attr, None)
            except:
                pass

    def __get_sliceLocations(self, spec):
        stop = 1
        if spec[1] is not None: stop = spec[1]
        count = stop-spec[0]
        if self.ndim > 2 and count > self.slices:
            raise IndexError(
                'Too few sliceLocations={} in template for {} slices. Giving up!'.format(
                self.slices, count))
        try:
            sl = getattr(self, 'sliceLocations', None)
        except:
            sl = None
        if sl is not None:
            slist = []
            logging.debug('__get_sliceLocations: start,stop={},{}'.format(spec[0], stop))
            for i in range(spec[0], stop):
                slist.append(sl[i])
            sl = np.array(slist)
        return sl

    def __get_imagePositions(self, spec):
        stop = 1
        if spec[1] is not None: stop = spec[1]
        count = stop-spec[0]
        if self.ndim > 2 and count > self.slices:
            raise IndexError(
                'Too few imagePositions={} in template for {} slices. Giving up!'.format(
                self.slices, count))
        try:
            ipp = getattr(self, 'imagePositions', None)
        except:
            ipp = None
        if ipp is not None:
            ippdict = {}
            j = 0
            logging.debug('__get_imagePositions: start,stop={},{}'.format(spec[0], stop))
            for i in range(spec[0], stop):
                ippdict[j] = ipp[i]
                j += 1
            ipp = ippdict
        return ipp

    def __get_DicomHeaderDict(self, specs):
        # DicomHeaderDict[slice].tuple(tagvalue, filename, dicomheader)
        z_stop = 1
        if specs[1][1] is not None: z_stop = specs[1][1]
        tag_stop = 1; tags = 1
        if specs[0][1] is not None: tag_stop = specs[0][1]
        if self.ndim > 3: tags = self.shape[-4]
        count_tags   = tag_stop - specs[0][0]
        count_slices = z_stop - specs[1][0]
        if self.ndim > 2 and count_slices > self.slices:
            raise IndexError(
                'Too few sliceLocations={} in template for {} slices. Giving up!'.format(
                self.slices, count_slices))
        if self.ndim > 3 and count_tags > tags:
            raise IndexError(
                'Too few tags={} in template for {} tags. Giving up!'.format(
                tags, count_tags))
        try:
            tmpl_hdr = getattr(self, 'DicomHeaderDict', None)
        except:
            tmpl_hdr = None
        if tmpl_hdr is None:
            return tmpl_hdr
        hdr = {}
        j = 0
        logging.debug('__get_DicomHeaderDict: slice start,stop={},{}'.format(specs[1][0], z_stop))
        logging.debug('__get_DicomHeaderDict: tag start,stop={},{}'.format(specs[0][0], tag_stop))
        for i in range(specs[1][0], z_stop):
            hdr[j] = [False for x in range(count_tags)]
            n = 0
            for m in range(specs[0][0], tag_stop):
                hdr[j][n] = tmpl_hdr[i][m]
                n += 1
            j += 1
        return hdr

    def __get_tags(self, specs):
        # tags: dict[slice] is np.array(tags)
        z_stop = 1
        if specs[1][1] is not None: z_stop = specs[1][1]
        tag_stop = 1; tags = 1
        if specs[0][1] is not None: tag_stop = specs[0][1]
        if self.ndim > 3: tags = self.shape[-4]
        count_tags   = tag_stop - specs[0][0]
        count_slices = z_stop - specs[1][0]
        if self.ndim > 2 and count_slices > self.slices:
            raise IndexError(
                'Too few sliceLocations={} in template for {} slices. Giving up!'.format(
                self.slices, count_slices))
        if self.ndim > 3 and count_tags > tags:
            raise IndexError(
                'Too few tags={} in template for {} tags. Giving up!'.format(
                tags, count_tags))
        try:
            tmpl_tags = getattr(self, 'tags', None)
        except:
            tmpl_tags = None
        if tmpl_tags is None:
            return tmpl_tags
        new_tags = {}
        j = 0
        logging.debug('__get_tags: slice start,stop={},{}'.format(specs[1][0], z_stop))
        logging.debug('__get_tags: tag start,stop={},{}'.format(specs[0][0], tag_stop))
        for i in range(specs[1][0], z_stop):
            new_tags[j] = [False for x in range(count_tags)]
            n = 0
            for m in range(specs[0][0], tag_stop):
                new_tags[j][n] = tmpl_tags[i][m]
                n += 1
            j += 1
        return new_tags

    def write(self, dirname, filename_template, opts=None, formats=None):
        """Write Series image

        Input:
        - self: Series array
        - dirname: directory name
        - filename_template: template including %d for image number
        - opts: Output options (dict)
        - formats: list of output formats, overriding opts.output_format (list
          or str)
        """

        logging.debug('Series.write: dirname: {}'.format(dirname))
        logging.debug('Series.write: formats: {}'.format(formats))
        logging.debug('Series.write: opts   : {}'.format(opts))
        readdata.write(self, dirname, filename_template, formats=formats,
                opts=opts)

    @property
    def input_order(self):
        """Input order

        How to sort input files:
        INPUT_ORDER_NONE: No sorting
        INPUT_ORDER_TIME: Sort on image time (acquisition time or trigger time)
        INPUT_ORDER_B: Sort on b value
        INPUT_ORDER_FA: Sort on flip angle
        INPUT_ORDER_FAULTY: Correct erroneous attributes

        Returns current input order.
        """
        return self.__input_order

    @input_order.setter
    def input_order(self, order):
        """Set input order

        Input:
        - order: new input_order

        Raises ValueError when order is illegal.
        """
        if order in imagedata.formats.input_order_set:
            self.__input_order = order
        else:
            raise ValueError("Unknown input order: {}".format(order))

    @property
    def input_format(self):
        """Input format

        Returns the input format (str).
        """
        return self.__input_format

    @input_format.setter
    def input_format(self, fmt):
        """Set input format

        Input:
        - fmt: new input_format
        """
        self.__input_format = fmt

    @property
    def input_sort(self):
        """Input order

        How to sort output files:
        SORT_ON_SLICE: Run over slices first
        SORT_ON_TAG  : Run over input order first, then slices

        Returns the input order.
        Raises ValueError when input order is not defined.
        """
        try:
            if self.__input_sort is not None:
                return self.__input_sort
        except:
            pass
        raise ValueError("Input sort order not set.")

    @input_sort.setter
    def input_sort(self, order):
        """Set input sort order

        Input:
        - order: new input sort order

        Raises ValueError when order is illegal.
        """
        if order is None or order in imagedata.formats.sort_on_set:
            self.__input_sort = order
        else:
            raise ValueError("Unknown sort order: {}".format(order))

    @property
    def sort_on(self):
        """Output order

        How to sort output files:
        SORT_ON_SLICE: Run over slices first
        SORT_ON_TAG  : Run over input order first, then slices

        Returns current output order.
        Raises ValueError when output order is not defined.
        """
        try:
            if self.__sort_on is not None:
                return self.__sort_on
        except:
            pass
        raise ValueError("Output sort order not set.")

    @sort_on.setter
    def sort_on(self, order):
        """Set output sort order

        Input:
        - order: new output sort order

        Raises ValueError when order is illegal.
        """
        if order in imagedata.formats.sort_on_set:
            self.__sort_on = order
        else:
            raise ValueError("Unknown sort order: {}".format(order))

    @property
    def slices(self):
        """Number of slices

        Returns number of slices.
        Raises ValueError when number of slices is not defined.
        """
        #try:
        #    if self.__slices is not None:
        #        if self.__slices == self.shape[1]:
        #            return self.__slices
        #        else:
        #            raise ValueError('Number of slices ({}) does not match shape {}'.format(self.__slices, self.shape))
        #    else:
        #        return self.shape[1]
        #except:
        #    pass
        #raise ValueError("Number of slices is not set.")
        if self.ndim < 3:
            raise ValueError("{}D dataset has no slices".format(self.ndim))
        return self.shape[-3]

    @slices.setter
    def slices(self, nslices):
        """Set number of slices

        Input:
        - nslices: number of slices

        Raises ValueError when number of slices is illegal.
        """
        if nslices > 0:
            if nslices != self.shape[-3]:
                logging.warning('Setting {} slices does not match shape {}'.format(nslices, self.shape))
            self.__slices = nslices
        else:
            self.__slices = None

    @property
    def sliceLocations(self):
        """Slice locations

        Sorted numpy array of slice locations, in mm.
        """
        try:
            if self.__sliceLocations is not None:
                return self.__sliceLocations
            # Some image formats do not provide slice locations.
            # If orientation and imagePositions are set, slice locations can
            # be calculated.
            if self.__orientation is not None and self.__imagePositions is not None:
                logging.debug('sliceLocations: calculate {} slice from orientation and imagePositions'.format(self.slices))
                loc = np.empty(self.slices)
                normal = self.transformationMatrix[0,:3]
                for slice in range(self.slices):
                    loc[slice] = np.inner(normal, self.imagePositions[slice])
                self.__sliceLocations = loc
                return self.__sliceLocations
        except:
            pass
        raise ValueError("Slice locations are not defined.")

    @sliceLocations.setter
    def sliceLocations(self, loc):
        """Set slice locations

        Input:
        - loc: list or numpy array of slice locations, in mm.
        """
        if loc is not None:
            assert len(loc) == self.slices, "Mismatch number of slices ({}) and number of sliceLocations ({})".format(self.slices,len(loc))
            self.__sliceLocations = np.sort(loc)
        else:
            self.__sliceLocations = None

    @property
    def DicomHeaderDict(self):
        """DICOM header dictionary

        DicomHeaderDict[slice].tuple(tagvalue, filename, dicomheader)

        Returns DicomHeaderDict instance.
        Raises ValueError when header is not set.
        """
        try:
            if self.__DicomHeaderDict is not None:
                return self.__DicomHeaderDict
        except:
            pass
        raise ValueError("Dicom Header Dict is not set.")

    @DicomHeaderDict.setter
    def DicomHeaderDict(self, dct):
        """Set DicomHeaderDict

        Input:
        - dct: DicomHeaderDict instance
        """
        self.__DicomHeaderDict = dct

    @property
    def tags(self):
        """Image tags for each slice
        
        Image tags can be an array of:
        - time points
        - diffusion weightings (b values)
        - flip angles
        
        tags is a dict with (slice keys, tag array)
        dict[slice] is np.array(tags)

        Usage:
        self.tags[slice][tag]

        Raises ValueError when tags are not set.
        """
        try:
            if self.__tags is not None:
                return self.__tags
        except:
            pass
        raise ValueError("Tags not set.")

    @tags.setter
    def tags(self, tags):
        """Set tag(s) for given slice(s)
        
        Input:
        - tags: dict() of np.array(tags)

        dict.keys() are slice numbers (int)
        """
        self.__tags = {}
        for slice in tags.keys():
            self.__tags[slice] = np.array(tags[slice])

    @property
    def spacing(self):
        """spacing

        Given as dz,dy,dx in mm.

        Return:
        - [dz,dy,dx] in mm, as numpy array
        Exceptions:
        - ValueError: when spacing is not set.
        """
        try:
            if self.__spacing is not None:
                return self.__spacing
        except:
            pass
        raise ValueError("Spacing is unknown")

    @spacing.setter
    def spacing(self, *args):
        """Set spacing

        Input:
        - dz,dy,dx in mm, given as numpy array, list or separate arguments
        Exceptions:
        - ValueError: when spacing is not a tuple of 3 coordinates
        """
        if args[0] is None:
            self.__spacing = None
            return
        logging.debug("spacing.setter {} {}".format(len(args), args))
        for arg in args:
            logging.debug("spacing.setter arg {} {}".format(len(arg), arg))
        # Invalidate existing transformation matrix
        self.__transformationMatrix = None
        # Handle both tuple and component spacings
        if len(args) == 3:
            self.__spacing = np.array((args))
        elif len(args) == 1:
            arg = args[0]
            if len(arg) == 3:
                self.__spacing = np.array(arg)
            elif len(arg) == 1:
                arg0 = arg[0]
                if len(arg0) == 3:
                    self.__spacing = np.array(arg0)
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

        Usage:
        z,y,x = self.imagePositions[slice]

        Return:
        - dict of imagePositions. dict.keys() are slice numbers (int)
        Exceptions:
        - ValueError: when imagePositions are not set.
        """
        try:
            if self.__imagePositions is not None:
                logging.debug('Series.imagePositions.get: len(self.__imagePositions)={}'.format(len(self.__imagePositions)))
                logging.debug('Series.imagePositions.get: self.slices={}'.format(self.slices))
                if len(self.__imagePositions) != self.slices and len(self.__imagePositions) == 1:
                    # Some image formats define one imagePosition only.
                    # Could calculate the missing imagePositions from origin and
                    # orientation.
                    # Set imagePositions for additional slices
                    logging.debug('Series.imagePositions.get: 1 positions only.  Calculating the other {} positions'.format(self.slices-1))
                    M = self.transformationMatrix
                    for slice in range(1,self.slices):
                        #logging.debug('Series.imagePositions: slice {} pos {}'.format(slice, 'pos'))
                        #self.imagePositions = {
                        #    slice: self.getPositionForVoxel(np.array([slice,0,0]), transformation=M)
                        #}
                        self.__imagePositions[slice] = self.getPositionForVoxel(np.array([slice,0,0]),
                                transformation=M)
                #logging.debug('Series.imagePositions: new self.slices={}'.format(self.slices))
                #logging.debug('Series.imagePositions: new self.imagePositions={}'.format(self.__imagePositions))
                assert len(self.__imagePositions) == self.slices, "Mismatch number of slices ({}) and number of imagePositions ({})".format(self.slices,len(self.__imagePositions))
                return self.__imagePositions
        except:
            raise
        raise ValueError("No imagePositions set.")

    @imagePositions.setter
    def imagePositions(self, poslist):
        """Set imagePositions

        poslist is added to existing imagePositions.
        Overlapping dict keys will replace exisiting imagePosition for given slice.

        Usage:
        for slice in range(slices):
            self.imagePositions = {
                slice: self.getPositionForVoxel(np.array([slice,0,0]))
            }

        Input:
        - poslist: dict of imagePositions. dict.keys() are slice numbers (int)
        Exceptions:
        - AssertionError: when poslist has wrong shape or datatype.
        """
        if poslist is None:
            self.__imagePositions = None
            return
        assert isinstance(poslist, dict), "ImagePositions is not dict() (%s)" % type(poslist)
        # Invalidate existing transformation matrix
        #self.__transformationMatrix = None
        try:
            if self.__imagePositions is None:
                self.__imagePositions = dict()
        except:
            self.__imagePositions = dict()
        logging.debug("imagePositions set for keys {}".format(poslist.keys()))
        for slice in poslist.keys():
            pos = poslist[slice]
            #logging.debug("imagePositions set slice {} to {}".format(slice,pos))
            assert isinstance(pos, np.ndarray), "Wrong datatype of position (%s)" % type(pos)
            assert len(pos) == 3, "Wrong size of pos (is %d, should be 3)" % len(pos)
            self.__imagePositions[slice] = np.array(pos)

    @property
    def orientation(self):
        """Orientation

        The direction cosines of the first row and the first column with respect
        to the patient.
        These attributes shall be provided as a pair.
        Row value (column index) for the z,y,x axes respectively,
        followed by the column value (row index) for the z,y,x axes respectively.

        Returns:
        - orientation as np.array with 6 elements.
        Exceptions:
        - ValueError: when orientation is not set.
        """
        try:
            if self.__orientation is not None:
                return self.__orientation
        except:
            pass
        raise ValueError("No orientation set.")

    @orientation.setter
    def orientation(self, orient):
        """Set orientation

        Input:
        - orient: np.array or list of 6 elements
        Exceptions:
        - AssertionError: when len(orient) != 6
        """
        if orient is None:
            self.__transformationMatrix = None
            self.__orientation = None
            return
        assert len(orient)==6, "Wrong size of orientation"
        # Invalidate existing transformation matrix
        #self.__transformationMatrix = None
        self.__orientation = np.array(orient)

    @property
    def seriesNumber(self):
        """Series number
        
        DICOM series number.

        Returns:
        - series number (int)
        Exceptions:
        - ValueError: when series number is not set.
        """
        try:
            if self.__seriesNumber is not None:
                return self.__seriesNumber
        except:
            pass
        raise ValueError("No series number set.")

    @seriesNumber.setter
    def seriesNumber(self, sernum):
        """Set series number

        Input:
        - sernum: numerical or string that can be converted to int
        Exceptions:
        - ValueError: when sernum cannot be converted to int
        """
        if sernum is None:
            self.__seriesNumber = None
            return
        try:
            self.__seriesNumber = int(sernum)
        except:
            raise ValueError("Cannot convert series number to integer")

    @property
    def seriesDescription(self):
        """Series description

        DICOM series description.
        
        Returns:
        - series description (string)
        Exceptions:
        - ValueError: When series description is not set.
        """
        try:
            if self.__seriesDescription is not None:
                return self.__seriesDescription
        except:
            pass
        raise ValueError("No series description set.")

    @seriesDescription.setter
    def seriesDescription(self, descr):
        """Set series description

        Input:
        - descr: series description (str)
        Exceptions:
        - AssertionError: when series description is not str
        """
        if descr is None:
            self.__seriesDescription = None
            return
        assert isinstance(descr, str), "Given series description is not str"
        self.__seriesDescription = descr

    @property
    def imageType(self):
        """Image type

        DICOM image type
        
        Returns:
        - image type, list of strings
        Exceptions:
        - ValueError: when image type is not set.
        """
        try:
            if self.__imageType is not None:
                return self.__imageType
        except:
            pass
        raise ValueError("No image type set.")

    @imageType.setter
    def imageType(self, imagetype):
        """Set image type

        Input:
        - imagetype: list of strings
        Exceptions:
        - TypeError: When imagetype is not printable
        """
        if imagetype is None:
            self.__imageType = None
            return
        self.__imageType = list()
        try:
            for s in imagetype:
                self.__imageType.append(str(s))
        except:
            raise TypeError("Given image type is not printable (is %s)" % type(imagetype))

    @property
    def frameOfReferenceUID(self):
        """Frame of reference UID

        DICOM frame of reference UID
        
        Returns:
        - uid type, frame of reference UID (str)
        Exceptions:
        - ValueError: when frame of reference UID is not set
        """
        try:
            if self.__frameOfReferenceUID is not None:
                return self.__frameOfReferenceUID
        except:
            pass
        raise ValueError("No frame of reference UID set.")

    @frameOfReferenceUID.setter
    def frameOfReferenceUID(self, uid):
        """Set frame of reference UID

        Input:
        - uid: frame of reference UID
        Exceptions:
        - TypeError: When uid is not printable
        """
        if uid is None:
            self.__frameOfReferenceUID = None
            return
        try:
            self.__frameOfReferenceUID = str(uid)
        except:
            raise TypeError("Given frame of reference UID is not printable")

    @property
    def transformationMatrix(self):
        """Transformation matrix

        Input:
        - self.spacing
        - self.imagePositions
        - self.orientation
        Returns:
        - transformation matrix as numpy array
        """

        debug = None
        #debug = True

        try:
            if self.__transformationMatrix is not None:
                return self.__transformationMatrix

            # Calculate transformation matrix
            logging.debug('Series.transformationMatrix: Calculate transformation matrix')
            ds,dr,dc    = self.spacing
            slices      = len(self.__imagePositions)
            T0          = self.__imagePositions[0].reshape(3,1)   # z,y,x
            Tn          = self.__imagePositions[slices-1].reshape(3,1)
            orient      = self.orientation
            #print("ds,dr,dc={},{},{}".format(ds,dr,dc))
            #print("z ,y ,x ={},{},{}".format(z,y,x))

            colr=np.array([[orient[5]], [orient[4]], [orient[3]]]) * dr
            colc=np.array([[orient[2]], [orient[1]], [orient[0]]]) * dc
            if slices > 1:
                logging.debug('Series.transformationMatrix: multiple slices case (slices={})'.format(slices))
                k = (T0-Tn) / (1-slices)
            else:
                logging.debug('Series.transformationMatrix: single slice case')
                #k = np.cross(colc, colr, axis=0)
                k = np.cross(colr, colc, axis=0) * ds
            logging.debug('Series.transformationMatrix: k={}'.format(k.T))
            #logging.debug("Q: k {} colc {} colr {} T0 {}".format(k.shape,
            #    colc.shape, colr.shape, T0.shape))
            A = np.eye(4)
            A[:3, :4] = np.hstack((k, colr, colc, T0))
            if debug:
                logging.debug("A:\n{}".format(A))
            self.__transformationMatrix = A
            return self.__transformationMatrix
        except:
            pass
        raise ValueError('Transformation matrix cannot be constructed.')

    @transformationMatrix.setter
    def transformationMatrix(self, M):
        """Set transformation matrix

        Input:
        - M: new transformation matrix
        Output:
        - self.orientation
        - self.imagePositions
        - self.transformationMatrix
        Requirements:
        - self.spacing must be set before setting transformationMatrix
        - self.slices  must be set before setting transformationMatrix
        """
        self.__transformationMatrix = M
        #if M is not None:
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

    @property
    def timeline(self):
        """Get timeline
        
        Returns:
        - timeline in seconds, as numpy array of floats
          Delta time is given as seconds. First image is t=0.
          Length of array is number of tags.
        Exceptions:
        - ValueError: tags for dataset is not time tags
        """
        if self.input_order == imagedata.formats.INPUT_ORDER_TIME:
            timeline = []
            timeline.append(0.0)
            for t in range(1, len(self.tags[0])):
                timeline.append(self.tags[0][t] - self.tags[0][0])
            return np.array(timeline)
        else:
            raise ValueError("No timeline tags are available. Input order: {}".format(self.input_order))

    def getPositionForVoxel(self, r, transformation=None):
        """Get patient position for center of given voxel r

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel
        Input:
        - r: (z,y,x) of voxel in voxel coordinates as numpy.array
        - transformation: transformation matrix when different from
                          self.transformationMatrix
        Output:
        - (z,y,x) of voxel in world coordinates (mm) as numpy.array
        """

        if transformation is None:
            transformation = self.transformationMatrix
        #Q = self.getTransformationMatrix()

        #V = np.array([[r[2]], [r[1]], [r[0]], [1]])  # V is [x,y,z,1]

        newpos = np.dot(transformation, np.hstack((r, [1])))

        #return np.array([newpos[2,0],newpos[1,0],newpos[0,0]])   # z,y,x
        return newpos[:3]

    def getVoxelForPosition(self, p, transformation=None):
        """ Get voxel for given patient position p

        Use Patient Position and Image Orientation to calculate world
        coordinates for given voxel
        Input:
        - p: (z,y,x) of voxel in world coordinates (mm) as numpy.array
        - transformation: transformation matrix when different from
                          self.transformationMatrix
        Output:
        - (z,y,x) of voxel in voxel coordinates as numpy.array
        """

        if transformation is None:
            transformation = self.transformationMatrix
        #Q = self.getTransformationMatrix()

        #V = np.array([[p[2]], [p[1]], [p[0]], [1]])    # V is [x,y,z,1]

        Qinv = np.linalg.inv(transformation)
        r = np.dot(Qinv, np.hstack((p, [1])))

        # z,y,x
        #return np.array([int(r[2,0]+0.5),int(r[1,0]+0.5),int(r[0,0]+0.5)], dtype=int)
        #return int(r+0.5)[:3]
        return (r+0.5).astype(int)[:3]
