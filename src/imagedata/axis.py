"""Axis defines a dimension of an images Series.
"""

from abc import ABCMeta  # , abstractmethod, abstractproperty
import sys
# import logging
import numpy as np


class Axis(object, metaclass=ABCMeta):
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return ("{0}({1})".format(self.__class__.__name__, self.name))

    def __str__(self):
        return "{0.name!s}".format(self)


class UniformAxis(Axis):
    """Define axis by giving start, stop and step (optional).
    Start, stop and step are given in actual values

    Examples:
        >>> ax = UniformAxis('row', 0, 128)
    """

    def __init__(self, name, start, stop, step=1):
        super(UniformAxis, self).__init__(name)
        self.start = start
        self.stop = stop
        self.step = step

    def __getitem__(self, item):
        # logging.debug('UniformAxis: item {}'.format(item))
        # logging.debug('UniformAxis: isinstance(self, Axis): %s' % isinstance(self, Axis))
        assert isinstance(self, Axis), "self instance is not Axis"

        start, stop, step = 0, None, 1
        # logging.debug('UniformAxis: item %s' % type(item))
        if type(item) == Ellipsis:
            # logging.debug('UniformAxis: Ellipsis')
            return self
        elif isinstance(item, slice):
            # logging.debug('UniformAxis: slice')
            start = self.start + (item.start or 0) * self.step
            stop = self.stop
            if item.stop is not None:
                stop = self.start + (item.stop * self.step)
            stop = min(self.stop, stop)
            step = (item.step or 1) * self.step
        # logging.debug('UniformAxis: slice %d,%d,%d' % (start,stop,step))
        return UniformAxis(self.name, start, stop, step)

    def __len__(self):
        try:
            return abs(int((self.stop - self.start) / self.step))
        except ValueError as e:
            return sys.maxsize
        except Exception:
            raise

    @property
    def slice(self):
        return self.start, self.stop, self.step

    def __repr__(self):
        return("{0}({1.name!s},{1.start!s},{1.stop!s},{1.step!s})".format(
            self.__class__.__name__, self
        ))

    def __str__(self):
        return "{0.name!s}: {0.start!s}:{0.stop!s}:{0.step!s}".format(self)


class UniformLengthAxis(UniformAxis):
    """Define axis by giving start, length and step (optional).
    Start and step are given in actual values.

    Examples:
        >>> ax = UniformLengthAxis('row', 0, 128)
    """

    def __init__(self, name, start, n, step=1):
        super(UniformLengthAxis, self).__init__(name, start, start + n * step, step)
        self.n = n

    def __getitem__(self, item):
        # logging.debug('UniformLengthAxis: item {}'.format(item))
        # logging.debug('UniformLengthAxis: isinstance(self, Axis): %s' % isinstance(self, Axis))
        assert isinstance(self, Axis), "self instance is not Axis"

        start, n, step = self.start, self.n, self.step
        # logging.debug('UniformLengthAxis: item %s' % type(item))
        if type(item) == Ellipsis:
            # logging.debug('UniformLengthAxis: Ellipsis')
            return self
        elif isinstance(item, slice):
            # logging.debug('UniformLengthAxis: slice')
            start = self.start + (item.start or 0) * self.step
            stop = self.start + (item.stop or self.n) * self.step
            step = (item.step or 1) * self.step
            try:
                n = int(round((stop - start) / step))
            except ValueError as e:
                n = sys.maxsize
            except Exception:
                raise
            n = min(self.n, n)
        # logging.debug('UniformLengthAxis: slice %d,%d,%d' % (start,stop,step))
        return UniformLengthAxis(self.name, start, n, step)

    def __len__(self):
        return self.n

    def __repr__(self):
        return("{0}({1.name!s},{1.start!s},{1.n!s},{1.step!s})".format(
            self.__class__.__name__, self
        ))

    def __str__(self):
        return "{0.name!s}: {0.n!s}*({0.start!s}:{0.step!s})".format(self)


class VariableAxis(Axis):
    """Define axis by giving an array of values.
    values are actual values.

    Examples:
        >>> ax = VariableAxis('time', [0, 1, 4, 9, 11, 13])
    """

    def __init__(self, name, values):
        super(VariableAxis, self).__init__(name)
        self.values = np.array(values)

    def __getitem__(self, item):
        """Slice the axis
        - item: tuple of slice indices
        """
        # logging.debug('VariableAxis: item {}'.format(item))
        # logging.debug('VariableAxis: isinstance(self, Axis): %s' % isinstance(self, Axis))
        assert isinstance(self, Axis), "self instance is not Axis"

        start, stop, step = 0, None, 1
        # logging.debug('VariableAxis: item %s' % type(item))
        if type(item) == Ellipsis:
            # logging.debug('VariableAxis: Ellipsis')
            return self
        elif isinstance(item, slice):
            # logging.debug('VariableAxis: slice')
            start = item.start or 0
            stop = item.stop or len(self.values)
            stop = min(len(self.values), stop)
            step = item.step or 1
        # logging.debug('VariableAxis: slice %d,%d,%d' % (start,stop,step))
        return VariableAxis(self.name, self.values[start:stop:step])

    def __len__(self):
        return len(self.values)

    def __repr__(self):
        return("{0}({1.name!s},{1.values!r})".format(
            self.__class__.__name__, self
        ))

    def __str__(self):
        return "{0.name!s}: {0.values!s}".format(self)
