"""Axis defines a dimension of an images Series.
"""

from __future__ import annotations
from abc import ABCMeta
from typing import Sequence, Union, overload, SupportsFloat
import numbers
import sys
from collections import namedtuple
import numpy as np

Number = type[SupportsFloat]


class Axis(object, metaclass=ABCMeta):
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:
        return "{0}({1})".format(self.__class__.__name__, self.name)

    def __str__(self) -> str:
        return "{0.name!s}".format(self)

    def __eq__(self, other):
        if not issubclass(type(other), Axis):
            # don't attempt to compare against unrelated types
            return NotImplemented
        return self.name == other.name

    def append(self, axis: Axis):
        """Append another axis"""
        pass

    @property
    def values(self):
        """Get all axis values as list"""
        return [_ for _ in self]


class UniformAxis(Axis):
    """Define axis by giving start, stop and step (optional).
    Start, stop and step are given in actual values

    Examples:
        >>> ax = UniformAxis('row', 0, 128)
    """
    start: Number
    stop: Number
    step: Number

    def __init__(self,
                 name: str,
                 start: Number,
                 stop: Number,
                 step: Number = 1) -> None:
        super(UniformAxis, self).__init__(name)
        self.start = start
        self.stop = stop
        self.step = step

    def copy(self,
             name: str = None,
             start: Number = None,
             stop: Number = None,
             step: Number = None,
             n: int = None
             ) -> UniformAxis:
        """Return a copy of the axis, where the length n can be different."""
        name = self.name if name is None else name
        start = self.start if start is None else start
        stop = self.stop if stop is None else stop
        step = self.step if step is None else step
        if n is not None:
            stop = start + (n + 1) * step
        return UniformAxis(name, start, stop, step)

    @overload
    def __getitem__(self, index: int) -> Number:
        ...

    @overload
    def __getitem__(self, index: slice) -> UniformAxis:
        ...

    def __getitem__(self, index: Union[int, slice]) -> (
            Union)[Number, UniformAxis, VariableAxis]:
        if type(index) is Ellipsis:
            return self
        elif isinstance(index, slice):
            start = self.start + (index.start or 0) * self.step
            stop = self.stop
            if index.stop is not None:
                stop = self.start + (index.stop * self.step)
            stop = min(self.stop, stop)
            step = (index.step or 1) * self.step
            return UniformAxis(self.name, start, stop, step)
        elif isinstance(index, int):
            _value = self.start + index * self.step
            if _value < self.stop:
                return _value
            raise StopIteration
        elif type(index) in (list, tuple):
            _values = [self[_] for _ in index]
            return VariableAxis(self.name, _values)
        else:
            raise ValueError('Cannot slice axis with {}'.format(type(index)))

    def __len__(self) -> int:
        try:
            return abs(int((self.stop - self.start) / self.step))
        except ValueError:
            return sys.maxsize
        except Exception:
            raise

    def __next__(self) -> Number:
        _value = self.start
        while _value < self.stop:
            yield _value
            _value += self.step

    @property
    def slice(self) -> slice:
        return self.start, self.stop, self.step

    def __repr__(self) -> str:
        return "{0}({1.name!s},{1.start!s},{1.stop!s},{1.step!s})".format(
            self.__class__.__name__, self
        )

    def __str__(self) -> str:
        return "{0.name!s}: {0.start!s}:{0.stop!s}:{0.step!s}".format(self)

    def __eq__(self, other):
        return super().__eq__(other) and \
        (self.start, self.stop, self.step) == (other.start, other.stop, other.step)

    def append(self, axis: Axis):
        """Append another axis"""
        assert self.name == axis.name, 'Cannot append axis "{}" to "{}"'.format(
            axis.name, self.name
        )
        assert self.step == axis.step, 'Cannot append axis "{}" with step {} to step {}'.format(
            axis.name, axis.step, self.step
        )
        self.stop += len(axis) * self.step


class UniformLengthAxis(UniformAxis):
    """Define axis by giving start, length and step (optional).
    Start and step are given in actual values.

    Examples:
        >>> ax = UniformLengthAxis('row', 0, 128)
    """
    n: int

    def __init__(self,
                 name: str,
                 start: Number,
                 n: int,
                 step: Number = 1) -> None:
        super(UniformLengthAxis, self).__init__(name, start, start + n * step, step)
        self.n = abs(n)

    def copy(self,
             name: str = None,
             start: Number = None,
             n: Number = None,
             step: Number = None
             ) -> UniformLengthAxis:
        """Return a copy of the axis, where the length n can be different."""
        name = self.name if name is None else name
        start = self.start if start is None else start
        n = self.n if n is None else n
        step = self.step if step is None else step
        return UniformLengthAxis(name, start, n, step)

    @overload
    def __getitem__(self, index: int) -> Number:
        ...

    @overload
    def __getitem__(self, index: slice) -> UniformLengthAxis:
        ...

    def __getitem__(self, index: Union[int, slice]) -> (
            Union)[Number, UniformLengthAxis, VariableAxis]:
        start, n, step = self.start, self.n, self.step
        if type(index) is Ellipsis:
            return self
        elif isinstance(index, slice):
            start: Number = self.start + (index.start or 0) * self.step
            stop: Number = self.start + (index.stop or self.n) * self.step
            step: Number = (index.step or 1) * self.step
            try:
                n: int = int(round((stop - start) / step))
            except ValueError:
                n: int = sys.maxsize
            except Exception:
                raise
            n = min(self.n, n)
            return UniformLengthAxis(self.name, start, n, step)
        elif isinstance(index, int):
            if index < n:
                return self.start + index * self.step
            raise StopIteration
        elif type(index) in (list, tuple):
            _values = [self[_] for _ in index]
            return VariableAxis(self.name, _values)
        else:
            raise ValueError('Cannot slice axis with {}'.format(type(index)))

    def __len__(self) -> int:
        return self.n

    def __next__(self) -> Number:
        _value = self.start
        for _ in range(self.n):
            yield _value
            _value += self.step

    def __repr__(self) -> str:
        return "{0}({1.name!s},{1.start!s},{1.n!s},{1.step!s})".format(
            self.__class__.__name__, self
        )

    def __str__(self) -> str:
        return "{0.name!s}: {0.n!s}*({0.start!s}:{0.step!s})".format(self)

    def __eq__(self, other):
        return super().__eq__(other) and \
            (self.start, self.n, self.step) == (other.start, other.n, other.step)

    def append(self, axis: Axis):
        """Append another axis"""
        assert self.name == axis.name, 'Cannot append axis "{}" to "{}"'.format(
            axis.name, self.name
        )
        assert self.step == axis.step, 'Cannot append axis "{}" with step {} to step {}'.format(
            axis.name, axis.step, self.step
        )
        self.n += len(axis)


class VariableAxis(Axis):
    """Define axis by giving an array of values.
    values are actual values.

    Examples:
        >>> ax = VariableAxis('time', [0, 1, 4, 9, 11, 13])
    """
    _values: np.ndarray
    step: float

    def __init__(self, name: str, values: Sequence) -> None:
        super(VariableAxis, self).__init__(name)
        if issubclass(type(values[0]), Sequence):
            self._values = np.ndarray(len(values), dtype=np.ndarray)
            for i, value in enumerate(values):
                self._values[i] = np.array(value)
        elif issubclass(type(values[0]), np.ndarray):
            self._values = np.array(values, dtype=np.ndarray)
        else:
            self._values = np.array(values)
        if len(values) < 2:
            self.step = 1
        elif not isinstance(values[0], numbers.Number):
            self.step = None
        else:
            ds: float = values[1] - values[0]
            for i in range(2, len(values)):
                d = values[i] - values[i - 1]
                if abs(d - ds) / ds > 1e-4:
                    ds = None
                    break
            self.step = ds

    def copy(self,
             name: str = None,
             n: Number = None
             ) -> VariableAxis:
        """Return a copy of the axis, where the length n can be different."""
        name = self.name if name is None else name
        n = len(self._values) if n is None else n
        _values = self._values.tolist()
        while len(_values) < n:
            delta = _values[-1] - _values[-2] if len(_values) >= 2 else 1
            _values.append(_values[-1] + delta)
        return VariableAxis(name, _values[:n])

    @overload
    def __getitem__(self, index: int) -> Number:
        ...

    @overload
    def __getitem__(self, index: slice) -> VariableAxis:
        ...

    def __getitem__(self, index: Union[int, slice]) -> Union[Number, VariableAxis]:
        """Slice the axis
        - item: tuple of slice indices
        """
        if type(index) is Ellipsis:
            return self
        elif isinstance(index, slice):
            start = index.start or 0
            stop = index.stop or len(self._values)
            stop = min(len(self._values), stop)
            step = index.step or 1
            return VariableAxis(self.name, self._values[start:stop:step])
        elif isinstance(index, int):
            return self._values[index]
        elif type(index) in (list, tuple):
            return VariableAxis(self.name, self._values[index])
        else:
            raise ValueError('Cannot slice axis with {}'.format(type(index)))

    def __len__(self) -> int:
        return len(self._values)

    def __next__(self) -> Number:
        for _ in self._values:
            yield _

    def __repr__(self) -> str:
        return "{0}({1.name!s},{1._values!r})".format(
            self.__class__.__name__, self
        )

    def __str__(self) -> str:
        return "{0.name!s}: {0._values!s}".format(self)

    def __eq__(self, other):
        return super().__eq__(other) and (self._values == other._values).all()

    def append(self, axis: Axis):
        """Append another axis"""
        assert self.name == axis.name, 'Cannot append axis "{}" to "{}"'.format(
            axis.name, self.name
        )
        values = self._values.tolist()
        values.extend(axis._values)
        self._values = np.array(values)


def to_namedtuple(axes) -> namedtuple:
    """Convert iterable (list, tuple, etc.) to namedtuple of Axes."""

    _keys = []
    for axis in axes:
        _keys.append(axis.name)
    Axes = namedtuple('Axes', _keys)
    return Axes._make(axes)