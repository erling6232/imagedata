"""Axis defines a dimension of an images Series.
"""

from __future__ import annotations
from abc import ABCMeta
from typing import Sequence, Union, overload, SupportsFloat
import numbers
import sys
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

    def __getitem__(self, index: Union[int, slice]) -> Union[Number, UniformAxis]:
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
        self.n = n

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

    def __getitem__(self, index: Union[int, slice]) -> Union[Number, UniformLengthAxis]:
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


class VariableAxis(Axis):
    """Define axis by giving an array of values.
    values are actual values.

    Examples:
        >>> ax = VariableAxis('time', [0, 1, 4, 9, 11, 13])
    """
    values: np.ndarray
    step: float

    def __init__(self, name: str, values: Sequence) -> None:
        super(VariableAxis, self).__init__(name)
        self.values = np.array(values)
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
        n = len(self.values) if n is None else n
        return VariableAxis(name, self.values[:n])

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
            stop = index.stop or len(self.values)
            stop = min(len(self.values), stop)
            step = index.step or 1
            return VariableAxis(self.name, self.values[start:stop:step])
        elif isinstance(index, int):
            return self.values[index]
        else:
            raise ValueError('Cannot slice axis with {}'.format(type(index)))

    def __len__(self) -> int:
        return len(self.values)

    def __next__(self) -> Number:
        for _ in self.values:
            yield _

    def __repr__(self) -> str:
        return "{0}({1.name!s},{1.values!r})".format(
            self.__class__.__name__, self
        )

    def __str__(self) -> str:
        return "{0.name!s}: {0.values!s}".format(self)
