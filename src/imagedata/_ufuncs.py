"""Series methods.
"""

import numpy as np
# from .series import Series


def max(self, **kwargs):
    if self.dtype == np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return np.array((255, 255, 255),
                        dtype=np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]))
    else:
        return super(Series, self).max(**kwargs)


def nanmax(self, **kwargs):
    if self.dtype == np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return np.array((255, 255, 255),
                        dtype=np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]))
    else:
        return super(Series, self).nanmax(**kwargs)


def min(self, **kwargs):
    if self.dtype == np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return np.array((0, 0, 0),
                        dtype=np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]))
    else:
        return super(Series, self).min(**kwargs)


def nanmin(self, **kwargs):
    if self.dtype == np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return np.array((0, 0, 0),
                        dtype=np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]))
    else:
        return super(Series, self).nanmin(**kwargs)


def __sub__(x1, x2, **kwargs):
    if x1.dtype != np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return super(Series, x1).__sub__(x2, **kwargs)

    out = np.empty_like(x1)
    for _color in ['R', 'G', 'B']:
        out[_color] = np.subtract(x1[_color], x2[_color], **kwargs)
    return out


def multiply(x1, x2, **kwargs):
    if x1.dtype != np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return super(Series, x1).multiply(x2, **kwargs)

    out = np.empty_like(x1)
    for _color in ['R', 'G', 'B']:
        if issubclass(type(x2), Series) and x2.color:
            out[_color] = np.multiply(x1[_color], x2[_color], **kwargs)
        else:
            out[_color] = np.multiply(x1[_color], x2, **kwargs)
    return out


def __mul__(self, other):
    if self.dtype != np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return super(Series, self).__mul__(other)

    out = np.empty_like(self)
    for _color in ['R', 'G', 'B']:
        if issubclass(type(other), Series) and other.color:
            out[_color] = np.multiply(self[_color], other[_color])
        else:
            out[_color] = np.multiply(self[_color], other)
    return out


def __imul__(x1, value):
    if x1.dtype != np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return super(Series, x1).__imul__(value)

    out = np.empty_like(x1)
    for _color in ['R', 'G', 'B']:
        out[_color] = np.__imul__(x1[_color], value)
    return out


def __rmul__(self, other):
    if self.dtype != np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return super(Series, self).__rmul__(other)

    out = np.empty_like(self)
    for _color in ['R', 'G', 'B']:
        if issubclass(type(other), Series) and other.color:
            out[_color] = np.__rmul__(self[_color], other[_color])
        else:
            out[_color] = np.__rmul__(self[_color], other)
    return out


def __rmatmul__(self, other):
    if self.dtype != np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return super(Series, self).__rmatmul__(other)

    out = np.empty_like(self)
    for _color in ['R', 'G', 'B']:
        if issubclass(type(other), Series) and other.color:
            out[_color] = np.__rmatmul__(self[_color], other[_color])
        else:
            out[_color] = np.__rmatmul__(self[_color], other)
    return out


def __matmul__(self, other):
    if self.dtype != np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return super(Series, self).__matmul__(other)

    out = np.empty_like(self)
    for _color in ['R', 'G', 'B']:
        if issubclass(type(other), Series) and other.color:
            out[_color] = np.__matmul__(self[_color], other[_color])
        else:
            out[_color] = np.__matmul__(self[_color], other)
    return out


def __truediv__(self, other):
    if self.dtype != np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return super(Series, self).__truediv__(other)

    out = np.empty_like(self,
                        dtype=np.dtype([('R', np.float32), ('G', np.float32), ('B', np.float32)])
                        )
    for _color in ['R', 'G', 'B']:
        if issubclass(type(other), Series) and other.color:
            out[_color] = np.true_divide(self[_color].view(np.ndarray), other[_color].view(np.ndarray))
        else:
            out[_color] = np.true_divide(self[_color].view(np.ndarray), other)
    return out


def rint(x, **kwargs):
    if x.dtype != np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')]):
        return super(Series, x).rint(**kwargs)

    out = np.empty_like(x)
    for _color in ['R', 'G', 'B']:
        out[_color] = np.rint(x[_color], **kwargs)
    return out

