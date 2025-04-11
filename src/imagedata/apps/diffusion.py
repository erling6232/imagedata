"""Extract diffusion MRI parameters.
"""
# Copyright (c) 2013-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import numpy as np
import struct
import warnings
import pandas as pd
from pydicom import Dataset
from typing import Sequence, SupportsFloat
from ..series import Series
from ..formats import NotImageError, CannotSort

Number = type[SupportsFloat]


def get_g_vectors(img):
    """Get diffusion gradient vectors

    Extracting diffusion gradient vectors has been tested on MRI data from some major vendors.

    Args:
        img (imagedata.Series): Series object.
    Returns:
        pd.DataFrame: Diffusion gradient vectors, columns b, z, y, x.
    Raises:
        IndexError: when no gradient vector is present in dataset.
    Examples:
        >>> from imagedata import Series
        >>> from imagedata.apps.diffusion import get_g_vectors
        >>> im = Series('data', 'b', opts={'accept_duplicate_tag': 'True'})
        >>> g = get_g_vectors(im)
        >>> print(g)
                   b         z         y         x
            0      0       NaN       NaN       NaN
            1    500 -0.706399  0.000000  0.707814
            2    500 -0.706399  0.000000 -0.707814
            3    500 -0.706399 -0.707814  0.000000
            4    500  0.707814 -0.706399  0.000000
            5    500  0.000000 -0.707107  0.707107
            6    500  0.000000 -0.707107 -0.707107
            7   1000 -0.706752  0.000000  0.707461
            8   1000 -0.706752  0.000000 -0.707461
            9   1000 -0.706753 -0.707460  0.000000
            10  1000  0.707460 -0.706754  0.000000
            11  1000  0.001414 -0.707106  0.707106
            12  1000  0.001414 -0.707106 -0.707106
            13  2500 -0.706824  0.000000  0.707390

    """

    def get_DICOM_g_vector(ds):
        # Attempt to address standard DICOM attributes
        return ds['DiffusionGradientOrientation'].value

    def get_Siemens_g_vector(ds):
        block = ds.private_block(0x0019, 'SIEMENS MR HEADER')
        return block[0x0e].value

    def get_GEMS_g_vector(ds):
        block = ds.private_block(0x0019, 'GEMS_ACQU_01')
        return [block[0xbb].value, block[0xbc].value, block[0xbd].value]

    _v = []
    # Extract pydicom dataset for first slice and each tag
    _dwi_weighted = False
    for tag in range(img.shape[0]):
        _b = get_b_value(img, tag)
        _dwi_weighted = _dwi_weighted or not np.isnan(_b)
        _G = [np.nan, np.nan, np.nan]
        _ds: Dataset = img.dicomTemplate

        for _method in [get_DICOM_g_vector, get_Siemens_g_vector, get_GEMS_g_vector]:
            try:
                _G = _method(_ds)
                break
            except KeyError:
                pass

        _v.append({'b': _b, 'z': _G[2], 'y': _G[1], 'x': _G[0]})
    if _dwi_weighted:
        return pd.DataFrame(_v)
    else:
        raise IndexError('No b values found in dataset')


def get_b_value(img: Series) -> float:
    """Get diffusion b value

    Extracting diffusion b value has been tested on MRI data from some major vendors.

    Args:
        img (imagedata.Series): Series object.
    Returns:
        float: b value. Returns NaN when no b value is present in dataset.
    """

    # Extract pydicom dataset for given slice and tag
    _ds: Dataset = img.dicomTemplate

    return get_ds_b_value(_ds)


def get_ds_b_vectors(ds: Dataset) -> np.ndarray:
    """Get diffusion b vector from Dataset

    Getting diffusion b vector has been tested on MRI data from some major vendors.

    Args:
        ds: Input dataset
    Returns:
        b vector

    """

    def get_DICOM_b_vector(_ds):
        # Attempt to address standard DICOM attribute
        return np.array(_ds['DiffusionGradientOrientation'].value)

    def get_Siemens_b_vector(_ds):
        raise NotImplementedError('get_Siemens_b_vector not implemented')
        block = _ds.private_block(0x0019, 'SIEMENS MR HEADER')
        return block[0x0c].value

    def get_Siemens_CSA_b_vector(_ds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            import nibabel.nicom.csareader as csa
        try:
            csa_head = csa.get_csa_header(_ds)
        except csa.CSAReadError:
            raise CannotSort("Unable to extract b vector from header.")
        if 'tags' in csa_head and 'DiffusionGradientDirection' in csa_head['tags']:
            bvec = csa_head['tags']['DiffusionGradientDirection']['items']
            return np.array(bvec)
        raise CannotSort("Unable to extract b vector from header.")

    def get_Siemens_E11_b_vector(_ds):
        try:
            block = _ds.private_block(0x0029, 'SIEMENS CSA HEADER')
            im_info = block[0x10].value
            # se_info = block[0x20].value
            if im_info[:4] == b'SV10':
                _h = _get_CSA2_header(im_info)
            else:
                _h = _get_CSA1_header(im_info)
            return np.array(_h['DiffusionGradientDirection'])
        except Exception:
            raise IndexError('Cannot get b vector')

    def get_GEMS_b_vector(_ds):
        raise NotImplementedError('get_GEMS_b_vector not implemented')
        block = _ds.private_block(0x0043, 'GEMS_PARM_01')
        return block[0x39].value

    errmsg = ""
    for _method in [get_DICOM_b_vector,
                    get_Siemens_b_vector, get_Siemens_CSA_b_vector,
                    get_Siemens_E11_b_vector,
                    get_GEMS_b_vector]:
        try:
            return _method(ds)
        except (KeyError, IndexError) as e:
            errmsg = '{}'.format(e)
            pass
        except NotImplementedError:
            pass
    raise IndexError('Cannot get b vector: {}'.format(errmsg))


def get_ds_b_value(ds: Dataset) -> float:
    """Get diffusion b value from Dataset

    Setting diffusion b value has been tested on MRI data from some major vendors.

    Args:
        ds: Input dataset
    Returns:
        b value

    """

    def get_DICOM_b_value(_ds):
        # Attempt to address standard DICOM attribute
        return _ds['DiffusionBValue'].value

    def get_Siemens_b_value(_ds):
        block = _ds.private_block(0x0019, 'SIEMENS MR HEADER')
        return block[0x0c].value

    def get_Siemens_CSA_b_value(_ds):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            import nibabel.nicom.csareader as csa
        try:
            csa_head = csa.get_csa_header(_ds)
        except csa.CSAReadError:
            raise CannotSort("Unable to extract b value from header.")
        if csa_head is None:
            raise CannotSort("Unable to extract b value from header.")
        try:
            value = csa.get_b_value(csa_head)
        except TypeError:
            raise CannotSort("Unable to extract b value from header.")
        return value

    def get_Siemens_E11_b_value(_ds):
        try:
            block = _ds.private_block(0x0029, 'SIEMENS CSA HEADER')
            im_info = block[0x10].value
            # se_info = block[0x20].value
            if im_info[:4] == b'SV10':
                _h = _get_CSA2_header(im_info)
            else:
                _h = _get_CSA1_header(im_info)
            return float(_h['B_value'][0])
        except Exception:
            raise IndexError('Cannot get b value')

    def get_GEMS_b_value(_ds):
        block = _ds.private_block(0x0043, 'GEMS_PARM_01')
        return block[0x39].value

    errmsg = ""
    for _method in [get_DICOM_b_value,
                    get_Siemens_b_value, get_Siemens_CSA_b_value,
                    get_Siemens_E11_b_value,
                    get_GEMS_b_value]:
        try:
            return float(_method(ds))
        except (KeyError, IndexError) as e:
            errmsg = '{}'.format(e)
            pass
    raise IndexError('Cannot get b value: {}'.format(errmsg))


def set_ds_b_value(ds: Dataset, value: Number):
    """Set diffusion b value

    Setting diffusion b value has been tested on MRI data from some major vendors.

    Args:
        ds (pydicom.Dataset): Dataset
        value: b value
    """

    def set_DICOM_b_value(_ds, _value):
        # Attempt to address standard DICOM attribute
        _ds.DiffusionBValue = _value

    def set_Siemens_b_value(_ds, _value):
        block = _ds.private_block(0x0019, 'SIEMENS MR HEADER')
        block[0x0c].value = _value

    def set_GEMS_b_value(_ds, _value):
        block = _ds.private_block(0x0043, 'GEMS_PARM_01')
        block[0x39].value = _value

    for _method in [set_DICOM_b_value, set_Siemens_b_value, set_GEMS_b_value]:
        try:
            _method(ds, value)
            return
        except (KeyError, IndexError):
            pass
    raise IndexError('Cannot set b value')


def set_ds_b_vector(ds: Dataset, value: Sequence[Number]):
    """Set diffusion b vector

    Setting diffusion b vector has been tested on MRI data from some major vendors.

    Args:
        ds (pydicom.Dataset): Dataset
        value: b vector
    """

    def set_DICOM_b_vector(_ds, _value):
        # Attempt to address standard DICOM attribute
        # raise NotImplementedError('set_DICOM_b_vector not implemented')
        _ds.DiffusionGradientOrientation = list(_value)

    def set_Siemens_b_vector(_ds, _value):
        raise NotImplementedError('set_Siemens_b_vector not implemented')
        block = _ds.private_block(0x0019, 'SIEMENS MR HEADER')
        block[0x0c].value = _value

    def set_Siemens_E11_b_vector(_ds, _value):
        raise NotImplementedError('set_Siemens_E11_b_vector not implemented')
        try:
            block = _ds.private_block(0x0029, 'SIEMENS CSA HEADER')
            im_info = block[0x10].value
            # se_info = block[0x20].value
            if im_info[:4] == b'SV10':
                _h = _get_CSA2_header(im_info)
            else:
                _h = _get_CSA1_header(im_info)
            _h['DiffusionGradientDirection'] = _value.tolist()
        except Exception:
            raise IndexError('Cannot set b vector')

    def set_Siemens_CSA_b_vector(_ds, _value):
        raise NotImplementedError('set_Siemens_CSA_b_vector not implemented')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            import nibabel.nicom.csareader as csa
        try:
            csa_head = csa.get_csa_header(_ds)
        except csa.CSAReadError:
            raise CannotSort("Unable to set b vector in header.")
        if 'tags' in csa_head and 'DiffusionGradientDirection' in csa_head['tags']:
            csa_head['tags']['DiffusionGradientDirection']['items'] = _value.tolist()
        raise CannotSort("Unable to set b vector in header.")

    def set_GEMS_b_vector(_ds, _value):
        raise NotImplementedError('set_GEMS_b_vector not implemented')
        block = _ds.private_block(0x0043, 'GEMS_PARM_01')
        block[0x39].value = _value

    _name: str = '{}.{}'.format(__name__, set_ds_b_vector.__name__)

    for _method in [set_Siemens_E11_b_vector, set_Siemens_CSA_b_vector,
                    set_Siemens_b_vector,
                    set_GEMS_b_vector,
                    set_DICOM_b_vector]:
        try:
            _method(ds, value)
            return
        except (KeyError, IndexError) as e:
            errmsg = '{}'.format(e)
            pass
        except NotImplementedError:
            pass
    raise IndexError('Cannot set b vector: {}'.format(errmsg))



def _get_CSA1_header(data):
    values = {}
    try:
        (n_tags, unused2) = struct.unpack('<ii', data[:8])
    except struct.error as e:
        raise NotImageError('{}'.format(e))
    except Exception as e:
        # logging.debug('{}: exception\n{}'.format(_name, e))
        raise NotImageError('{}'.format(e))
    pos = 8
    for t in range(n_tags):
        try:
            (name, vm, vr, syngodt, nitems, xx
             ) = struct.unpack('<64si4s3i', data[pos:pos+84])
            pos += 84
            i = name.find(b'\0')
            name = name[:i]
            name = name.decode("utf-8")
            values[name] = []
            for _item in range(nitems):
                (item_len, xx1, xx2, xx3
                 ) = struct.unpack('<4i', data[pos:pos+16])
                pos += 16
                if item_len > 0:
                    value = data[pos:pos+item_len]
                    value = value.decode("utf-8").split('\0')[0].strip()
                    pos += (item_len // 4) * 4
                    if item_len % 4 > 0:
                        pos += 4
                    values[name].append(value)
        except struct.error as e:
            raise NotImageError('{}'.format(e))
        except Exception as e:
            raise
    return values


def _get_CSA2_header(data):
    values = {}
    try:
        (hdr_id, unused1,
         n_tags, unused2) = struct.unpack('<4siii', data[:16])
    except struct.error as e:
        raise NotImageError('{}'.format(e))
    except Exception as e:
        # logging.debug('{}: exception\n{}'.format(_name, e))
        raise NotImageError('{}'.format(e))
    pos = 16
    for t in range(n_tags):
        try:
            (name, vm, vr, syngodt, nitems, xx
             ) = struct.unpack('<64si4s3i', data[pos:pos+84])
            pos += 84
            i = name.find(b'\0')
            name = name[:i]
            name = name.decode("utf-8")
            values[name] = []
            for _item in range(nitems):
                (item_len, xx1, xx2, xx3
                 ) = struct.unpack('<4i', data[pos:pos+16])
                pos += 16
                if item_len > 0:
                    value = data[pos:pos+item_len]
                    value = value.decode("utf-8").split('\0')[0].strip()
                    pos += (item_len // 4) * 4
                    if item_len % 4 > 0:
                        pos += 4
                    values[name].append(value)
        except struct.error as e:
            raise NotImageError('{}'.format(e))
        except Exception as e:
            raise
    return values
