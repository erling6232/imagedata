"""Extract diffusion MRI parameters.
"""
# Copyright (c) 2013-2023 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import numpy as np
import pandas as pd


def get_g_vectors(img):
    """Get diffusion gradient vectors

    Args:
        img (Series): Series object.
    Returns:
        pd.DataFrame: Diffusion gradient vectors, columns b, z, y, x.
    Raises:
        IndexError: when no gradient vector is present in dataset.
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
    _slice = 0
    _dwi_weighted = False
    for tag in range(img.shape[0]):
        _b = get_b_value(img, tag)
        _dwi_weighted = _dwi_weighted or not np.isnan(_b)
        _G = [np.nan, np.nan, np.nan]
        _ds = img.DicomHeaderDict[_slice][tag][2]

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


def get_b_value(img, tag=0):
    """Get diffusion b value

    Args:
        img: Series object.
    Returns:
        float: b value. Returns NaN when no b value is present in dataset.
    """

    def get_DICOM_b_value(ds):
        # Attempt to address standard DICOM attribute
        return ds['DiffusionBValue'].value

    def get_Siemens_b_value(ds):
        block = ds.private_block(0x0019, 'SIEMENS MR HEADER')
        return block[0x0c].value

    def get_GEMS_b_value(ds):
        block = ds.private_block(0x0043, 'GEMS_PARM_01')
        return block[0x39].value

    # Extract pydicom dataset for given slice and tag
    _slice = 0
    _ds = img.DicomHeaderDict[_slice][tag][2]

    for _method in [get_DICOM_b_value, get_Siemens_b_value, get_GEMS_b_value]:
        try:
            return _method(_ds)
        except KeyError:
            pass

    return np.nan
