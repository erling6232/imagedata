"""Extract diffusion MRI parameters.
"""

# Copyright (c) 2013-2023 Erling Andersen, Haukeland University Hospital, Bergen, Norway


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

    def get_Siemens_g_vector(ds):
        block = ds.private_block(0x0019, 'SIEMENS MR HEADER')
        return block[0x0e].value

    def get_GEMS_g_vector(ds):
        block = ds.private_block(0x0019, 'GEMS_ACQU_01')
        return [block[0xbb].value, block[0xbc].value, block[0xbd].value]

    v = []
    # Extract pydicom dataset for first slice and each tag
    _slice = 0
    for tag in range(img.shape[0]):
        G = [0, 0, 0]
        ds=img.DicomHeaderDict[_slice][tag][2]

        # Attempt to address standard DICOM attributes
        if 'DiffusionGradientOrientation' in ds:
            G = ds['DiffusionGradientOrientation'].value

        # Special solutions for Siemens and GE systems
        try:
            G = get_Siemens_g_vector(ds)
        except KeyError:
            try:
                G = get_GEMS_g_vector(ds)
            except KeyError:
                pass

        v.append({'b': get_b_value(img, tag), 'z': G[2], 'y': G[1], 'x': G[0]})
    return pd.DataFrame(v)

def get_b_value(img, tag=0):
    """Get diffusion b value

    Args:
        img: Series object
    Returns:
        b: float
    Raises:
        IndexError: when no diffusion b value is present in dataset
    """

    def get_Siemens_b_value(ds):
        block = ds.private_block(0x0019, 'SIEMENS MR HEADER')
        return block[0x0c].value

    def get_GEMS_b_value(ds):
        block = ds.private_block(0x0043, 'GEMS_PARM_01')
        return block[0x39].value

    # Extract pydicom dataset for given slice and tag
    _slice = 0
    ds=img.DicomHeaderDict[_slice][tag][2]

    # Attempt to address standard DICOM attribute
    if 'DiffusionBValue' in ds:
        return ds['DiffusionBValue'].value

    # Special solutions for Siemens and GE systems
    try:
        return float(get_Siemens_b_value(ds))
    except KeyError:
        try:
            return get_GEMS_b_value(ds)
        except KeyError:
            pass

    return 0.0
