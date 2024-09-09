"""DICOM UID tool."""

# Copyright (c) 2013-2024 Erling Andersen, Haukeland University Hospital

import pydicom._uid_dict


def get_uid_for_storage_class(name) -> str:
    """Return DICOM UID for given DICOM Storage Class

    Args:
        name: name or UID of DICOM storage class (str)
    Returns:
        DICOM UID
    Raises:
        ValueError: When name does not match a SOP Class
    """
    if name == "SC":
        name = "SecondaryCaptureImageStorage"
    for uid in pydicom._uid_dict.UID_dictionary.keys():
        if name == uid:
            return uid
        if name == uid[4] or name + "Storage" == uid[4] or name + "ImageStorage" == uid[4]:
            return uid
    raise ValueError("Storage class {} is unknown.".format(name))
