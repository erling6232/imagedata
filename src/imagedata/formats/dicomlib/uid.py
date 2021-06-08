"""DICOM UID tool."""

# Copyright (c) 2013-2021 Erling Andersen, Haukeland University Hospital

# import os
import os.path
import uuid
import time
import pydicom._uid_dict
# from pydicom.uid import UID

_hostid = None


def get_hostid() -> str:
    """Return hostid of running system.
    """
    global _hostid
    if _hostid is None:
        # _hostid = os.popen('hostid').read().strip()
        _hostid = hex(uuid.getnode())[2:]
    return _hostid


def get_uid() -> str:
    """Generator function which will return a unique UID.
    """
    k = 0
    # hostid = get_hostid()[:-1]
    hostid = get_hostid()
    ihostid = int(hostid, 16)
    my_root = "2.16.578.1.37.1.1.2.%d.%d.%d" % (ihostid, os.getpid(), int(time.time()))
    while True:
        k += 1
        yield "%s.%d" % (my_root, k)


def uid_append_instance(root, num) -> str:
    return root + "." + str(num)


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
        if name == uid[4] or name+"Storage" == uid[4] or name+"ImageStorage" == uid[4]:
            return uid
    raise ValueError("Storage class {} is unknown.".format(name))
