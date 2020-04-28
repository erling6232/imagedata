#!/usr/bin/env python

"""DICOM UID tool."""

# Copyright (c) 2013 Erling Andersen, Haukeland University Hospital

import os
import os.path
import time

_hostid = None


def get_hostid():
    """
    import tempfile
    tmpnamt=tempfile.mkstemp()
    tmpnam=tmpnamt[1]
    os.system("hostid >| "+tmpnam)
    with open(tmpnam, 'r') as f:
        hostid = f.readline()
    os.remove(tmpnam)
    return(hostid)
    """
    global _hostid
    if _hostid is None:
        _hostid = os.popen('hostid').read().strip()
    return _hostid


def get_uid():
    """Generator function which will return a unique UID"""

    k = 0
    hostid = get_hostid()[:-1]
    ihostid = int(hostid, 16)
    my_root = "2.16.578.1.37.1.1.2.%d.%d.%d" % (ihostid, os.getpid(), int(time.time()))
    while True:
        k += 1
        yield "%s.%d" % (my_root, k)


def uid_append_instance(root, num):
    return root + "." + str(num)
