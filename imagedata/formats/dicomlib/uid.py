#!/usr/bin/env python

"""DICOM UID tool."""

# Copyright (c) 2013 Erling Andersen, Haukeland University Hospital

import os, os.path
import time

def get_hostid():
	import tempfile
	tmpnamt=tempfile.mkstemp()
	tmpnam=tmpnamt[1]
	os.system("hostid >| "+tmpnam)
	f = open(tmpnam, 'r')
	hostid = f.readline()
	f.close()
	os.remove(tmpnam)
	return(hostid)

def get_uid():
	hostid=get_hostid()[:-1]
	ihostid=int(hostid, 16)
	return "2.16.578.1.37.1.1.2.%d.%d.%d" % (ihostid, os.getpid(), int(time.time()))

def uid_append_instance(root, num):
	return root+"."+str(num)
