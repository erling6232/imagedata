.. imagedata documentation master file, created by
   sphinx-quickstart on Fri Jun 26 13:35:42 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Imagedata Documentation
=======================

Imagedata is a python library to read and write medical image data into numpy arrays.
Imagedata will handle multi-dimensional data.
In particular, imagedata will read and sort DICOM 3D and 4D series based on
defined tags.
Imagedata will handle geometry information between the formats.

Imagedata initially supports the following formats:

* DICOM
* Nifti
* ITK (MetaIO)
* Matlab
* PostScript (input only)

Other formats can be added through a plugin architecture.

A simple example reading a time series from in_dir, and writing it to out_dir:

.. code-block:: python

  from imagedata.series import Series
  a = Series('in_dir', 'time')
  a.write('out_dir')

The :ref:`Getting Started <GettingStarted>` section explains how to install and
use imagedata.

*Loading image data with this imagedata Python package simplify image
processing pipelines both in native Python, and when stitching together
modules that require different image file formats. This is especially
important when setting up pipelines for clinical data where patient
information matters.*

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   GettingStarted
   Tutorial
   Introduction
   CommandLine
   Plugins
   APIReference
   DeveloperDocumentation

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
