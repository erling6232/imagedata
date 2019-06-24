imagedata
=========

Read/write medical image data

Python library to read and write image data into numpy arrays.

Handles geometry information between the formats.

The following formats are included:

* DICOM
* Nifti
* ITK (MetaIO)
* Matlab
* PostScript (input only)

Other formats can be added through a plugin architecture.

Simple python3 code
-------------------

A simple example reading a time series from in_dir, and writing it to out_dir::

  from imagedata.series import Series
  a = Series('in_dir', 'time')
  a.write('out_dir')
  
Series fields
-------------

The Series object is inherited from numpy.ndarray, adding a number of useful fields:

Axes
  a.axes defines the unit and size of each dimension of the matrix
  
Addressing
  4D: a[tags, slices, rows, columns]
  
  3D: a[slices, rows, columns]
  
  2D: a[rows, columns]
  
  RGB: a[..., rgb]
  
patientID, patientName, patientBirthDate
  Identifies patient

accessionNumber
  Identifies study

slices
  Returns number of slices
  
spacing
  Returns spacing for each dimension. Units depend on dimension, and could e.g. be mm or sec.
  
tags
  Returns tags for each slice
  
timeline
  Returns time steps for when a time series
  
transformationMatrix
  The transformation matrix to calculate physical coordinates from pixel coordinates

Series instancing
-----------------

From image data file(s):
  a = Series('in_dir')
  
From a list of directories:
  a = Series(['1', '2', '3'])

From a numpy array:
  e = np.eye(128)
  
  a = Series(e)

Series methods
--------------

write
  Write the image data as a Matlab file to out_dir:
  
  a.write('out_dir', formats=['mat'])

slicing
  The image data array can be sliced like numpy.ndarray. The axes will be adjusted accordingly:
  
  b = a[0, ...]
  
  will give a 3D **b** image when **a** is 4D.

Command line usage
------------------

The command line program *image_data* can be used to convert between various image data formats::

  image_data --order time out_dir in_dirs
