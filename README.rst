#########
imagedata
#########

|Docs Badge| |buildstatus|  |coverage| |pypi|


Imagedata is a python library to read and write medical image data into numpy arrays.
Imagedata will handle multi-dimensional data.
In particular, imagedata will read and sort DICOM 3D and 4D series based on
defined tags.
Imagedata will handle geometry information between the formats.

The following formats are included:

* DICOM
* Nifti
* ITK (MetaIO)
* Matlab
* PostScript (input only)

Other formats can be added through a plugin architecture.

Install
-------------------

.. code-block::

    pip install imagedata

Documentation
----------------
See the Documentation_ page for info.

.. _Documentation: https://imagedata.readthedocs.io

Example code
-------------------

A simple example reading two time series from dirA and dirB, and writing their mean to dirMean:

.. code-block:: python

    from imagedata.series import Series
    a = Series('dirA', 'time')
    b = Series('dirB', 'time')
    assert a.shape == b.shape, "Shape of a and b differ"
    # Notice how a and b are treated as numpy arrays
    c = (a + b) / 2
    c.write('dirMean')

Sorting
-------

Sorting of DICOM slices is considered a major task. Imagedata will sort slices into volumes based on slice location.
Volumes may be sorted on a number of DICOM tags:

* 'time': Dynamic time series, sorted on acquisition time
* 'b': Diffusion weighted series, sorted on diffusion b value
* 'fa': Flip angle series, sorted on flip angle
* 'te': Sort on echo time TE

In addition, volumes can be sorted on user defined tags.

Non-DICOM formats usually don't specify the labelling of the 4D data.
In this case, you can specify the sorting manually.

Viewing
-------

A simple viewer. Scroll through the image stack, step through the tags of a 4D dataset.
These operations are possible:

* Window/level adjustment: Move mouse with left key pressed.
* Scroll through slices of an image stack: Mouse scroll wheel, or up/down array keys.
* Step through tags (time, b-values, etc.): Left/right array keys.
* Move through series when many series are displayed: PageUp/PageDown keys.

.. code-block:: python

      # View a Series instance
      a.view()

      # View both a and b Series
      a.view(b)

      # View several Series
      a.view([b, c, d])

Converting data from DICOM and back
-----------------------------------

In many situations you need to process patient data using a tool that do not accept DICOM data.
In order to maintain the coupling to patient data, you may convert your data to e.g. Nifti and back.

Example using the command line utility image_data:

.. code-block:: bash

  image_data --of nifti niftiDir dicomDir
  # Now do your processing on Nifti data in niftiDir/, leaving the result in niftiResult/.

  # Convert the niftiResult back to DICOM, using dicomDir as a template
  image_data --of dicom --template dicomDir dicomResult niftiResult
  # The resulting dicomResult will be a new DICOM series that could be added to a PACS

  # Set series number and series description before transmitting to PACS using DICOM transport
  image_data --sernum 1004 --serdes 'Processed data' \
    dicom://server:104/AETITLE dicomResult

The same example using python code:

.. code-block:: python

  from imagedata.series import Series
  a = Series('dicomDir')
  a.write('niftiDir', formats=['nifti'])   # Explicitly select nifti as output format

  # Now do your processing on Nifti data in niftiDir/, leaving the result in niftiResult/.

  b = Series('niftiResult', template=a)    # Or template='dicomDir'
  b.write('dicomResult')   # Here, DICOM is default output format

  # Set series number and series description before transmitting to PACS using DICOM transport
  b.seriesNumber = 1004
  b.seriesDescription = 'Processed data'
  b.write(' dicom://server:104/AETITLE')

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

seriesNumber, seriesDescription, imageType
  Labels DICOM data

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

.. code-block:: python

  a = Series('in_dir')
  
From a list of directories:

.. code-block:: python

  a = Series(['1', '2', '3'])

From a numpy array:

.. code-block:: python

  e = np.eye(128)
  a = Series(e)

Series methods
--------------

write()
  Write the image data as a Matlab file to out_dir:
  
.. code-block:: python

    a.write('out_dir', formats=['mat'])

slicing
  The image data array can be sliced like numpy.ndarray. The axes will be adjusted accordingly.
  This will give a 3D **b** image when **a** is 4D.

.. code-block:: python

      b = a[0, ...]
  
Archives
--------

The Series object can access image data in a number of **archives**. Some archives are:

Filesystem
  Access files in directories on the local file system.

.. code-block:: python

    a = Series('in_dir')
  
Zip
  Access files inside zip files.
  

.. code-block:: python

  # Read all files inside file.zip:
  a = Series('file.zip')

  # Read named directory inside file.zip:
  b = Series('file.zip?dir_a')
  
  # Write the image data to DICOM files inside newfile.zip:
  b.write('newfile.zip', formats=['dicom'])

Transports
----------

file
  Access local files (default):
  
.. code-block:: python

    a = Series('file:in_dir')
  
dicom
  Access files using DICOM Storage protocols. Currently, writing (implies sending) DICOM images only:
  
.. code-block:: python

    a.write('dicom://server:104/AETITLE')

Command line usage
------------------

The command line program *image_data* can be used to convert between various image data formats:

.. code-block:: bash

  image_data --order time out_dir in_dirs

.. |Docs Badge| image:: https://readthedocs.org/projects/imagedata/badge/
    :alt: Documentation Status
    :scale: 100%
    :target: https://imagedata.readthedocs.io

.. |buildstatus| image:: https://github.com/erling6232/imagedata/actions/workflows/ci.yml/badge.svg
    :target: https://github.com/erling6232/imagedata/actions?query=branch%3Amaster
    :alt: Build Status

.. _buildstatus: https://github.com/erling6232/imagedata/actions

.. |coverage| image:: https://codecov.io/gh/erling6232/imagedata/branch/master/graph/badge.svg?token=GT9KZV2TWT
    :alt: Coverage
    :target: https://codecov.io/gh/erling6232/imagedata

.. |pypi| image:: https://img.shields.io/pypi/v/imagedata.svg
    :target: https://pypi.python.org/pypi/imagedata
    :alt: PyPI Version