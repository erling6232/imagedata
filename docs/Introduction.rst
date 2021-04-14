.. _Introduction:

Introduction
===============

DICOM is the standard image format and protocol when working with
medical images in the clinic. Python has support for reading and writing
DICOM images through the use of python packages, e.g. pydicom or GDCM.
These packages, however, leave the reading and sorting of multiple files
to the user.  Also, they do not easily provide access to medical images
stored in other formats: NIfTI and ITK to name a few.
When setting up pipelines to process clinical data, patient information
should be maintained throughout to maintain patient safety. If the
process involves DICOM data only, this requirement is easily fulfilled.
However, some popular image processing systems require formats that do
not maintain patient information. The ability to attach DICOM header
data to these other formats let the user exploit a wider set of image
processing software.

Imagedata is a python package that allows working with medical image
data both in DICOM and other image format, converting between formats
when needed.

Series Class
=============

The Series class holds an image series (3D or multi-dimensional) loaded
or produced in Python. Figure 1 shows the steps to instantiate an image
series :

.. code-block:: console

   >>> from imagedata.series import Series
   >>> si = Series(’dicom/volume/’)
   >>> print(si.shape, si.dtype)
   (16, 160, 160) uint16
   >>> si.spacing
   Array([2.    ,  1.3125,  1.3125])
   >>> si.sliceLocations
   array([ -3.16799537,  -1.16799537,   0.83200463,   2.83200463,
            4.83200463,   6.83200463,   8.83200463,  10.83200463,
           12.83200463,  14.83200463,  16.83200463,  18.83200463,
           20.83200463,  22.83200463,  24.83200463,  26.83200463])

   >>> # Save series in output/ directory, original format
   >>> si.input_format
   ’dicom’
   >>> si.write(’output/’)
   >>> # Save series in ITK format (MHA format by default)
   >>> si.write(’itk_output/’, formats=[’itk’])

The Series class is subclassed from numpy.ndarray, and as such inherits
the methods of ndarray. Examples include the shape and dtype
attributes. In addition, Series objects add a few attributes, like
spacing, sliceLocations, orientation, seriesNumber and
seriesDescription. These attributes map directly to the corresponding
DICOM attributes. When loading non-DICOM data, imagedata will load
those attributes supported by the image format.

The addressing of the Series array is row-major (also known as C order),
which leads to the most efficient NumPy processing. E.g.  (slice, row,
column) in the 3D case, and (tag, slice, row, column) in the 4D case.
This is contrary to e.g. MATLAB (The MathWorks, Inc.) which addresses
arrays in column-major (Fortran) order. Also note that NumPy indices
start at zero, while MATLAB indices start at one.

The tag index for a 4D series indicates any attribute that distinguishes
the 3D volumes, like time for a dynamic scan, b value for a diffusion
weighted scan, flip angle or echo time for contrast weighted scans.

In a pipeline where non-DICOM data are converted to DICOM data, a DICOM
template is required to fill in missing attribute values.  The original
input DICOM images can usually be applied as template, such that a PACS
will add the new images as a new series to an existing patient and
study. Figure 1 shows an example where the loaded DICOM data are written
to files in both original (DICOM) and ITK’s MetaImage formats.

Like the ndarray, the Series object can be sliced. The imagedata package
attempts to maintain the geometry of the sliced data. The example in
Figure 2 extracts slice 5, showing that the sliceLocations attribute has
been adjusted. Next, slice 5 is stored to disk:

.. code-block:: console

   >>> # Extract _slice no. 5
   >>> slice5 = si[5,…]
   >>> slice5.sliceLocations
   array(6.8320046343748)
   >>> # Save _slice 5 to slice5/ directory
   >>> slice5.write(’slice5/’)
