---

title: 'Imagedata: A Python library to manage medical image data
in a NumPy array subclass `Series`'

tags:
  - DICOM
  - python
  - medical
  - imaging
  - pydicom
  - pynetdicom
  - ITK
  - NIfTI

authors:
  - name: Erling Andersen
    orcid: 0000-0001-8268-1206
    affiliation: "1"

affiliations:
  - name: Haukeland University Hospital, Dept. of Clinical Engineering, N-5021 Bergen, Norway
    index: 1

date: 13 January 2022

bibliography: paper.bib

---

# Summary

`Imagedata` is a python library to read and write medical image data into
NumPy arrays.
`Imagedata` will handle multi-dimensional data.
In particular, imagedata will read, sort and write ``DICOM`` 3D and 4D series based on
defined tags.
Imagedata will handle geometry information between the medical image data formats
like ``DICOM``, ``NIfTI`` and ``ITK``.

# Statement of need

``DICOM`` is the standard image format and protocol when working with
medical images in the clinic. Python has support for reading and writing
DICOM images through the use of python packages, e.g. pydicom or GDCM.
These packages, however, leave the reading and sorting of multiple files
to the user.  Also, they do not easily provide access to medical images
stored in other formats: ``NIfTI`` [@nifti1] and ``ITK`` [@itk2002] to name a few.

When setting up pipelines to process clinical data, patient information
should be maintained throughout to maintain patient safety. If the
process involves ``DICOM`` data only, this requirement is easily fulfilled.
However, some popular image processing systems require formats that do
not maintain patient information. The ability to attach DICOM header
data to these other formats let the user exploit a wider set of image
processing software.

``Imagedata`` extends NumPy arrays [@harris2020array] with ``DICOM``
information and functionality.
Additionally, importing and exporting images to other image formats is available
through a plugin architecture.

``Imagedata`` as a Python package provides functions which supports developing Python
applications, including reading and writing complete image series, and displaying
image series using a simple viewer.

# Architecture

Figures can be included like this:
![Plugin architecture.\label{fig:plugins}](Figure_Architecture.png)
and referenced from text using \autoref{fig:plugins}.

# Features

## Example code

A simple example reading two time series from _dirA_ and _dirB_, and writing their mean to _dirMean_:

~~~
from imagedata.series import Series
a = Series('dirA', 'time')
b = Series('dirB', 'time')
assert a.shape == b.shape, "Shape of a and b differ"

# Notice how a and b are treated as numpy arrays
c = (a + b) / 2
c.write('dirMean')
~~~

## Sorting

Sorting of ``DICOM`` slices is considered a major task. Imagedata will sort slices into volumes based on slice location.
Volumes may be sorted on a number of ``DICOM`` tags:

* 'time': Dynamic time series, sorted on acquisition time
* 'b': Diffusion weighted series, sorted on diffusion _b_ value
* 'fa': Flip angle series, sorted on flip angle
* 'te': Sort on echo time _TE_

In addition, volumes can be sorted on user defined tags.

Some non-DICOM formats don't specify the labelling of the 4D data.
In this case, you can specify the sorting manually.

## Viewing

A simple viewer. Scroll through the image stack, step through the tags of a 4D dataset.
These operations are possible:

* Window/level adjustment: Move mouse with left key pressed.
* Scroll through slices of an image stack: Mouse scroll wheel, or up/down array keys.
* Step through tags (time, b-values, etc.): Left/right array keys.
* Move through series when many series are displayed: PageUp/PageDown keys.

~~~
# View a Series instance
a.show()

# View both a and b Series
a.show(b)

# View several Series
a.show([b, c, d])
~~~

## Converting data from DICOM and back

In many situations you need to process patient data using a tool that do not accept ``DICOM`` data.
In order to maintain the coupling to patient data, you may convert your data to e.g. ``NIfTI`` and back.

Example using the command line utility image_data:

~~~
image_data --of nifti niftiDir dicomDir

# Now do your processing on Nifti data in niftiDir/, leaving the result in niftiResult/.

# Convert the niftiResult back to DICOM, using dicomDir as a template
image_data --of dicom --template dicomDir dicomResult niftiResult
# The resulting dicomResult will be a new DICOM series that could be added to a PACS

# Set series number and series description before transmitting to PACS using DICOM transport
image_data --sernum 1004 --serdes 'Processed data' \
           dicom://server:104/AETITLE dicomResult
~~~

The same example using python code:

~~~
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
~~~

# Acknowledgements

This work is partly funded by a grant from the Regional Health Authority of
Western Norway (Helse Vest RHF) (grant no. 911713).

# References