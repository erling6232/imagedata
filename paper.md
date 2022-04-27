---
title: 'Imagedata: A Python library to handle medical image data in NumPy array subclass Series'

tags:
  - dicom
  - python
  - medical imaging
  - pydicom
  - pynetdicom
  - itk
  - nifti

authors:
  - name: Erling Andersen
    orcid: 0000-0001-8268-1206
    affiliation: "1, 2"

affiliations:
  - name: Haukeland University Hospital, Dept. of Clinical Engineering, N-5021 Bergen, Norway
    index: 1
  - name: Mohn Medical Imaging and Visualization Centre, Haukeland University Hospital, Dept. of Radiology, N-5021 Bergen, Norway
    index: 2

date: 14 January 2022
bibliography: paper.bib
---

# Summary

`Imagedata` is a python library to read and write medical image data into
`Series` objects.
In particular, imagedata will read, sort and write DICOM\textsuperscript{\textregistered}
3D and 4D series based on defined attributes.
As far as possible, imagedata will handle geometry information between
the medical image data formats
like DICOM, NIfTI [@nifti1] and ITK [@itk2002].

Imagedata provides a Series class inheriting the `numpy.ndarray` class [@harris2020array],
adding DICOM data structures.
Plugins provide functions to import and export DICOM and other data formats.
The DICOM plugin can read complete series, sorting the data as requested into
multidimensional arrays.
Input and output data can be accessed on various locations, including local files,
DICOM servers and XNAT servers [@marcus2007].
The Series class enables NumPy and derived libraries (like SciPy [@2020SciPy-NMeth])
to work on medical images, simplifying input and output.

A feature is the conversion between different image formats.
_E.g._, a pipeline based on a clinical DICOM series can be converted to NIfTI,
processed by some NIfTI-based tool (_e.g._ FSL [@FSL2004]).
Finally, the result can be converted back to DICOM and stored as a new series in
PACS (Picture Archive and Communication system).

A viewer is included, allowing the display of a stack of images,
including modifying window width and centre, and scrolling through 3D and 4D image stacks.
A region of interest (ROI) can be drawn, resulting in a mask as a NumPy ndarray,
or as an outline.

# Statement of need

DICOM is the standard image format and protocol when working with clinical
medical images in a hospital.
In tomographic imaging, the legacy DICOM formats
like computed tomography (CT) and magnetic resonance (MR)
information object definitions (IOD),
are in common use.
These formats store slices file by file, leaving the sorting of the
files to the user.
The more recent enhanced formats which can accommodate a complete 3D or 4D acquisition in
one file, are only slowly adopted by manufacturers of medical equipment.

Working with legacy DICOM medical images in python can be accomplished using libraries
like `pydicom` [@darcy_mason_2021_5543955], `GDCM` [@GDCMReferenceManual],
`NiBabel` [@nibabel2020] or `ITK`.
Pydicom and GDCM are native DICOM libraries. As such, they do not
provide access to medical images stored in other formats.
NiBabel and ITK are mostly focused on NIfTI and ITK MetaIO image formats, respectively.
These formats are popular in research tools. However, DICOM support is rudimentary.
All these libraries typically leave the sorting of legacy DICOM image files to the user.

`Highdicom` [@bridge2021highdicom] focus on parametric maps, annotations
and segmentations, using enhanced DICOM images.
Highdicom does an excellent job of promoting the enhanced DICOM standards,
including storage of `boolean` and floating-point data.
The handling of legacy DICOM objects are left to pydicom. 

NumPy ndarrays is the data object of choice for numerical computations in Python. 
Imagedata extends NumPy arrays with DICOM
information and functionality.
Additionally, importing and exporting images to other image formats is available
through the plugin architecture.

When setting up pipelines to process clinical data, patient information
should be maintained throughout to ensure patient safety. If the
process involves DICOM data only, this requirement is easily fulfilled.
However, some popular image processing systems require formats that do
not maintain patient information. The ability to attach DICOM metadata
to these other formats let the user exploit a wider set of image
processing software.

`Imagedata` builds on several of these libraries,
attempting to solve the problem of sorting legacy DICOM images,
providing NumPy ndarrays, and accessing medical images in various formats.

# Architecture

The `Series` class is a `numpy.ndarray` subclass. 
A Series object is instantiated from an image source, either from input files, 
from a server connection, or from an ndarray.
DICOM metadata is handled by a `Header` class, which also maintains an `Axes` class
defining the axes of the array dimensions.

Handling specific image data formats are done by `Formats` plugins,
while `Archives` plugins give access to files stored both
in the filesystem and in compressed archives.
The `Transports` plugins let the user access networked resources given by a _URL_.
See the plugin architecture and main classes in \autoref{fig:plugins}.
_E.g._, an xnat:// _URL_ will employ the `XnatTransport` plugin
to fetch a compressed zip archive,
and the `ZipfileArchive` will extract individual files from the archive.

Plugins are defined using python's `entry_point` [@pythonEntryPoints] mechanism.
The naming convention requires any plugin to advertise itself on the `imagedata_plugins` list.

![Plugin architecture and main classes of the imagedata package.
The Series class is subclassed from numpy.ndarray.
Handling specific image data formats are done by Formats plugins,
while Archives plugins give access to files stored both in the filesystem
and in compressed archives.
The Transports plugins let the user access networked resources given
by a _URL_.\label{fig:plugins}](docs/Plugin_Architecture.png){ width=75% }


# Examples

In this section, we demonstrate the use of imagedata in the python language.
In addition, there is a console application `image_data` which comes in handy when
the sole purpose is to convert and store an image dataset from one format
to another.

## Compute mean of two datasets

A basic example reading two time series from folders _dirA_ and _dirB_,
and writing their mean to folder _dirMean_.
The format of the input data is automatically detected, and is not specified:

~~~
from imagedata.series import Series
a = Series('dirA', 'time')
b = Series('dirB', 'time')
assert a.shape == b.shape, "Shape of a and b differ"

# Notice how series a and b are treated as NumPy arrays
c = (a + b) / 2
c.write('dirMean')
~~~

## Sorting

Sorting of DICOM slices is an important feature.
Imagedata will sort slices into volumes based on slice location.
Volumes may be sorted on various DICOM attributes:

* 'time': Dynamic time series, sorted on acquisition time
* 'b': Diffusion weighted series, sorted on diffusion _b_ value
* 'fa': Flip angle series, sorted on MR flip angle
* 'te': Sort on MR echo time _TE_

In addition, volumes can be sorted on user-defined attributes.

Some non-DICOM formats do not specify the labelling of 4D data.
In this case, the sorting can be specified manually.

## Slicing

Like ndarray, the Series object can be sliced.
The imagedata package attempts to maintain the geometry of the sliced data.

~~~
>>> ...
>>> # Extract slice no. 5
>>> slice5 = si[5,…]
>>> slice5.sliceLocations
array(6.8)
>>> # Save slice 5 to slice5/ folder
>>> slice5.write(’slice5/’)
~~~

## Viewing

A viewer based on `matplotlib imshow` [@Hunter:2007] is included.
The viewer lets the user scroll through the image stack,
and step through the tags of a 4D dataset.
These operations are implemented:

* Read-out voxel value: Move mouse over.
* Window/level adjustment: Move mouse with left key pressed.
* Scroll through slices of an image stack: Mouse scroll wheel, or up/down array keys.
* Step through tags (time, b-values, etc.): Left/right array keys.
* Page through series in a multi-series display: PageUp/PageDown keys.

~~~
# View a Series instance
a.show()

# View both a and b Series
a.show(b)

# View several Series
a.show([b, c, d])
~~~

## Draw a region of interest

A region of interest (ROI) can be drawn, producing a mask as a NumPy ndarray. 
This example will obtain a mask image `segment`,
convert the original grayscale image into a corresponding RGB image,
and mask the green and blue color bands inside the ROI.

~~~
from imagedata.series import Series

T2 = Series('801_Obl T2 TSE HR SENSE/')
segment = T2.get_roi()

# Convert grayscale image to RGB image
T2rgb = T2.to_rgb()
segment_indices = segment == 1

# Clear green and blue components inside segment,
# leaving the red component
T2rgb[segment_indices, 1:] = 0

# Display final image where pixels inside the ROI are red
T2rgb.show()
~~~

## Converting data from DICOM and back

Some workflows process patient data using a tool that do not accept DICOM data.
In order to maintain the coupling to patient data, the data can be converted to
_e.g._ NIfTI and back.

### Example using the console application image_data

~~~
# Original DICOM data in dicomDir/
image_data --of nifti niftiDir dicomDir

# Now process on Nifti data in niftiDir/,
# ...
# leaving the result in niftiResult/.

# Convert the niftiResult back to DICOM,
# using dicomDir as a template
image_data --of dicom --template dicomDir dicomResult niftiResult

# The resulting dicomResult will be a new DICOM series
# that could be added to a PACS

# Set series number and series description before 
# transmitting to PACS using DICOM transport
image_data --sernum 1004 --serdes 'Processed data' \
           dicom://server:104/AETITLE dicomResult
~~~

### Example using python code

This code will store the Series data in a NIfTI format, letting some
NIfTI-dependent code produce a result in _niftiResult_.
This NIfTI dataset is loaded into a Series object, using the original
DICOM data as template to maintain patient and study metadata.
Finally, the new dataset is sent to a DICOM server
using the DICOM protocol.

~~~
import tempfile
from imagedata.series import Series
a = Series('dicomDir')

# Prepare temporary storage for NIfTI data
with tempfile.TemporaryDirectory() as niftiDir, \
    tempfile.TemporaryDirectory() as niftiResult:
    # Explicitly select nifti as output format
    a.write(niftiDir, formats=['nifti'])

    # Now process on NIfTI data in niftiDir
    # ...
    # leaving the result in niftiResult

    # Load the NIfTI data, using original Series `a' as template
    b = Series(niftiResult, template=a)

# Set series number and series description before
# transmitting to PACS using DICOM transport
b.seriesNumber = 1004
b.seriesDescription = 'Processed data'
b.write('dicom://server:104/AETITLE')
~~~

# Acknowledgements

This work is partly funded by a grant from the Regional Health Authority of
Western Norway (Helse Vest RHF) (grant no. 911745).
The authors want to thank Erlend Hodneland for valuable discussions and feedback.

# References