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

from imagedata.series import Series
a = Series('in_dir', 'time')
a.write('out_dir')

Command line usage
------------------

image_data --order time out_dir in_dirs
