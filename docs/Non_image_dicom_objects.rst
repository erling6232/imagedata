.. _Non_image_dicom_objects:

Non-image DICOM objects
=======================

DICOM Image objects have pixel data and are stored as NumPy.ndarray.
However, non-image objects (*e.g.* structured reports) typically has no pixel data
associated.
Such objects can be read using Series(), Study(), etc.
The resulting Series object will have an empty pixel array.
The DICOM Datasets will be present in the `header.datasets` list.

`Imagedata` will not interpret the datasets.
They are stored and available for application program to use.
