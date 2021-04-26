.. _DICOMAttributes:

DICOM Attributes
=================

Series properties
-----------------

A handful of DICOM attributes are available directly as Series properties.
The table below lists the available Series properties.

These attributes can be set and interrogated directly from the Series object:

.. code-block:: python

  from imagedata.series import Series
  a = Series('in_dir')
  patientName = a.patientName
  a.seriesDescription = 'DWI MASK'
  a.imageType = ['DERIVED', 'SECONDARY', 'MPR']

+-------------------------+-------------------------+-----------------------+
| Series property         | DICOM                   | Usage                 |
| name                    | Attribute Name          |                       |
+=========================+=========================+=======================+
|**SOP Common Module**                                                      |
+-------------------------+-------------------------+-----------------------+
|SOPClassUID              |SOPClassUID              |str                    |
+-------------------------+-------------------------+-----------------------+
|**Patient Module Attributes**                                              |
+-------------------------+-------------------------+-----------------------+
|patientName              |PatientName              |str                    |
+-------------------------+-------------------------+-----------------------+
|patientID                |PatientID                |str                    |
+-------------------------+-------------------------+-----------------------+
|patientBirthDate         |PatientBirthDate         |str                    |
+-------------------------+-------------------------+-----------------------+
|**General Study Module Attributes**                                        |
+-------------------------+-------------------------+-----------------------+
|studyInstanceUID         |StudyInstanceUID         |str                    |
+-------------------------+-------------------------+-----------------------+
|studyID                  |StudyID                  |str                    |
+-------------------------+-------------------------+-----------------------+
|accessionNumber          |AccessionNumber          |str                    |
+-------------------------+-------------------------+-----------------------+
|**General Series Module Attributes**                                       |
+-------------------------+-------------------------+-----------------------+
|seriesInstanceUID        |SeriesInstanceUID        |str                    |
+-------------------------+-------------------------+-----------------------+
|seriesNumber             |SeriesNumber             |int                    |
+-------------------------+-------------------------+-----------------------+
|seriesDescription        |SeriesDescription        |str                    |
+-------------------------+-------------------------+-----------------------+
|**Frame Of Reference Module Attributes**                                   |
+-------------------------+-------------------------+-----------------------+
|frameOfReferenceUID      |FrameOfReferenceUID      |str                    |
+-------------------------+-------------------------+-----------------------+
|**General Image Module Attributes**                                        |
+-------------------------+-------------------------+-----------------------+
|imageType                |ImageType                |List of str            |
+-------------------------+-------------------------+-----------------------+
|timeline                 |AcquisitionTime          |Read-only numpy array  |
|                         |                         |(can be set using tags)|
+-------------------------+-------------------------+-----------------------+
|bvalues                  |DiffusionBValue          |Read-only numpy array  |
|                         |or propietary            |                       |
+-------------------------+-------------------------+-----------------------+
|**Image Plane Module**                                                     |
+-------------------------+-------------------------+-----------------------+
|spacing                  |PixelSpacing and         |numpy array(dz,dy,dx)  |
|                         |SliceThickness           |in mm                  |
+-------------------------+-------------------------+-----------------------+
|orientation              |ImageOrientationPatient  |numpy array            |
|                         |                         |with 6 elements        |
+-------------------------+-------------------------+-----------------------+
|imagePositions           |ImagePositionPatient     |dict of ImagePositions |
|                         |                         |[z,y,x] of upper left  |
|                         |                         |hand corner (in mm).   |
|                         |                         |dict.keys() are slice  |
|                         |                         |numbers (int)          |
+-------------------------+-------------------------+-----------------------+
|sliceLocations           |SliceLocation            |numpy array (in mm)    |
+-------------------------+-------------------------+-----------------------+
|**Image Pixel Module**                                                     |
+-------------------------+-------------------------+-----------------------+
|color                    |SamplesPerPixel          |bool                   |
+-------------------------+-------------------------+-----------------------+
|photometricInterpretation|PhotometricInterpretation|str                    |
+-------------------------+-------------------------+-----------------------+
|rows                     |Rows                     |Read-only (int)        |
+-------------------------+-------------------------+-----------------------+
|columns                  |Columns                  |Read-only (int)        |
+-------------------------+-------------------------+-----------------------+
|**Composite Attributes**                                                   |
+-------------------------+-------------------------+-----------------------+
|slices                   |                         |Read-only (int)        |
+-------------------------+-------------------------+-----------------------+
|tags                     |Input order tag          |Tags for each slice.   |
|                         |                         |a.tags[slice][tag]     |
+-------------------------+-------------------------+-----------------------+
|axes                     |                         |List of Axis objects   |
+-------------------------+-------------------------+-----------------------+
|transformationMatrix     |                         |numpy array 4x4        |
|                         |                         |in z,y,x order         |
+-------------------------+-------------------------+-----------------------+



Full access to DICOM attributes
-------------------------------

Any DICOM attribute can be set or fetched using the getDicomAttribute()
and setDicomAttribute() methods.

The getDicomAttribute() method will by default fetch the DICOM attribute
for slice 0 and tag 0. When an attribute varies in a series, a
specific slice and/or tag can be specified.

By default the setDicomAttribute() method will set an attribute
for all slices and tags of a series. Alternatively, a specific slice
and/or tag can be targeted.

.. code-block:: python

  # Fetch the MR Repetition Time
  TR = a.getDicomAttribute('RepetitionTime')

  # Fetch the Acquisition Time from tag 9, and duplicate this for tag 10
  acqTime = a.getDicomAttribute('AcquisitionTime', slice=5, tag=9)
  a.setDicomAttribute('AcquisitionTime', acqTime, slice=5, tag=10)
