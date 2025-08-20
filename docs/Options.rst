.. _Options:

Options
=================

Some plugins accept options which modify the plugin behaviour.
These options can be provided by adding them to the opts dictionary
on the Series() read. The can also be given as **kwargs on the object
instantiation. *E.g.*:

.. code-block:: python

  from imagedata import Series
  a = Series('in_dir', opts={'accept_duplicate_tag': True})
  b = Series('in_dir', accept_duplicate_tag=True)


Options can be given to the Cohort(), Patient(), Study() and Series() objects.

The following plugin options are known at the time of writing:

+-------------------------+-------------------------+-----+-----------------------+
| Plugin                  | DICOM                   |Type | Usage                 |
|                         | Attribute Name          |     |                       |
+=========================+=========================+=====+=======================+
|**Collections**                                                                  |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |strict_values            |bool |Require study          |
|                         |                         |     |attributes to match in |
|                         |                         |     |each series/study.     |
|                         |                         |     |Default: True          |
+-------------------------+-------------------------+-----+-----------------------+
|**DICOMPlugin**                                                                  |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |skip_broken_series       |bool |Skip broken series     |
|                         |                         |     |in a study.            |
|                         |                         |     |Do not raise exception.|
+-------------------------+-------------------------+-----+-----------------------+
|read                     |accept_uneven_slices     |bool |Accept series with     |
|                         |                         |     |uneven number of       |
|                         |                         |     |slices.                |
|                         |                         |     |Keep last  image for   |
|                         |                         |     |each position only.    |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |sort_on_slice_location   |bool |Sort stack on slice    |
|                         |                         |     |location, not on       |
|                         |                         |     |distance along normal  |
|                         |                         |     |vector.                |
|                         |                         |     |Default: False         |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |accept_duplicate_tag     |bool |Accept series where tag|
|                         |                         |     |is duplicated.         |
|                         |                         |     |Each image is added    |
|                         |                         |     |to image list at slice |
|                         |                         |     |position.              |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |split_acquisitions       |str  |Split series on        |
|                         |                         |     |DICOM Acquisition      |
|                         |                         |     |Number.                |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |split_echo_numbers       |str  |Split series on        |
|                         |                         |     |DICOM Echo Numbers.    |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |headers_only             |bool |Skip pixel data        |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |correct_acq              |bool |Correct acquisition    |
|                         |                         |     |times for dynamic      |
|                         |                         |     |series                 |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |ignore_series_uid        |str  |Ignore Series Instance |
|                         |                         |     |UID, i.e. do not sort  |
|                         |                         |     |images into different  |
|                         |                         |     |Series                 |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |input_serinsuid          |str  |Filter input files on  |
|                         |                         |     |specified              |
|                         |                         |     |Series Instance UID    |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |input_echo               |int  |Filter input files on  |
|                         |                         |     |specified              |
|                         |                         |     |Echo Numbers           |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |input_acquisition        |int  |Filter input files on  |
|                         |                         |     |specified              |
|                         |                         |     |Acquisition Number     |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |slice_tolerance          |float|Slice distance         |
|                         |                         |     |tolerance when sorting.|
|                         |                         |     |Default: 1E-5          |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |dir_cosine_tolerance     |float|Tolerance for          |
|                         |                         |     |difference in          |
|                         |                         |     |directional cosine     |
|                         |                         |     |tolerance.             |
|                         |                         |     |Default: 0.0           |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |<input_order>            |str  |User-defined input     |
|                         |                         |     |order                  |
+-------------------------+-------------------------+-----+-----------------------+
|write                    |keep_uid                 |bool |When False, create     |
|                         |                         |     |new                    |
|                         |                         |     |Instance UIDs when     |
|                         |                         |     |writing.               |
|                         |                         |     |Default: False         |
+-------------------------+-------------------------+-----+-----------------------+
|write                    |output_sort              |str  |Which tag will sort    |
|                         |                         |     |the output images,     |
|                         |                         |     |'slice' or 'tag'       |
+-------------------------+-------------------------+-----+-----------------------+
|write                    |output_dir               |str  |Store all images in a  |
|                         |                         |     |single or multiple     |
|                         |                         |     |directories, 'single'  |
|                         |                         |     |or 'multi'             |
+-------------------------+-------------------------+-----+-----------------------+
|**ITKPlugin**                                                                    |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |input_sort               |int  |Sort images on         |
|                         |                         |     |SORT_ON_SLICE or       |
|                         |                         |     |SORT_ON_TAG.           |
|                         |                         |     |Useful for image       |
|                         |                         |     |formats that do not    |
|                         |                         |     |provide geometry data  |
+-------------------------+-------------------------+-----+-----------------------+
|write                    |output_sort              |str  |Which tag will sort    |
|                         |                         |     |the output images,     |
|                         |                         |     |'slice' or 'tag'       |
+-------------------------+-------------------------+-----+-----------------------+
|**MatPlugin**                                                                    |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |input_sort               |int  |Sort images on         |
|                         |                         |     |SORT_ON_SLICE or       |
|                         |                         |     |SORT_ON_TAG.           |
|                         |                         |     |Useful for image       |
|                         |                         |     |formats that do not    |
|                         |                         |     |provide geometry data  |
+-------------------------+-------------------------+-----+-----------------------+
|write                    |output_sort              |str  |Which tag will sort    |
|                         |                         |     |the output images,     |
|                         |                         |     |'slice' or 'tag'       |
+-------------------------+-------------------------+-----+-----------------------+
|**NiftiPlugin**                                                                  |
+-------------------------+-------------------------+-----+-----------------------+
|*No options used*        |                         |     |                       |
+-------------------------+-------------------------+-----+-----------------------+
