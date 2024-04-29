.. _Options:

Options
=================

Some plugins accept options which modify the plugin behaviour.
These options can be provided by adding them to the opts= dictionary
on the Series() read, *e.g.*:

.. code-block:: python

  from imagedata import Series
  a = Series('in_dir', opts={'accept_duplicate_tag': True})


The following plugin options are known at the time of writing:

+-------------------------+-------------------------+-----+-----------------------+
| Plugin                  | DICOM                   |Type | Usage                 |
|                         | Attribute Name          |     |                       |
+=========================+=========================+=====+=======================+
|**DICOMPlugin**                                                                  |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |accept_uneven_slices     |bool |Accept series with     |
|                         |                         |     |uneven number of slices|
+-------------------------+-------------------------+-----+-----------------------+
|read                     |accept_duplicate_tag     |bool |Accept series where tag|
|                         |                         |     |is duplicated          |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |headers_only             |bool |Skip pixel data        |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |correct_acq              |bool |Correct acquisition    |
|                         |                         |     |times for dynamic      |
|                         |                         |     |series                 |
+-------------------------+-------------------------+-----+-----------------------+
|read                     |separate_series          |bool |Sort by Series Instance|
|                         |                         |     |UID.                   |
|                         |                         |     |Used by Collections    |
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
|read                     |<input_order>            |str  |User-defined input     |
|                         |                         |     |order                  |
+-------------------------+-------------------------+-----+-----------------------+
|write                    |keep_uid                 |bool |When True, create new  |
|                         |                         |     |Instance UIDs when     |
|                         |                         |     |writing                |
+-------------------------+-------------------------+-----+-----------------------+
|write                    |window                   |str  |Acquisition Number     |
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
