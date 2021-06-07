.. _Input_output:

Input/output
===============

Local files
--------------

Input:
    Read a file by specifying a path name, or read a directory by giving a directory name.

    * file.mha
    * /home/dicom/file.mha
    * file:///home/dicom/file.mha
    * dir
    * /home/dicom/dir
    * file:///home/dicom/dir

When reading a directory, every file inside will be scanned and combined into a Series instance.
See :ref:`Sorting <Sorting>` for details on sorting images.

Write:
    When writing, the provided path name will be taken as a (new) directory, and images file(s)
    are added to that directory.

ZIP file
----------

A zip file can contain image files.

Read:
    Access files in an existing zip file. Selected files or directories can be specified:

    * file.zip
    * file.zip?time

    The latter example will read the time/ directory in the zip file.

Write:
    Files can be added to a zip file. When the zip file does not exist, it will be created prior to writing.

XNAT
----

Image data in XNAT can be accessed. Reading data will fetch the image data and construct a Series instance.
Writing a Series instance will create a new scan in the XNAT subject and experiment.

.. code-block:: python

    img = Series('xnat://xnat.server/project/subject/experiment/scan')
    img.seriesNumber = 9999
    img.seriesDescription = 'New Series'
    img.write('xnat://xnat.server/project/subject/experiment')

In order to access the XNAT server, you typically need to authenticate.
Handling authenticating is outside the scope of imagedata.
See https://xnat.readthedocs.io/en/latest/static/tutorial.html#credentials for details.
On a Linux system, authentication can be handled by adding

.. code-block::

    machine xnat.server
     login <username>
     password <pwd>

to a ~/.netrc file. Remember to restrict access to yourself only:

.. code-block:: bash

    chmod 0600 ~/.netrc

DICOM
-----

Read:
   DICOM data can be read from a DICOM archive which supports the C-GET method.

   A DICOM series can be specified by providing patient ID, study and series instance UIDs.
   Instead of study instance UID, the accession number can be used.
   A series can be specified by providing the series number instead of series instance UID.

   Consult the DICOM Conformance Statement of the DICOM archive to determine whether
   C-GET is supported.

.. code-block:: python

        patID = '123456'
        stuInsUID = '1.2.3.4'
        serInsUID = '1.2.3.4.5'
        # A Series can be specified by providing Study and Series instance UID
        si1 = Series('dicom://localhost:11112/Temp/{}/{}/{}'.format(
            patID, stuInsUID, serInsUID
        ))
        print(si1.shape, si1.spacing, si1.patientName, si1.accessionNumber, si1.seriesNumber)

        accno = '98765'
        serNum = 4
        # A Series can also be specified by providing accession number and series number
        si2 = Series('dicom://localhost:11112/Temp/{}/{}/{}'.format(
            patID, accno, serNum
        ))

Write:
   A Series instance can be sent to a DICOM archive using its write() method.
   There is no need to specify a path name on the DICOM archive.
   The image will be associated with the correct patient and study, according to the study, series and instance UIDs
   present in the Series instance.

.. code-block:: python

    # Write Series instance to DICOM Archive
    # DICOM Archive address: dicom.server
    # DICOM Archive port number: 11112
    # DICOM Archive application entity title: AET
    img.write('dicom://dicom.server:11112/AET', formats=['dicom'])

It is mandatory that a full DICOM header is associated with the Series instance.
This will be the case when the Series instance was loaded from DICOM data in the first place.
When no DICOM header is present, a DICOM template can be used to construct a complete DICOM header:

.. code-block:: python

    # Fetch PostScript file, add DICOM template, and send to DICOM archive
    img = Series('postscript.ps', template='dicom/data')
    img.write('dicom://dicom.server:11112/AET', formats=['dicom'])
