.. _Installation:

Installation
===============

Typically the base package **imagedata** is installed.
This package provides
the core functionality, in addition to several plugins for formats, archives
and transports:

.. code-block:: bash

    pip install imagedata

The formats include DICOM, ITK, mat files and Nifti-1.

The archives include filesystem and zip archive.

The transports include file, dicom and xnat.

Separate plugins provide support for formats including PostScript (PDF) (read-only) and BIFF. To install:

.. code-block:: bash

    pip install imagedata-format-ps
    pip install imagedata-format-biff

The *imagedata-format-ps* plugin depends on the presense of a local
ghostscript installation.

On Ubuntu, install ghostscript using:

.. code-block:: bash

    apt install ghostscript
