.. _DeveloperDocumentation:

Developer Documentation
=======================

Build the documentation
-----------------------

From the package directory::
    python3 setup.py build_sphinx

Version numbers
---------------

Bump patch number::
    python3 -m incremental.update imagedata --patch

Set a new version::
    python3 -m incremental.update imagedata --newversion=<version>

Set as release candidate::
    python3 -m incremental.update imagedata --patch --rc

Set as final release::
    python3 -m incremental.update imagedata
