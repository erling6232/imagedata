.. _DeveloperDocumentation:

Developer Documentation
=======================

Writing plugins
---------------

Plugins will be defined by entry points in setup.cfg. See
`imagedata_format_ps <https://github.com/erling6232/imagedata_format_ps>`_
for an example.

Entry points are defined the *setup.cfg*, and shall always be named **imagedata_plugins**:

.. code-block::

    [options.entry_points]
    imagedata_plugins =
        psformat = imagedata_format_ps.psplugin:PSPlugin

The plugin should inherit **AbstractPlugin**, **AbstractArchive** or
**AbstractTransport**.
The inheritance will determine whether the plugin handles format, archive or transport.


Generating distribution archives
--------------------------------

See: https://packaging.python.org/tutorials/packaging-projects/

Update setuptools and wheel:

.. code-block:: bash

    python3 -m pip install --user --upgrade setuptools wheel

Generate distribution packages for the imagedata:

.. code-block:: bash

    make all

Uploading the distribution archives
-----------------------------------

The first thing you’ll need to do is register an account on Test PyPI.
Test PyPI is a separate instance of the package index intended for testing
and experimentation.

Update twine:

.. code-block:: bash

    python3 -m pip install --user --upgrade twine

Run Twine to upload all of the archives under dist:

.. code-block:: bash

    make test_upload

You will be prompted for a username and password. For the username, use
__token__. For the password, use the token value, including the pypi- prefix.

Once uploaded your package should be viewable on TestPyPI, for example,
https://test.pypi.org/project/imagedata

Installing your newly uploaded package
--------------------------------------

You can use pip to install your package and verify that it works. Create a new
virtualenv (see Installing Packages for detailed instructions) and install your
package from TestPyPI:

.. code-block:: bash

    python3 -m pip install --upgrade --index-url https://test.pypi.org/simple/ imagedata

You can test that it was installed correctly by importing the package. Run the Python interpreter (make sure you’re still in your virtualenv):

.. code-block:: console

    python3
    import imagedata

Note that the import package is imagedata regardless of what name you
gave your distribution package in setup.py (in this case,
example-pkg-YOUR-USERNAME-HERE).

Build the documentation
-----------------------

From the package directory:

.. code-block:: bash

    make html

Version numbers
---------------

Bump patch number:

Edit VERSION.txt.

Use version number 1.2.9dev0, ..dev1, etc., for development work.

Use version number 1.2.9rc0, etc., for release candidates.

Use version number 1.2.9 for official release.

To label the github storage with VERSION.txt version:

.. code-block:: bash

    make git

Uploading official release
-----------------------------------

Make sure VERSION.txt has a valid version number.

.. code-block:: bash

    make all
    make git
    make upload

Then go to https://readthedocs.org/projects/imagedata/ and build documentation.