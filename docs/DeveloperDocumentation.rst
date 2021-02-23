.. _DeveloperDocumentation:

Developer Documentation
=======================

Generating distribution archives
--------------------------------

See: https://packaging.python.org/tutorials/packaging-projects/

Update setuptools and wheel:

.. code-block:: bash

    python3 -m pip install --user --upgrade setuptools wheel

Generate distribution packages for the imagedata:

.. code-block:: bash

    python3 setup.py sdist bdist_wheel
    ls -l dist/

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

    python3 -m twine upload --repository testpypi dist/*

You will be prompted for a username and password. For the username, use
__token__. For the password, use the token value, including the pypi- prefix.

Once uploaded your package should be viewable on TestPyPI, for example,
https://test.pypi.org/project/example-pkg-YOUR-USERNAME-HERE

Installing your newly uploaded package
--------------------------------------

You can use pip to install your package and verify that it works. Create a new
virtualenv (see Installing Packages for detailed instructions) and install your
package from TestPyPI:

.. code-block:: bash

    python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps example-pkg-YOUR-USERNAME-HERE

You can test that it was installed correctly by importing the package. Run the Python interpreter (make sure you’re still in your virtualenv):

.. code-block:: console

    python3
    import example_pkg

Note that the import package is example_pkg regardless of what name you
gave your distribution package in setup.py (in this case,
example-pkg-YOUR-USERNAME-HERE).

Build the documentation
-----------------------

From the package directory:

.. code-block:: bash

    python3 setup.py build_sphinx

Version numbers
---------------

Bump patch number:

.. code-block:: bash

    python3 -m incremental.update imagedata --patch

Set a new version:

.. code-block:: bash

    python3 -m incremental.update imagedata --newversion=<version>

Set as release candidate:

.. code-block:: bash

    python3 -m incremental.update imagedata --patch --rc

Set as final release:

.. code-block:: bash

    python3 -m incremental.update imagedata
