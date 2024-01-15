# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
docs = os.path.dirname(__file__)
root = os.path.dirname(docs)
sys.path.insert(0, root)
sys.path.insert(0, os.path.join(root, 'src'))


# -- Project information -----------------------------------------------------

project = 'imagedata'
copyright = '2013-2023, Erling Andersen, Haukeland University Hospital, Bergen, Norway'
author = 'Erling Andersen'


def get_version():
    """The full version, including alpha/beta/rc tags"""

    version_file = open('../VERSION.txt')
    return version_file.read().strip()


version = get_version()
release = version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.napoleon',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosectionlabel',
    'sphinx_rtd_theme'
]

# Make sure the target is unique
autosectionlabel_prefix_document = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'tests/*']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

napoleon_google_docstring = True
napoleon_use_param = True
napoleon_use_ivar = True
numfig = True


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
#html_theme = 'alabaster'
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
#html_static_path = ['_static']

# -- Extension configuration -------------------------------------------------

# autodoc_mock_imports = ["numpy", "scipy", "scipy.linalg", "numpy.core.multiarray", "nibabel"]
# autodoc_mock_imports = ["scipy", "scipy.linalg", "numpy.core.multiarray", "nibabel"]

# -- Options for todo extension ----------------------------------------------

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True
