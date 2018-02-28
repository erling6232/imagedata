#!/usr/bin/env python

from distutils.core import setup

setup(name = 'imagedata',
    packages = ['imagedata'],
    version = '1.0.0a1',
    description = 'Read/write medical image data',
    author = 'Erling Andersen',
    author_email = 'Erling.Andersen@Helse-Bergen.NO',
    url = 'https://github.com/erling6232/imagedata',
    license = 'MIT',
    keywords = 'dicom python medical imaging',
    classifiers = [
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
    ],
    requires = ['pydicom',
        'itk',
        'vtk',
        'nibabel',
        'numpy',
        'pathfinder',
    ],
    tests_require = ['tests'],
    long_description = """\
    # imagedata

    Read/write medical image data.

    Python library to read and write image data into numpy arrays.

    Handles geometry information between the formats.

    The following formats are included:
    * DICOM
    * Nifti
    * ITK (MetaIO)
    """
)
