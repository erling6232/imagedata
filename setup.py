#!/usr/bin/env python3

#from distutils.core import setup
from setuptools import setup, find_packages

setup(name = 'imagedata',
    #packages = ['imagedata'],
    packages = find_packages(exclude=["contrib", "docs", "tests*"]),
    version = '1.1.5a13',
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
    install_requires = ['pydicom>=1.0.1',
        'itk',
        'vtk',
        'nibabel',
        'numpy >= 1.13',
        'scipy',
        'fs',
        'scandir',
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
    ,
    entry_points={
        'console_scripts': [
            'image_calculator = imagedata.image_data:calculator',
            'image_data = imagedata.image_data:conversion',
            'image_statistics = imagedata.image_data:statistics',
            'timeline = imagedata.image_data:timeline',
            'dicom_dump = imagedata.image_data:dump',
        ]
    }
)
