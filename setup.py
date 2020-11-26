#!/usr/bin/env python3

import setuptools
import os

# this is only necessary when not using setuptools/distribute
from sphinx.setup_command import BuildDoc
cmdclass = {'build_sphinx': BuildDoc}

# Version
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
VERSION_FILE = os.path.join(BASE_DIR, 'imagedata', '_version.py')
with open(VERSION_FILE) as fp:
    exec(fp.read())

# Get the long description from the README file
with open(os.path.join(BASE_DIR, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

name = 'imagedata'
version = __version__.public()
release = version
setuptools.setup(name = name,
    packages = setuptools.find_packages(exclude=['contrib', 'docs', 'tests', 'data']),
    #version = __version__.public(),
    use_incremental=True,
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
    setup_requires=['incremental', 'Sphinx'],
    install_requires = [
        'pydicom>=1.0.1',
        'pynetdicom>=1.2.0',
        'itk>=5.0',
        'nibabel',
        'numpy >= 1.13',
        'scipy',
        'fs',
        'python-magic',
        'incremental',
        'ghostscript',
        'imageio',
        'scandir',
        'pathfinder',
        'xnat',
        'pillow',
        'matplotlib'
    ],
    python_requires='>=3.5, <4',
    tests_require = ['tests'],
    long_description = long_description,
    # Denotes that our long_description is in Markdown; valid values are
    # text/plain, text/x-rst, and text/markdown
    #
    # Optional if long_description is written in reStructuredText (rst) but
    # required for plain-text or Markdown; if unspecified, "applications should
    # attempt to render [the long_description] as text/x-rst; charset=UTF-8 and
    # fall back to text/plain if it is not valid rst" (see link below)
    #
    # This field corresponds to the "Description-Content-Type" metadata field:
    # https://packaging.python.org/specifications/core-metadata/#description-content-type-optional
    long_description_content_type='text/markdown',
    cmdclass=cmdclass,
    # these are optional and override conf.py settings
    command_options={
        'build_sphinx': {
            'project': ('setup.py', name),
            'version': ('setup.py', version),
            'release': ('setup.py', release),
            'source_dir': ('setup.py', 'docs')}},
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
