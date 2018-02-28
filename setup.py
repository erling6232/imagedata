#!/usr/bin/env python

from distutils.core import setup

setup(name = 'imagedata',
    packages = ['imagedata'],
    version = '1.0.0a1',
    description = 'Read/write image data to file(s). Handles DICOM, Nifti, VTI and mhd.',
    author = 'Erling Andersen',
    author_email = 'Erling.Andersen@Helse-Bergen.NO',
	url = 'http://www.helse-bergen.no', # Change this later
	classifiers = [
		"Programming Language :: Python :: 2.7",
		"Programming Language :: Python :: 3",
		"Development Status :: 3 - Alpha",
		"Environment :: Other Environment",
        "Intended Audience :: Developers",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
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
	long_description = """\
	imagedata
	==========

    Read/write image data to file(s). Handles DICOM, Nifti, VTI and mhd.
	"""
)
