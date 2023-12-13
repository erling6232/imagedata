# Changelog / release notes

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--next-version-placeholder-->

## [v3.3.0-rc2] - 2023-12-13
### Fixed
* Protected Series and Header objects for missing properties.

## [v3.3.0-rc1] - 2023-12-01
### Fixed
* Fixed signed integer overflow in DICOMPlugin.write_slice().

## [v3.3.0-rc0] - 2023-11-30
### Changed
* Major modification to Series class, removing the DicomHeaderDict attribute.

## [v3.2.4-rc1] - 2023-11-17
### Fixed
* Header: When using a template to create DicomHeaderDict, use deepcopy to get a unique
  DicomHeaderDict for the new instance.
### Added
* Viewer: Save window center/width to Series object when returning.
* Series.fuse_mask(): New parameter `blend` which determines whether
  the self image is blended. Default is False, so the self images is
  not blended, only the mask is blended.

## [v3.2.4-rc0] - 2023-11-15
### Fixed
* Series: Handle the special case when a ufunc is called with a where= Series object used
  by NumPy >= 1.25.
* Series.write(): Updated documentation on 'window' option.

## [v3.2.4-dev0] - 2023-11-14
### Added
* Series.write(..., opts={'window': 'original'}): Use original
  window center/level from data object, do not calculate window
  from present data.
* Viewer: Accept MONOCHROME1 photometric interpretation, using the Greys colormap.
### Fixed
* Viewer: Re-enabled linked scrolling of several images series.

## [v3.2.3] - 2023-11-09
### Fixed
* Error in documentation example on drawing a time curve when mask is moved. Fixed.
* Viewer: Updated MyPolygonSelector to match Matplotlib 3.8.
* Require numpy < 1.25 as temporary fix, due to a mismatch where array_likes can override
  ufuncs if used as `where=`.

## [v3.2.3-dev0] - 2023-10-17
### Fixed
* Series: Protect for NaN values.
* Viewer: Protect for NaN values.
* Viewer: Display real part only of complex values.

## [v3.2.2] - 2023-10-11
### Fixed
* Added sphinx_rtd_theme as requirement.

## [v3.2.1] - 2023-10-11
### Fixed
* Added sphinx_rtd_theme to Sphinx conf.py.

## [v3.2.0] - 2023-10-11
### Added
* imagedata.__init__(): Import all plugins to enable plugin-specific initialisation,
   e.g. setting mimetypes.

## [v3.1.0] - 2023-09-05
### Fixed
* Restored logging facility in command-line utilities.
* Better logging when reader plugins fail.

## [v3.1.0-rc1] - 2023-09-01
### Fixed
* Collections error fixed.

## [v3.1.0-rc0] - 2023-09-01
### Added
* Viewer.update() and Viewer.onselect():
Call onselect when setting up display which includes a ROI.
This allows e.g. drawing a time curve on initial display.

### Fixed
* Corrected problem in collections where writing images failed when study datetime is None.
The study instance UID is used instead to construct directory names.

## [v3.0.0] - 2023-08-24
### Added
* Stable release.

### Changed
* Modified NIfTI reading/writing.
DICOM to NIfTI is now compatible with popular `dcm2niix` tool.

### Fixed
* Require itk-io version >= 5.3. Previous versions use the np.bool dtype.

## [v3.0.0-rc7] - 2023-08-23
### Fixed
* numpy.bool is deprecated.

## [v3.0.0-rc6] - 2023-08-17
### Fixed
* Series: Fixed a problem where calculated float window center and level could not be represented in DICOM DS tag.
Window center and level are now calculated using float32.
* Series: Calculate proper window center and level when a new instance is created.
* imagedata.formats.dicomplugin: write_slice will change Series into ndarray before rescaling content.
This avoids crosstalk with original window center and level in Series.

## [v3.0.0-rc5] - 2023-08-11
### Fixed
* imagedata.formats.niftiplugin: Corrected writing 4D images.
When the 4th dimension is not time, the zoom is set to 1.

### Added
* Series.show() and Series.get_roi() take an `ax` argument to show image in existing matplotlib Axes.
* Series.get_roi() takes an `onselect` callback function which is called when the ROI is modified.

## [v3.0.0-rc4] - 2023-08-10
### Changed
* Drop dependency on importlib_metadata.
* Require numpy version 1.19 or later.
* Require nibabel version 5.0.0 or later.
* Drop support for Python 3.7.
* imagedata.formats.niftiplugin: Use sform, not qform, when possible.
* Removed unused testing code.

## [v3.0.0-rc3] - 2023-08-09
### Fixed
* imagedata.formats.niftiplugin corrected for rotated volumes.
### Changed
* Drop support for Python 3.6.
* Require nibabel version 4.0.0 or later.

## [v3.0.0-rc2] - 2023-08-08
### Changed
* Series.align(): Resulting image is rounded to nearest integer when the moving image is integer.

## [v3.0.0-rc1] - 2023-08-07
### Fixed
* imagedata.formats.niftiplugin also calculates imagePositions on read.

## [v3.0.0-rc0] - 2023-08-03
### Changed
* imagedata.formats.niftiplugin improved to handle nifti files
like the popular tool dcm2niix (https://www.nitrc.org/projects/dcm2nii/).
* Notice that nifti files written with imagedata prior to version 3 will be incompatible
with version 3 onwards.

## [v2.1.2] - 2023-05-15
### Changed
* imagedata.apps.diffusion: Improved documentation.

## [v2.1.1] - 2023-05-15
### Changed
* Modified fetching package metadata and entry_points.
 
## [v2.1.0] - 2023-05-12
### Added
* imagedata.apps.diffusion module to extract diffusion MRI parameters.

## [v2.0.0] - 2023-02-13
### Added
* Study class: a collection of Series instances.
  Sort images into separate Series depending on SeriesInstanceUID.
  (https://github.com/erling6232/imagedata/issues/22)
* Patient and Cohort classes: Patient is a collection of Study instances,
  while Cohort is a collection of Patient instances.
* Simpler import statements in user code: `from imagedata import Series, Study`
* Add support Python 3.11 (https://github.com/erling6232/imagedata/issues/21)
* Series.to_rgb(): Added clip parameter whether clipping to DICOM window or
  to histogram probabilities.
* Series.fuse_mask(): Color fusion of mask.
* Added Series.align() method.
### Changed
* `input_order='auto'`: Auto-detect the sorting of Series,
  depending on which DICOM attribute varies.
  The input_orders time/b/fa/te are attempted in order.
* `auto` is now the default input_order.
* DWI images will typically have varying time. Let `b` values override the time stamps
  during auto-detect sorting of Series.
* The dicom read has been modified to keep the UIDs of the input files.
  This way SOPInstanceUIDs are correct when you later want to look up SOPInstanceUID from a DICOM PR.
* image_data.statistics: print patient, study and series properly.
* image_data.conversion: improved conversion of date/time for directory names.
* The series write() method now has an option to keep the UID when writing.
  The UIDs used to be modified at output to indicate that the data might have been modified.
  Do something like:

  a.write(destination, opts={'keep_uid': True})

### Fixed
* DICOMPlugin: Catch errors when converting DICOM attributes to numbers.
* Header.add_geometry() takes one template only.
* Axis: Enhanced class with __getitem__ and __next__ to enable iteration over axis values.

## [v2.0.0-rc2] - 2023-02-13
### Changed
* cmdline: Changed default input order to `auto`.

## [v2.0.0-rc1] - 2023-02-10
### Added
* Added Series.align() method.
### Fixed
* Header.add_geometry() takes one template only.
* Axis: Enhanced class with __getitem__ and __next__ to enable iteration over axis values.

## [v2.0.0-rc0] - 2023-01-31
### Added
* Series.to_rgb(): Added clip parameter whether clipping to DICOM window or
to histogram probabilities.
* Series.fuse_mask(): Color fusion of mask.

## [v2.0.0-dev6] - 2023-01-30
### Changed
* Collections.Study: studyTime attribute is datetime.time object.
* image_data.statistics: print patient, study and series properly.
* image_data.conversion: improved conversion of date/time for directory names.
### Fixed
* Collections.Study: Handle missing study date and/or time.

## [v2.0.0-dev5] - 2023-01-30
### Fixed
* DICOMPlugin: Catch errors when converting DICOM attributes to numbers.

## [v2.0.0-dev4] - 2023-01-16
### Fixed
* Use pydicom.valuerep.format_number_for_ds when writing float numbers with VR=DS.
### Changed
* Set writing_validation_mode to RAISE to notify when DICOM elements are illegal.

## [v2.0.0-dev3] - 2023-01-09
### Added
* Documentation on Collections classes.
### Fixed
* Re-examined the case of changing UIDs during dicom write. Corrected.

## [v2.0.0-dev2] - 2023-01-06
### Changed
* Study class: studyDate/studyTime are now datetime.datetime instances.
* Patient class: patientSize/patientWeight are now float numbers.
### Fixed
* Corrected UID handling from v2.0.0-dev1.

## [v2.0.0-dev1] - 2023-01-05
### Added
* Patient and Cohort classes: Patient is a collection of Study instances, while
Cohort is a collection of Patient instances.
* The dicom read has been modified to keep the UIDs of the input files.
This way SOPInstanceUIDs are correct when you later want to look up SOPInstanceUID from a DICOM PR.
* The series write() method now has an option to keep the UID when writing. The UIDs used to be modified at output to indicate that the data might have been modified. Do something like:

	a.write(destination, opts={'keep_uid': True})


## [v2.0.0-dev0] - 2022-12-19
### Added
* Study class: a collection of Series instances.
Sort images into separate Series depending on SeriesInstanceUID.
(https://github.com/erling6232/imagedata/issues/22)
* Simpler import statements in user code: from imagedata import Series, Study
* Add support Python 3.11 (https://github.com/erling6232/imagedata/issues/21)
### Changed
* `input_order='auto'`: Auto-detect the sorting of Series,
depending on which DICOM attribute varies.
The input_orders time/b/fa/te are attempted in order.
* `auto` is now the default input_order.
* DWI images will typically have varying time. Let `b` values override the time stamps
during auto-detect sorting of Series.

## [1.6.1]
### Fixed
* Viewer: self.callback_quit was uninitialized when show() was used.
* Series.__array_finalize__(): Copy Header object from any array object which has a Header object.
* Series.__getitem__(): Finalize ret object for all Series instances, independent on slicing status.
* Header.__init__(): No need to set self.axes explicitly.
* Series.axes(): Define RGB axis when last dimension is size 3 or 4 and uint8.
* Series.color: Color property is determined by the presence of an RGB axis. Do not set color property.
* Series.photometricInterpretation: Do not set color property.
* Remove all references to header.color. Color status is determined by the presence of an RGB axis.

## [1.6.0]
### Added
* show() and get_roi() works in Jupyter notebook, too.
* In notebook: Series.get_roi_mask() must be called after get_roi() to get the actual mask.

## [1.5.1]
### Fixed
* MyPolygonSelector.__init__(): define self._polygon_complete.
* Viewer.grid_from_roi(): Set input_order='none' for resulting mask when follow is False.
* Series.tags: Ensure that tag lists are always Numpy arrays.
* Series: Handle properly the situation where a template has wrong number of slices or tags.

## [1.5.0]
### Changed
* Improved reading of DICOM files. Each file is read once only.

## [1.4.5.4]
### Added
* Support for Python 3.10.
* Added documentation example using Series and FSL MCFLIRT.
### Fixed
* Fixed importlib_metadata handling for Python before 3.8.

## [1.4.5]
### Fixed
* Header.__make_DicomHeaderDict_from_template(): Catch IndexError exception when template size does not match data size.
* Documentation updates.

## [1.4.4]
### Added
* Support for pydicom version 2.3.
### Changed
* Do not scan all local subdirectories when looking up a specific file.
* Removed setup.py.
* AbstractPlugin.read(): Use try..finally to ensure files are closed.
* FilesystemArchive: Do not maintain a cache of local files.
### Fixed
* Using relative imports for internal imagedata modules.

## [1.4.3]
### Fixed
* Specify exact python versions supported.
* Verify that template axes are adjusted to present data.
* Refactor calculation of smallest/largest pixel value in image/series.
* image_data.calculator: When no indirs arguments, calculate a new image, optionally with template and geometry given.
* Series: Set default header when input is np.ndarray.
* ITKPlugin.get_image_from_numpy(): return image as np.ndarray, not itk matrix.
* Header.set_default_values(): Set default geometry.

## [1.4.2]
### Changed
* Set Pixel Representation depending on dtype.
* Handle DICOM signed integer depending on value of Pixel Representation.

## [1.4.1]
### Fixed
* Fixed problem where Window Center and Level with multiple pairs of windows were not read correctly.

## [1.4.0]
### Added
* New plugin architecture using python's entry_points.
* Added image_show console application.
### Changed
* imagedata_format_ps is split out in separate package.

## [1.3.8]
### Fixed
Dicomplugin.sort_images(): Re-raise exception when data cannot be sorted, or when tag is None.

## [1.3.7]
### Added
* Added Series.deepcopy().
### Changed
* Header class: hdr is now a Header instance. Was: dict. Dict is no longer accepted.
* Viewer class now inherits object class.
* Series.setDicomAttribute(): Always make a new attribute to avoid cross-talk after copying Series instances.
### Fixed
* Format plugin classes: Header.add_template() and Header.add_geometry are now Header member functions.

## [1.3.6]
### Added
* Accept Path object as url.
* Series.get_roi: New parameter `single` when only one slice per tag has a ROI.
* Series.to_rgb(): Use matplotlib colormaps to create color images. Add colormap and lut parameters.
* Display colorbar when colormap is not grayscale, and image is not RGB image.
* Series.to_rgb(): Added option 'norm' to determine normalization method. Methods implemented: 'linear' and 'log'. Modified behavior to honor normalization method.
### Changed
* Moved formats.ps to separate package. Thereby removed dependency on ghostscript package.
* Default colormap is 'Greys_r'.
* Viewer: Renamed option `cmap` to `colormap`.
* Set appropriate Window Center and Width for new Series instance.
* Viewer: Let the delta increment of Window Center and Width depend on actual value range. Avoid non-positive Window Width. Let Window Center be a floating point number when less than 2.
### Fixed
* NiftiPlugin: Stable version where SForm matrices are read and written properly for coronal orientation.
* DICOMPlugin: Do not scale pixel data when RescaleSlope == 1 and RescaleIntercept == 0.
