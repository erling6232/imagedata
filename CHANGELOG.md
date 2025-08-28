# Changelog / release notes

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--next-version-placeholder-->

## [v3.8.7] - 2025-08-28
### Fixed
* Series.__getitem__(): Corrected labeling tags when slicing in the slice direction.
  Added test to verify correct labeling.
* Series.__getitem__(): Properly label axes when using numpy ufuncs.
* Series.__getitem__(): Handle indexing when additional dimension(s) are added with np.newaxis.

### Changed
* Require setuptools >= 78.1.1
* Require xnat < 0.7

## [v3.8.6] - 2025-08-20
### Added
* dicom_sort commandline program.

### Changed
* DICOMPlugin: Print series description and number in warning and error messages. 

### Fixed
* DICOMPlugin: Improved error handling when a series cannot be sorted.
* DICOMPlugin._calculate_distances: Protect for missing slice location.
* DICOMPlugin._verify_spacing: Print warning message only once.

## [v3.8.5] - 2025-08-07
### Changed
* DICOMPlugin: Reworked sorting slices into stacks based on slice location.
  Slice location is now determined by the distance along the normal vector to the plane.
* DICOMPlugin: Legacy slice sorting can be enabled by setting `sort_on_slice_location` to True.
* NiftiPlugin: Improved geometry handling following the dcm2niix implementation.

## [v3.8.5-rc1] - 2025-08-06
### Changed
* NiftiPlugin: Improved geometry handling.

## [v3.8.5-rc0] - 2025-07-10
### Changed
* DICOMPlugin: Corrected calculation of tranformationMatrix and imagePositions.
* DICOMPlugin: Added option `sort_on_slice_location` to sort stack on slice location,
  not on distance along normal vector. Default: False.
### Fixed
* UniformLenghtAxis: Use absolute value of `n` as axis length.

## [v3.8.4] - 2025-05-14
### Added
* DICOMPlugin: Calculate slice spacing from actual slice locations. Not from SliceThickness
  nor from SpacingBetweenSlices.
* Series.align: Added optional `fill_value` for voxels outside field-of-view.

## [v3.8.4-rc0] - 2025-05-14
* Release candidate 3.8.4-rc0

## [v3.8.4-dev3] - 2025-05-14
### Added
* Series.align: Added optional `fill_value` for voxels outside field-of-view.
* DICOMPlugin: Added options `slice_tolerance`and `dir_cosine_tolerance`.
* DICOMPlugin.__get_voxel_spacing(): New implementation to calculate slice position from `Image Position (Patient)` attribute.
* DICOMPlugin.getDicomAttributeValues(): New function to return a list of values for all slices.

## [v3.8.4-dev0] - 2025-05-06
### Fixed
* Series.__getitem__(): Limit number of axes to ndim.
* DICOMPlugin: Limit number of axes to ndim.
* Updated tests to accept correct number of axes.
### Changed
* Require setuptools >= 78.1.1

## [v3.8.3] - 2025-05-06
### Added
* Series honors the `dtype` parameter. Added tests to verify usage.
* Series.write(): Accept kwargs options.
### Fixed
* Removed outdated documentation on `window` option.
### Changed
* Series.seriesInstanceUID: Modified behaviour depending on `keep_uid` option. Default: False.
* Improved Series.timeline.

## [v3.8.3-dev2] - 2025-05-06
* Series honors the dtype parameter. Added tests to verify usage.
* Series.seriesInstanceUID: Modified behaviour depending on keep_uid option. Default: False.
* Series.write(): Accept kwargs options.
* Removed outdated documentation on 'window' option.

## [v3.8.3-dev1] - 2025-05-05
* Improved Series.timeline

## [v3.8.2] - 2025-04-30
### Fixed
* Viewer.build_info(): Fixed error prohibiting display of 3D Series.

## [v3.8.1] - 2025-04-29
### Fixed
* Series.__getitem__(): Do not index on ndarray and Series indexes.
* Series: Properly slice arrays on np.int64 in addition to int.

## [v3.8.0] - 2025-04-28
### Added
* Novel sorting routine for n-dimensional DICOM datasets.
* Allow user-defined sorting criteria, and overriding default sorting.
* Sort on Trigger Time also.
* Added code to sort series on diffusion gradient direction (b-vector),
  and specifically diffusion RSI data.

### Changed
* Image tags property (tags) is now a multi-dimensional array when the image dimension > 4.
  In this case the image tags will be tuples.
* diffusion.py: Added code to get diffusion b value from Siemens E11 format.
* Patient class: Limit the strict check of patient attributes to
  patientName, patientID, patientBirthDate
  and patientSex.
* Series.__getitem__(): Accept tuple slicing specification, like ((2,),(2,),(2,)).
* Improved some test suites to run better on Windows.

### Fixed
* Corrected parsing windows UNC path names with leading double slash.
* ZipFileArchive: Windows: Do not complain when temporary local file cannot be removed.
* XnatTransport.open(): Corrected scan search to search for series description,
  not series number.
* Viewer.viewport_set(): Fixed error where the viewport did not include the last image.
* evidence2roi: Do not raise exception for unknown roi type. Log a warning instead.

## [v3.8.0-rc3] - 2025-04-28
* Added sort on TriggerTime

## [v3.8.0-rc2] - 2025-04-23
* Allow user-defined sorting criteria, and overriding default sorting.

## [v3.8.0-rc1] - 2025-04-11
* Novel sorting routine for n-dimensional DICOM datasets.

## [v3.7.3-dev8] - 2025-03-07
* Corrected parsing windows UNC path names with leading double slash.

## [v3.7.3-dev7] - 2025-03-07
* ZipFileArchive: Windows: Do not complain when temporary local file cannot be removed.
* Improved some test suites to run better on Windows.

## [v3.7.3-rc0] - 2025-01-06
* Release candidate 3.7.3-rc0

## [v3.7.3-dev6] - 2024-12-13
* Series.__getitem__(): Accept tuple slicing specification, like ((2,),(2,),(2,)).

## [v3.7.3-dev5] - 2024-12-12
* Patient class: Limit the strict check of patient attributes to
  patientName, patientID, patientBirthDate
  and patientSex.

## [v3.7.3-dev4] - 2024-12-04
* evidence2roi: Do not raise exception for unknown roi type. Log a warning instead.

## [v3.7.3-dev3] - 2024-11-26
* Working on n-dimensional sorting.
* Added code to sort series on diffusion gradient direction (b-vector),
  and specifically diffusion RSI data.

## [v3.7.3-dev2] - 2024-11-25
* diffusion.py: Added code to get diffusion b value from Siemens E11 format.

## [v3.7.3-dev1] - 2024-11-06
* Viewer.viewport_set(): Fixed error where the viewport did not include the last image.

## [v3.7.3-dev0] - 2024-11-04
* XnatTransport.open(): Corrected scan search to search for series description,
  not series number.

## [v3.7.2] - 2024-10-28
* Fixed viewing in jupyter notebook using backend `widget`.
* Renamed collections.py to collection.py.
* 
## [v3.7.2-dev2] - 2024-10-25
* Fixed viewing in jupyter notebook using backend `widget`.

## [v3.7.2-dev0] - 2024-10-10
* Renamed collections.py to collection.py.

## [v3.7.1] - 2024-10-10
* Support python 3.13.

## [v3.7.0] - 2024-10-09
* Series: Axes property implementation is changed from a list to a namedtuple.

## [v3.7.0-rc4] - 2024-10-02
* Series.__get_tags(): Corrected looping over slice objects.

## [v3.7.0-rc3] - 2024-10-02
* Series: Let __array_function__ call numpy for functions we do not implement.
  This will possibly return an ndarray instance, not a Series instance.
* Axis: Added property `values` which will give all axis values.
* DICOMPlugin.write_slice(): Added parameter tag_value which gives the tag value
  instead of relying on tags property.
* Series.concatenate(): Corrected behaviour.

## [v3.7.0-rc2] - 2024-09-30
* Series: timeline from time axis, not from tags.
* Series: implement __array_function__ and a number of NumPy functions
  on Series instances.
* Series.concatenate() to concatenate a number of images along specified axis.

## [v3.7.0-dev8] - 2024-09-27
* readdata._get_location_part(): Modified behaviour to detect local file url. 
  urllib.parse.urlunparse was changed in 3.12.6.

## [v3.7.0-dev7] - 2024-09-23
* DICOMPlugin: Changed type definitions into classes.

## [v3.7.0-rc0] - 2024-09-09
### Added
* DICOMPlugin: Read non-image datasets (e.g. structured reports).

### Changed
* Support pydicom 3.0.0.
* Support numpy >= 2.0.0.
* Support upcoming python 3.13.

### Fixed
* DICOMPlugin.__get_transformation_matrix(): Fixed problem when there is one slice only.
* DICOMPlugin._extract_dicom_attributes(): Protected for missing imagePositions.

## [v3.6.6] - 2024-08-06
### Fixed
* Series.__getitem__(): Set correct input_order when slicing 4D Series.
* Series.__getitem__(): Accept discrete slice selection specified by list or tuple.
* Series.__get_imagePositions(): Accept list and tuple indexes.
* Series.__get_tags(): Accept list and tuple indexes.
* Series.sliceLocations setter: Do not set slice axis.

## [v3.6.5] - 2024-07-12
### Fixed
* Improved UID generation to guarantee unique DICOM SeriesInstanceUID,
  while keeping the SeriesInstanceUID when slicing a Series.

## [v3.6.4] - 2024-07-11
### Fixed
* DICOMPlugin._extract_all_tags(): Fixed error where float tag was printed with :0x formatting.

## [v3.6.3] - 2024-07-11
### Fixed
* XNATTransport.walk(): Only return file names from the `top` down.

## [v3.6.2] - 2024-07-09
### Added
* DICOMPlugin: Added `use_cross_product` option to demand that the z column of the transformation
  matrix is calculated using vector cross product.
* XNATTransport.open(): Can now download data also at patient and experiment level.
### Changed
* evidence2mask: Removed functions make_mask_in_slice() and
  transform_data_points_to_voxels().`

## [v3.6.1] - 2024-06-28
* image_data._reduce(): Corrected indexing collections classes. 
* cmdline.DictAction: Use ast.literal_eval() to evaluate input options.
* Collections and Series: Access input_options in opts similar to kwargs.
* Remove exception UnevenSlicesError. Use CannotSort instead.

## [v3.6.0] - 2024-06-27
* Final release 3.6.0.

## [v3.6.0-rc4] - 2024-06-25
### Changed
* Depend on pylibjpeg.
* Standardized logging to log proper module and function names.
### Fixed
* Honor the `skip_broken_series` option.

## [v3.6.0-rc3] - 2024-06-21
* Release candidate 3.

## [v3.6.0-rc2] - 2024-06-20
* Release candidate 2.

## [v3.6.0-rc0] - 2024-06-17
* Release candidate 0.

## [v3.6.0-dev2] - 2024-06-10
## Added
* DICOMPlugin: Print tag values in hex and keyword.

## Changed
* Series.seriesDescription: Return empty string when not defined.
* Collections: Added possibility to index Study, Patient and Cohort by integer keys
  in addition to uid.
* Depend on itk-io 5.4.0 on all python versions.
* Depend on pydicom 2.4.0 and matplotlib 3.8.0.

## [v3.6.0-dev0] - 2024-06-07
### Changed
* Refactored DICOMPlugin to better split series based AcquisitionNumber and/or EchoNumber.
* Format plugin read() now return hdr and si as dicts of series.
* Drop support for python 3.8.

## [v3.5.6-dev0] - 2024-05-30
### Changed
* Modify DICOMPlugin to improve sorting based on several criteria.

## [v3.5.5] - 2024-05-28
### Fixed
* Added `strict_values` to Options documentation for Study/Patient/Cohort classes.

## [v3.5.5-rc4] - 2024-05-22
### Added
* DICOMPlugin: Option `skip_broken_series` to bypass broken series in a study.
  Otherwise, an exception is raised.

## [v3.5.5-rc3] - 2024-05-15
### Fixed
* DICOMPlugin: Raise exception CannotSort when CSA header cannot be read.

## [v3.5.5-rc2] - 2024-05-14
### Added
* DICOMPlugin: multiple acquisition numbers and slice thicknesses are resolved by
  keeping the thin slices only. Option `select_thickness` is used to select 'thin'
  slices (default), or 'thick' slices.
* Cohort, Patient, Study and Series: accept options as kwargs.
* Add EchoNumbers and AcquisitionNumber to Header.

## [v3.5.5-rc1] - 2024-04-29
### Added
* Series: Added parameter `input_format` to specify a particular input format.
* test_formats_dicom: Added unittest for input_format.

### Fixed
* DICOMPlugin.read: Raise CannotSort exception when UnevenSlicesError is raised internally.

## [v3.5.5-rc0] - 2024-04-26
### Fixed
* DICOMPlugin.sort_images(): Raise exception UnevenSlicesError when number of slices differ
  across a volume.
* DICOMPlugin.get_dicom_files(): Raise exception CannotSort when sort_images()
  raises UnevenSlicesError.
* imagedata._reduce(): Protect for empty lists.

## [v3.5.4] - 2024-04-25
### Added
* Added option `input_acquisition` to Series to select a particular Acquisition Number in a CT series.
* Added command line option --input_acquisition to select a particular Acquisition Number.

## [v3.5.3] - 2024-04-10
### Fixed
* NiftiPlugin and ITKPlugin: create output directory before writing local files.
* Header: set default sort_on=SORT_ON_SLICE.

### Added
* conversion: write cohort data at the lowest level possible.

## [v3.5.2] - 2024-04-09
### Fixed
* Corrected file name generation when writing 4D DICOM using SORT_ON_TAG.

## [v3.5.1] - 2024-04-09
### Added
* AbstractPlugin: Honor the input_sort option.
### Fixed
* Corrected behaviour when writing 4D DICOM with output_dir == 'multi'.

## [v3.5.1-rc3] - 2024-04-09
### Added
* AbstractPlugin: Honor the input_sort option.

## [v3.5.1-rc2] - 2024-04-08
### Fixed
* Corrected behaviour when writing 4D DICOM with output_dir == 'multi'.

## [v3.5.0] - 2024-02-29
* Modified file name generation to allow user to specify single file output when the format plugin support this.

## [v3.5.0-rc1] - 2024-02-29
* Release candidate 1

## [v3.5.0-rc0] - 2024-02-22
* Release candidate 0

## [v3.4.4-dev3] - 2024-02-02
* Format plugins: Improved file name generation.

## [v3.4.4-rc2] - 2024-01-31
* FilesystemArchive.open(): Corrected behavior, adding root path.

## [v3.4.4-rc0] - 2024-01-31
* ITKPlugin, MatPlugin and NiftiPlugin: when writing to a local file with proper
  file extension (.mat, .nii.gz or .nii), bypass transport plugin to
  create file in given directory.

## [v3.4.3] - 2024-01-22
* Added get_transporter_list() to imagedata.transports.
* Added get_archiver_list() to imagedata.archives.

## [v3.4.2] - 2024-01-18
* Release 3.4.2

## [v3.4.2-rc0] - 2024-01-18
### Added
* Added documentation on viewing.

## [v3.4.2-dev0] - 2024-01-18
### Changed
* Series.get_roi_mask(): Return 2D mask when original images is 2D.
* Viewer: Display image text even when some attributes are missing.
* Viewer: Better presentation of window center/width when they are floats.
* Viewer: Reimplemented PageDown/PageUp to scroll one page at a time.
* Viewer: Adjusting window center/width is modified to use
  series min/max, not existing window.
  Will improve user feedback when window is unreasonable.

### Added
* Viewer: Ctrl+Home/End will scroll to first/last series.
* Viewer: Ctrl+Array Left/Right will scroll one series.
* Viewer: Ctrl+Array Up/Down will scroll one row of series.
* Viewer: Toggle the hide text will affect all series.

## [v3.4.1] - 2024-01-15
### Changed
* Series.fuse_mask(): Corrected scaling of mask range to colors.

### Added
* Documentation on using color Series objects.

## [v3.4.0] - 2024-01-12
### Changed
* RGB-image representation.
  Use NumPy structured dtype to represent color data.
  Simplify index calculations when color is not a dimension.
* fuse_mask(): Accept variable mask, rendered in a specified colormap.
* fuse_mask(): Save color map for later viewing.
* Viewer: Display color map when this exists.

## [v3.4.0-rc2] - 2024-01-12
* fuse_mask(): Accept variable mask, rendered in a specified colormap.
* fuse_mask(): Save color map for later viewing.
* Viewer: Display color map when this exists.

## [v3.4.0-rc0] - 2024-01-09
* Release candidate rc0

## [v3.4.0-dev0] - 2024-01-09
### Changed
* RGB-image representation.
  Use NumPy structured dtype to represent color data.
  Simplify index calculations when color is not a dimension.

## [v3.3.3] - 2023-12-21
### Added
* Series.get_roi() will now accept an existing Series grid as roi, in addition to a `vertices` dict.
* Added Series.vertices_from_grid()
* Moved get_slice_axis() and get_tag_axis from viewer.py to series.py.

## [v3.3.1] - 2023-12-20
* Corrected release scripts on GitHub.

## [v3.3.0] - 2023-12-19
* Ready for public distribution.

## [v3.3.0-rc4] - 2023-12-19
* Viewer: Key press 'w' will normalize window level/width to displayed slice
  using a histogram normalization.
* Viewer: Key press 'h' will toggle display of text on display.
* Viewer: Display demographic and acquisition info in upper right and level corners.
### Changed
* Support Python 3.12

## [v3.3.0-rc3] - 2023-12-18
### Added
* Viewer: Display seriesDescription upper right.

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
  the self image is blended. Default is False, such that the self image is
  not blended, only the mask is blended.

## [v3.2.4-rc0] - 2023-11-15
### Fixed
* Series: Handle the special case when an ufunc is called with a where= Series object used
  by NumPy >= 1.25.
* Series.write(): Updated documentation on 'window' option.

## [v3.2.4-dev0] - 2023-11-14
### Added
* Series.write(..., opts={'window': 'original'}): Use original
  window center/level from data object, do not calculate window
  from present data.
* Viewer: Accept MONOCHROME1 photometric interpretation, using the Greys colormap.
### Fixed
* Viewer: Re-enabled linked scrolling of several image series.

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
