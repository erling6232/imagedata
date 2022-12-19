# Changelog / release notes

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!--next-version-placeholder-->

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