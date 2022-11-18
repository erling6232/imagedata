.. _ReleaseNotes:

Release Notes
=============

1.5.1
-----
Bug fix release:

* MyPolygonSelector.__init__(): define self._polygon_complete.
* Viewer.grid_from_roi(): Set input_order='none' for resulting mask when follow is False.
* Series.tags: Ensure that tag lists are always Numpy arrays.
* Series: Handle properly the situation where a template has wrong number of slices or tags.

1.5.0
-----
* Improved reading of DICOM files. Each file is read once only.

1.4.5.4
-------
* Fixed importlib_metadata handling for Python before 3.8.
* Added example using Series and FSL MCFLIRT.
* Support for Python 3.10.


1.4.5
-----
* Header.__make_DicomHeaderDict_from_template(): Catch IndexError exception when template size does not match data size.
* Documentation updates.


1.4.4
-----
* Do not scan all local subdirectories when looking up a specific file.
* Removed setup.py.
* AbstractPlugin.read(): Use try..finally to ensure files are closed.
* Using relative imports for internal imagedata modules.
* FilesystemArchive: Do not maintain a cache of local files.
* Support for pydicom version 2.3.

1.4.3
-----
* Specify exact python versions supported.
* Verify that template axes are adjusted to present data.
* Refactor calculation of smallest/largest pixel value in image/series.
* image_data.calculator: When no indirs arguments, calculate a new image, optionally with template and geometry given.
* Series: Set default header when input is np.ndarray.
* ITKPlugin.get_image_from_numpy(): return image as np.ndarray, not itk matrix.
* Header.set_default_values(): Set default geometry.

1.4.2
-----
* Set Pixel Representation depending on dtype.
* Handle DICOM signed integer depending on value of Pixel Representation.

1.4.1
-----
* Fixed problem where Window Center and Level with multiple pairs of windows were not read correctly.

1.4.0
-----
* New plugin architecture using python's entry_points.
* imagedata_format_ps is split out in separate package.
* Added image_show console application.

1.3.8
-----
Dicomplugin.sort_images(): Re-raise exception when data cannot be sorted, or when tag is None.

1.3.7
-----
* Format plugin classes: Header.add_template() and Header.add_geometry are now Header member functions.
* Header class: hdr is now a Header instance. Was: dict. Dict is no longer accepted.
* Viewer class now inherits object class.
* Series.setDicomAttribute(): Always make a new attribute to avoid cross-talk after copying Series instances.
* Added Series.deepcopy().

1.3.6
-----

* Moved formats.ps to separate package. Thereby removed dependency on ghostscript package.
* NiftiPlugin: Stable version where SForm matrices are read and written properly for coronal orientation.
* DICOMPlugin: Do not scale pixel data when RescaleSlope == 1 and RescaleIntercept == 0.
* Accept Path object as url.
* Series.get_roi: New parameter `single` when only one slice per tag has a ROI.
* Series.to_rgb(): Use matplotlib colormaps to create color images. Add colormap and lut parameters.
* Default colormap is 'Greys_r'.
* Display colorbar when colormap is not grayscale, and image is not RGB image.
* Set appropriate Window Center and Width for new Series instance.
* Series.to_rgb(): Added option 'norm' to determine normalization method. Methods implemented: 'linear' and 'log'. Modified behavior to honor normalization method.
* Viewer: Renamed option `cmap` to `colormap`.
* Viewer: Let the delta increment of Window Center and Width depend on actual value range. Avoid non-positive Window Width. Let Window Center be a floating point number when less than 2.
