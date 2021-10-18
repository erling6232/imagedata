.. _ReleaseNotes:

ReleaseNotes
============

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
