.. _Sorting:

Sorting
=======

When reading data files to construct a Series instance, the images will be sorted into a
multidimensional array.
In particular, DICOM series typically comes as 2D slice images.
These images are sorted into 3D volumes on the basis of their geometrical information
(Image position, or slice location).

The following example loads a 3D volume from 'disk/volume' and sorts the images into
a 3D volume of 40 slices.

.. code-block:: python

    img = Series('dicom/volume')
    print(img.shape)
    >>> (40, 192, 152)

4D data can be read by sorting the volumes on some DICOM attribute. In the following example, a
4D dataset is loaded and volumes are sorted according to Acquisition Time. Notice that there
are 10 time steps in the dataset.

.. code-block:: python

    img = Series('dicom/time', input_order='time')
    print(img.shape)
    >>> (10, 40, 192, 152)

Several sort criteria are predefined:

* none: No sorting (2D/3D datasets only) (default)
* time: Sort on Acquisition Time
* b: Sort on MR diffusion b-value
* fa: Sort on MR Flip Angle
* te: Sort on MR Echo Time

When your data does not match any of these sort criteria, a new sort criteria can be defined.
E.g., sorting MR images on Inversion Time ('ti'), a new sort criteria 'ti' is coupled to the
DICOM attribute Inversion Time:

.. code-block:: python

    img = Series('dicom/time', input_order='ti', opts={'ti': 'InversionTime'})
    print(img.shape)
    >>> (10, 40, 192, 152)
