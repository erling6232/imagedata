.. _Sorting:

Sorting
=======

When reading data files to construct a Series instance, the images will be sorted into a
multidimensional array.
In particular, DICOM series typically comes as 2D slice images.
These images are sorted into 3D volumes on the basis of their geometrical information
(image position, or slice location).

The following example loads a 3D volume from 'dicom/volume' and sorts the images into
a 3D volume of 40 slices.

.. code-block:: python

    img = Series('dicom/volume')
    print(img.shape)
    >>> (40, 192, 152)

4D data can be read by sorting the volumes on some DICOM attribute. In the following example, a
4D dataset is loaded and volumes are sorted according to Acquisition Time. Notice that there
are 10 time steps and 40 slices in the dataset.

.. code-block:: python

    img = Series('dicom/time', input_order='time')
    print(img.shape)
    >>> (10, 40, 192, 152)

Several sort criteria are predefined:

* none: No sorting (2D/3D datasets only)
* auto: Determine sorting criteria automatically (default) (see below)
* time: Sort on Acquisition Time
* triggertime: Sort on Trigger Time
* b: Sort on MR diffusion b-value
* bvector: Sort on MR diffusion b vector
* fa: Sort on MR Flip Angle
* te: Sort on MR Echo Time

Auto-sorting
------------

`Imagedata` can determine the sorting automatically in some typical cases,
attempting to sort on `time`, `b`-value, echo time or flip angle.

The exact list of sorting criteria can be set using the `auto_sort` list.
The default auto sort list is ['time', 'triggertime', 'b', 'fa', 'te'].

.. code-block:: python

    img = Series('dicomdata', auto_sort=['time', 'b'])

User-defined sorting criteria
-----------------------------

When your data does not match any of these sort criteria, a new sort criteria can be defined.
E.g., sorting MR images on Inversion Time ('ti'), a new sort criteria 'ti' is coupled to the
DICOM attribute Inversion Time:

.. code-block:: python

    img = Series('dicom/time', input_order='ti', ti='InversionTime')
    print(img.shape)
    >>> (10, 40, 192, 152)

More advanced extraction of sorting values can be implemented creating a
user function which returns a value for each Dataset:

.. code-block:: python

    from pydicom.dataset import Dataset
    def get_TI(im: Dataset) -> float:
        return float(im.data_element('InversionTime').value)

    img = Series('dicom/TIdata', input_order='ti', ti=get_TI)

A similar user function to calculate time in seconds from Trigger Time (ms).
Notice how the get_TriggerTime() function overloads the standard `time`
definition. This allow to treat the acquired series as a time series with
the `timeline` property.

NOTICE: Trigger Time is now implemented in the standard library.
This discussion remains here to document the possible uses.

.. code-block:: python

    from pydicom.dataset import Dataset
    def get_TriggerTime(im: Dataset) -> float:
        return float(im.data_element('TriggerTime').value / 1000.)

    img = Series('dicom/triggerTime', input_order='time', time=get_TriggerTime)
    img.timeline

Auto-sorting either on Acquisition Time or Trigger Time can be implemented.
In this case, the resulting series will not have the `timeline` property when
Trigger Time is the sorting criteria:

.. code-block:: python

    img = Series('dicomdata', auto_sort=['time', 'trigger'],
        trigger=get_TriggerTime)
    )

N-dimensional sorting
---------------------

While 4D data can be sorted automatically, higher dimensions must be defined explicitly.
The `input_order` parameter can be a comma-separated list of sorting criteria.

A dynamic dual-echo MR acquisition can be sorted on time and echo time into a 5D Series object, like:

.. code-block:: python

    img = Series('dyn_dual_echo', input_order='time,te')

In particular, MR RSI diffusion data can be sorted on `b` value and `b` vector:

.. code-block:: python

    img = Series('diff_rsi', input_order='b,bvector')
    tags = img.tags[0]
    for idx in np.ndindex(tags.shape):
        try:
            b, bvector = tags[idx]
        except TypeError:
            continue
        rsi = img[idx]
        print(b, bvector, rsi.shape)

