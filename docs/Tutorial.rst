.. _Tutorial:

Tutorial
===============

Reading data
-------------

Let's start by reading some data:

.. code-block:: python

    from imagedata import Series
    a = Series('in_dir')

The basic assumption when reading data from files is that a folder (directory)
contains one image series. In the example above, the 'in_dir' folder will be
searched for image files which will constitute the 'a' variable. Imagedata will
not sort multiple series for you.

When reading a Series, the library will automatically detect the image format,
whether it is DICOM, Nifti, or one of the other formats supported.

All files found will be sorted according to your 'input_order' criteria.
The default input_order is 'none', which will assume that the data
can be sorted into a 3D volume. When there is one slice only, the data will
be stored as a 2D slice. To see how the data has been sorted, look at the shape
and axes properties of 'a':

.. code-block:: python

    a.shape
    >>> (40, 192, 152)
    a.axes
    >>> [UniformLengthAxis(slice,209.07554320292,40,3.0),
    >>>  UniformLengthAxis(row,-29.985630130569,192,2.0833332538605),
    >>>  UniformLengthAxis(column,-199.72427744686,152,2.0833332538605)]


When reading a 4D dataset, you need to specify the input_order criteria.
At present, the following input_orders can be specified:

* 'none': 2D/3D dataset, sorted on slice position
* 'time': 4D time-dependent dataset, sorted on time
* 'b': 4D MRI diffusion-weighted dataset, sorted on diffusion b value
* 'fa': 4D MRI flip-angle dataset, sorted on flip angle
* 'te': 4D MRI echo time dataset, sorted on echo time (TE)

An example reading a time-dependent dataset:

.. code-block:: python

    dynamic = Series('dynamic_dir', 'time')
    dynamic.shape
    >>> (10, 40, 192, 152)
    dynamic.axes
    >>> [VariableAxis(time,array([52531.6625, 52534.65  , 52537.6375, 52540.625 , 52543.6125,
    >>>     52546.6025, 52549.59  , 52552.5775, 52555.565 , 52558.5525])),
    >>>  UniformLengthAxis(slice,209.07554320292,40,3.0),
    >>>  UniformLengthAxis(row,-29.985630130569,192,2.0833332538605),
    >>>  UniformLengthAxis(column,-199.72427744686,152,2.0833332538605)]
    dynamic.timeline
    >>> array([ 0.    ,  2.9875,  5.975 ,  8.9625, 11.95  , 14.94  , 17.9275,
    >>>        20.915 , 23.9025, 26.89  ])

Notice how the time axis is given in seconds from midnight (DICOM style),
while the timeline property give the time as seconds from the first time step.

Displaying data
---------------

.. code-block:: python

    a.show()

The show() method will use matplotlib to display your dataset.
The following controls can be used to manipulate the viewer:

* Mouse scroll wheel: scroll through the slices of a volume.
* Array up/down: scroll through the slices of a volume.
* Array left/right: step through the 4th dimension of a 4D dataset.
* PgUp/PgDown: Page through datasets when there are more datasets than views.
* Ctrl+Home/Ctrl+End: Advance view to first/last dataset.
* Ctrl+Array left/right: Advance view one step back/forward.
* Ctrl+Array up/down: Advance view one row back/forward.
* Left mouse key pressed: adjust window/level of display.
* Mouse over: will display 2D coordinate and signal intensity at mouse position.
* 'h': Toggle hiding text on display.
* 'w': Normalize window center/width based on image histogram.
* 'q': Quit. Will end the show() method.

.. code-block:: python

    dynamic.show(a)

The show() method can display multiple series. The example above will setup
a viewport of two series, where each series can be manipulated independently.
When you want to display additional datasets, specify them in a list:

.. code-block:: python

    dynamic.show([a, b, c])

Additionally, you can draw a region of interest (ROI):

.. code-block:: python

    roi = a.get_roi()

The returned 'roi' variable will be a new 3D Series instance, where
voxels are one inside the ROI, and zero elsewhere.

For dynamic data, it is possible to draw ROI for each time step:

.. code-block:: python

    roi = dynamic.get_roi(follow=True)

Draw a ROI for the first time step, then move to next time step using right array key.
For each time step, the ROI polygon can be adjusted using the mouse:

* Move a polygon vertex using left mouse key
* Move the polygon outline using shift key and left mouse key

The returned ROI will be a 4D ROI Series.

Saving data
-----------
.. code-block:: python

    a.write('my_dir')

The write() method will save the given series in a new file or folder.
With no additional information, the series will be saved in the
same format (DICOM, Nifti, ...) as the input data.
You can specify a different image format, e.g.:

.. code-block:: python

    a.write('my_itk_dir', formats=['itk'])

or even multiple formats, where '%p' is replaced be the format name:

.. code-block:: python

    a.write('my_dirs/%p', formats=['nifti', 'mat'])

This will save the data in Nifti format in 'my_dirs/nifti', and
in Matlab format in 'my_dirs/mat'.

Add DICOM template to other image formats
-------------------------------------------

.. code-block:: python

    b = Series('my_dirs/mat', template=a, geometry=a)

The above example will read a series from a Matlab formatted file, then
add DICOM headers and geometry from existing Series instance 'a'.

Alternatively, the template can be given as a URL:

.. code-block:: python

    b = Series('my_dirs/mat', template='in_dir', geometry='in_dir')

Add DICOM template to numpy array
---------------------------------

When processing image data using e.g. NumPy or SciPy, you may end up
with numpy arrays with no imagedata header. The DICOM header from an
existing dataset can be added to the numpy array:

.. code-block:: python

    # eye is numpy array
    eye = np.eye(128)
    c = Series(eye, template=a, geometry=a)
    c.seriesNumber = 50
    c.seriesDescription = 'eye
    c.axes
    >>> [UniformLengthAxis(row,-29.985630130569,192,2.0833332538605),
    >>>  UniformLengthAxis(column,-199.72427744686,152,2.0833332538605)]
    print(c)
    >>> Patient: 19.02.07-14:04:17-STD-1.3.12.2.1107.5.2.43.66035 PHANTOM^T1
    >>> Study  Time: 20190207 140516.555000
    >>> Series Time: 20190207 143634.267000
    >>> Series #50: eye
    >>> Shape: 128x128, dtype: float64, input order: none

Sorting DICOM files into multiple Series
----------------------------------------

The Study class can be used to sort DICOM file according to SeriesInstanceUID.
The input order of each Series is auto-detected.

.. code-block:: python

    from imagedata import Study

    vibe, dce = None
    study = Study('data/dicom')
    for uid in study:
        series = study[uid]
        if series.seriesDescription == 'vibe':
            vibe = series
        ...
    If not (vibe and dce):
        raise ValueError('Some series not found in study.')
