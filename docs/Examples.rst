.. _Examples:

Examples
========

Split MRI diffusion series into separate series for each b value
----------------------------------------------------------------

Using python:
    Write each b value in turn by looping over first index of 4D DWI dataset.
    Each new series will have a unique Series Instance UID,
    such that a PACS will handle each b value as separate series.

.. code-block:: python

    from imagedata.series import Series

    dwi = Series('10_ep2d_diff_b50_400_800/', 'b')

    # dwi.tags[0] has the b values, e.g. [50, 400, 800]

    for i, b in enumerate(dwi.tags[0]):
        # Output to folders b50/, b400/ and b800/
        dwi[i].write('b{}'.format(b))

More elaborate example in python:
    To set a separate series number and description for each series,
    each volume of b values must be separate objects:

.. code-block:: python

    from imagedata.series import Series

    dwi = Series('10_ep2d_diff_b50_400_800/', 'b')

    for i, b in enumerate(dwi.tags[0]):
        # Output to folders b50/, b400/ and b800/
        dwi[i].write('b{}'.format(b))
        for i, b in enumerate(dwi.tags[0]):
            s = dwi[i]
            s.seriesNumber = 100 + i
            s.seriesDescription = 'b {}'.format(b)
            s.write('b{}'.format(b))

Using command line:
    Split b values using `--odir multi' parameter. Each b value
    will be written to folder tmp/b0, tmp/b1, etc.
    However, all folders will share the Series Instance UID.

    To make unique Series Instance UIDs, run image_data on each
    created folder.

    In the following example, the b values are first split to folders
    tmp/b0, tmp/b1, etc.
    Next, each tmp/b* series is copied again, producing separate
    Series Instance UIDs.
    Notice how each series is given a separate series number and
    series description.

.. code-block:: sh

   image_data --order b --odir multi tmp 10_ep2d_diff_b50_400_800

   image_data --sernum 100 --serdes "b0" out/b0 tmp/b0
   image_data --sernum 101 --serdes "b50" out/b50 tmp/b1
   image_data --sernum 102 --serdes "b100" out/b100 tmp/b2

   rm -r tmp

Segment an image, display image with segmented ROI in red
---------------------------------------------------------

The following example will let the user segment an image (using get_roi()
method).
An RGB version of the original image is produced by the get_roi() method,
where each of the RGB components are set to the original gray scale value.

`segment_indices' address the selected area, and is
used to set the green (1) and blue components (2) to zero.
Hence, the [1:] slicing of the color components RGB.

Finally, the color image is display with the segmented area in red.

.. code-block:: python

    from imagedata.series import Series

    T2 = Series('801_Obl T2 TSE HR SENSE/')
    segment = T2.get_roi()

    T2rgb = T2.to_rgb()
    segment_indices = segment == 1

    # Clear green and blue components inside segmentation,
    # leaving the red component
    T2rgb[segment_indices,1:] = 0

    T2rgb.show()
