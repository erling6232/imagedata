.. _Examples:

Examples
========

Split MRI diffusion series into separate series for each b value
----------------------------------------------------------------

.. code-block:: python

    from imagedata.series import Series

    dwi = Series('10_ep2d_diff_b50_400_800_p2/', 'b')

    # dwi.tags[0] has the b values, e.g. [50, 400, 800]

    for i, b in enumerate(dwi.tags[0]):
        # Output to folders b50/, b400/ and b800/
        dwi[i].write('b{}'.format(b))

