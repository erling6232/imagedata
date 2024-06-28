.. _Jupyter_Notebook:

Jupyter Notebook
================

Showing image series
--------------------

Always start your notebook with:

.. code-block:: python

    %matplotlib ipympl

Let's start by creating some random data:

.. code-block:: python

    from imagedata import Series
    from numpy.random import default_rng
    rng = default_rng()
    s = Series(rng.standard_normal(4*3*128*128).reshape((4,3,128,128)))

Next, show the image series in the notebook.
Notice how you can step through the slices and time points using the arrow keys.
When using mouse scrolling, keep `shift` key pressed to avoid scrolling the window frame.

.. code-block:: python

    s.show()

Drawing a mask
--------------

When drawing a mask using get_roi(), the notebook will go ahead without waiting for
the interactive drawing. First draw the mask:

.. code-block:: python

    s.get_roi()

When the mask is drawn (possibly in several slices), execute the get_roi_mask() to get
the final mask, and show the mask using roi.show():

.. code-block:: python

    roi = s.get_roi_mask()
    roi.show()
