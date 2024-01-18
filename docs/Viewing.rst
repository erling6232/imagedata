.. _Viewing:

Viewing Data
===================

Viewing
"""""""

The basic method to view a Series object is
the :meth:`imagedata.Series.show` method:

.. code-block:: python

    a = Series(...)
    a.show()

View multiple series
--------------------

This example will view 4 series in 4 viewports.
The ``link=True`` parameter links the scrolling of these series,
such that they all show the same slice (assuming that the series have
the same shape).
The viewer will decide the layout of viewports depending on the number of series.

.. code-block:: python

   a.show([b, c, d], link=True)

When there is a long list of series, the viewer may decide to limit the number of viewports.
The series will be shown on multiple pages, and the user can page through the series list.

To override the number of viewports, define these in advance using matplotlib subplots.
*E.g.*, to set up a 2x2 viewport and display 8 series:

.. code-block:: python

   fig, ax = plt.subplots(nrows=2, ncols=2)
   images = [b, c, d, e, f, g, h]
   a.show(images, link=True, ax=ax)


Draw region of interest (ROI)
"""""""""""""""""""""""""""""

The :meth:`imagedata.Series.get_roi` method will let the user draw a ROI outline as a polygon.
The returned mask will have the same shape, and be one (1) inside the ROI, and zero (0) elsewhere.

.. code-block:: python

   mask = a.get_roi()

The ROI polygon can be adjusted using the mouse:

  * Move a polygon vertex using left mouse key
  * Move the polygon outline using shift key and left mouse key

The vertices of the ROI polygon can be retrieved by:

.. code-block:: python

   mask, vertices = a.get_roi(vertices=True)

These vertices can be used to present the user with an existing ROI, allowing modification interactively:

.. code-block:: python

   mask, vertices = a.get_roi(roi=vertices, vertices=True)

Alternatively, an existing binary ROI mask can be used as input.
The binary ROI will be converted to a list of vertices.
Depending on the input data, the polygon may differ somewhat from the expected result!

.. code-block:: python

   mask, vertices = a.get_roi(roi=some_mask, vertices=True)

Special 4D applications
-----------------------

Normally,
the :meth:`imagedata.Series.get_roi` method
will draw a 3D mask on a 4D series.
The user can step through the 4D series, modifying the mask at any tag step.
The final 3D mask will be returned.

In a time-resolved series with patient motion, it may be useful to follow an organ from time point
to time point.
The ``follow=True`` flag will let the user draw a ROI on the first time point.
The ROI is then copied to the next time point, allowing the user to reposition the vertices
before proceeding to next time step.
The final ROI will be a 4D mask.

Similarly, when the organ movement is mostly in-plane, the
``single=True`` will draw ROi in a single slice only.
The final ROI will be a 4D mask.