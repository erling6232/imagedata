.. _Jupyter Notebook:

Jupyter Notebook
================

Showing image series
--------------------

Always start your notebook with:

.. code-block:: python

    %matplotlib widget

Let's start by creating some random data:

.. code-block:: python

    from imagedata.series import Series
    from numpy.random import default_rng
    rng = default_rng()
    s = Series(rng.standard_normal(4*3*128*128).reshape((4,3,128,128)))

