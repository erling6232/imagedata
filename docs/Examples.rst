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


Motion correction using FSL MCFLIRT
-----------------------------------

Motion correction using image registration is a process where different images of a patient
are transformed to a common reference frame.
This example uses the FSL MCFLIRT program for this task.
MCFLIRT takes NIfTI input and output. Hence, this example will write a Series instance
to a temporary NIfTI file, call MCFLIRT, then read back the resulting NIfTI file using the
original Series instance as a template for DICOM header information.

.. code-block:: python

    import tempfile
    from pathlib import Path
    from imagedata.series import Series
    import nipype.interfaces.fsl as fsl

    def mcflirt(dce, fx):
        """Register dynamic series using FSL MCFLIRT
        Args:
            dce: dynamic series [t, slice, row, column]
            fx: index of fixed volume in dce (int)
        Returns:
            registered Series
        """

        assert fx >= 0 and fx < len(dce), "Wrong fixed index {}".format(fx)
        print('\nPreparing for MCFLIRT ...')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            tmp_fixed = p / 'fixed'
            dce[fx].write(tmp_fixed, formats=['nifti'])
            fixed = list(tmp_fixed.glob('*'))[0]
            tmp_moving = p / 'moving'
            dce.write(tmp_moving, formats=['nifti'])
            moving = list(tmp_moving.glob('*'))[0]

            print('MCFLIRT running ...')
            tmp_out = p / 'out.nii.gz'

            mcflt = fsl.MCFLIRT()
            mcflt.inputs.in_file = str(moving)
            # mcflt.inputs.ref_file = str(fixed)
            mcflt.inputs.ref_vol = fx
            mcflt.inputs.out_file = str(tmp_out)
            mcflt.inputs.cost = "corratio"
            # mcflt.inputs.cost     = "normcorr"
            print('{}'.format(mcflt.cmdline))
            result = mcflt.run()

            dce2 = Series(tmp_out, input_order=dce.input_order, template=dce, geometry=dce)
            dce2.tags = dce.tags
            dce2.axes = dce.axes
            dce2.seriesDescription = 'MCFLIRT {}'.format(mcflt.inputs.cost)
        print('MCFLIRT ended.\n')
        return dce2