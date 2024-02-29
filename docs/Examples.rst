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

    from imagedata import Series

    dwi = Series('10_ep2d_diff_b50_400_800/', 'b')

    # dwi.tags[0] has the b values, e.g. [50, 400, 800]

    for i, b in enumerate(dwi.tags[0]):
        # Output to folders b50/, b400/ and b800/
        dwi[i].write('b{}'.format(b))

More elaborate example in python:
    To set a separate series number and description for each series,
    each volume of b values must be separate objects:

.. code-block:: python

    from imagedata import Series

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
    Split b values using `--odir multi` parameter. Each b value
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

Segment an image, display fused image with segmented ROI in red
---------------------------------------------------------------

The following example will let the user segment an image (using get_roi()
method).
A fused image where the segmented area is enhanced with red color is produced by fuse_segment().

Finally, the color image is displayed.

.. code-block:: python

    from imagedata import Series

    T2 = Series('801_Obl T2 TSE HR SENSE/')
    segment = T2.get_roi()

    T2rgb = T2.fuse_mask(segment)

    T2rgb.show()


Draw a time curve when mask is moved
------------------------------------

.. code-block:: python

    import numpy as np
    import matplotlib.pyplot as plt
    from imagedata.viewer import grid_from_roi

    def plot_aif(idx, tag, vertices):
        # Called from Viewer when vertices are modified
        # Clear any previous axis plot
        plt.sca(ax[1])
        plt.cla()
        if vertices is None:
            return
        mask = grid_from_roi(si, {idx: vertices})
        curve = np.sum(si, axis=(1, 2, 3), where=mask == 1) / np.count_nonzero(mask)
        ax[1].plot(curve, label='AIF')
        ax[1].legend()
        ax[1].figure.canvas.draw()

    si = Series('data', 'time')
    fig, ax = plt.subplots(1, 2)
    mask = si.get_roi(ax=ax[0], onselect=plot_aif)


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
    from imagedata import Series
    import nipype.interfaces.fsl as fsl


    def mcflirt(moving, fixed):
        """Register dynamic series using FSL MCFLIRT
        Args:
            moving: moving (Series)
            fixed:  reference volume, either
                index into moving (Series), or
                separate volume (int)
        Returns:
            registered Series
        """

        if issubclass(type(fixed), Series):
            if fixed.ndim == 3:
                ref = fixed
                ref_volume = fixed
            else:
                raise ValueError('Fixed volume should be 3D (is {})'.format(fixed.ndim))
        else:
            assert fixed >= 0 and fixed < len(moving), "Wrong fixed index {}".format(fixed)
            ref = fixed
            ref_volume = moving[ref]
        print('\nPreparing for MCFLIRT ...')
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            fixed_file = p / 'fixed.nii.gz'
            moving_file = p / 'moving.nii.gz'
            tmp_out = p / 'out.nii.gz'
            moving.write(moving_file, formats=['nifti'])

            print('MCFLIRT running ...')

            mcflt = fsl.MCFLIRT()
            mcflt.inputs.in_file = str(moving_file)
            if issubclass(type(ref), Series):
                ref.write(fixed_file, formats=['nifti'])
                mcflt.inputs.ref_file = str(fixed_file)
            else:
                mcflt.inputs.ref_vol = ref
            mcflt.inputs.out_file = str(tmp_out)
            mcflt.inputs.cost = "corratio"  # "normcorr"
            # mcflt.inputs.cost     = "normcorr"
            print('{}'.format(mcflt.cmdline))
            result = mcflt.run()
            print('Result code: {}'.format(result.runtime.returncode))

            try:
                out = Series(tmp_out, input_order=moving.input_order, template=moving, geometry=ref_volume)
            except Exception as e:
                print(e)
            out.tags = moving.tags
            out.axes = moving.axes
            out.seriesDescription = 'MCFLIRT {}'.format(mcflt.inputs.cost)
        print('MCFLIRT ended.\n')
        return out
