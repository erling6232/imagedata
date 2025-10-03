.. _Segmentation:

Segmentation DICOM objects
==========================

DICOM Segmentation objects have pixel data and are stored as NumPy.ndarray.

Typically, the segmentation objects do not provide geometry attributes.
When the referenced series instance UID is available in a Study(),
the geometry attributes of the segmentation series will be copied from
the referenced series.

Example:

.. code-block:: python

    from imagedata import Study
    s = Study('.')

    img = tree = aorta = None
    # Locate datasets
    for seriesUID in s:
        si = s[seriesUID]
        if 'Segmentations' in si.seriesDescription:
            for i, descr in enumerate(si.axes[0]):
                if 'Aorta' in '{}'.format(descr):
                    assert aorta is None, "Multiple aorta masks found"
                    aorta = si[i]
                elif 'CoronaryTree' in '{}'.format(descr):
                    assert tree is None, "Multiple coronary tree masks found"
                    tree = si[i]
        else:
            img = si

    assert img is not None, "No image dataset found"
    assert aorta is not None, "No aorta mask found"
    assert tree is not None, "No coronary tree mask found"

    fused_aorta = img.fuse_mask(1 - aorta)
    fused_tree = img.fuse_mask(tree)
    fused_tree.show([tree, fused_aorta, aorta], link=True)
