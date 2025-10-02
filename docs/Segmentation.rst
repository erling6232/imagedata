.. _Segmentation:

Segmentation DICOM objects
==========================

DICOM Segmentation objects have pixel data and are stored as NumPy.ndarray.

Example:

.. code-block:: python

    from imagedata import Study
    s = Study('.')

    img = tree = aorta = None
    # Locate img dataset
    for seriesUID in s:
        si = s[seriesUID]
        if 'Segmentations' not in si.seriesDescription:
            img = si
            break
    assert img is not None, "No image dataset found"

    # Locate segmentation datasets
    for seriesUID in s:
        si = s[seriesUID]
        if 'Segmentations' in si.seriesDescription:
            for i, descr in enumerate(si.axes[0]):
                if 'Aorta' in '{}'.format(descr):
                    assert aorta is None, "Multiple aorta masks found"
                    aorta = si[i]
                    d = si.header.datasets[i]
                    parent = d.ReferencedSeriesSequence[0].SeriesInstanceUID
                    if parent == img.seriesInstanceUID:
                        aorta.header.add_geometry(img.header)
                elif 'CoronaryTree' in '{}'.format(descr):
                    assert tree is None, "Multiple coronary tree masks found"
                    tree = si[i]
                    d = si.header.datasets[i]
                    parent = d.ReferencedSeriesSequence[0].SeriesInstanceUID
                    if parent == img.seriesInstanceUID:
                        tree.header.add_geometry(img.header)

    assert aorta is not None, "No aorta mask found"
    assert tree is not None, "No coronary tree mask found"
    assert parent == img.seriesInstanceUID, "Mask does not belong to series"

    fused_aorta = img.fuse_mask(1 - aorta)
    fused_tree = img.fuse_mask(tree)
    fused_tree.show([tree, fused_aorta, aorta], link=True)
