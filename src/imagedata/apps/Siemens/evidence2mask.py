"""Read Siemens Syngo.VIA Basic Reading
"""

# Copyright (c) 2017-2020 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import math
import numpy as np
import logging
from PIL import Image, ImageDraw
from imagedata.apps.Siemens.ROI import PolygonROI, EllipseROI

logger = logging.getLogger(__name__)


def xyz_to_zyx(polygon):
    """Swap (x,y,z) with (z,y,x) for each point

    Args:
        polygon: np.array shape (n,3)
    Returns:
        rearranged np.array shape (n,3)
    """
    assert polygon.ndim == 2, "Array ndim should be 2, not {}".format(polygon.ndim)
    new_polygon = np.zeros_like(polygon)
    for i in range(polygon.shape[0]):
        x, y, z = polygon[i, :]
        new_polygon[i, :] = z, y, x
    return new_polygon


def evidence2roi(im, uid_table=None, content=None):
    # def get_rois(image, separate_laterality=False, uid_table=None, content={}):

    """
    overrows = image[(0x6000, 0x0010)].value
    overcols = image[(0x6000, 0x0011)].value
    overkey=(0x6000, 0x3000)
    overlay=image[overkey].value

    a=bitarray.bitarray(overlay, endian='little')

    while len(a) < len(overlay)*8:
            a.append(0)

    c = np.array(a.tolist())
    c = c.astype(np.uint8)
    #c = c.reshape((overcols, overrows))
    c = c.reshape((overrows, overcols))
    """

    # print("get_rois: {}".format(image.InstanceNumber))
    # content = {}
    if content is None:
        content = {}
    rois = []  # List of ROI tuples

    seqkey = (0x0029, 0x1102)
    if seqkey in im:
        presentation_sq = im[seqkey]
    else:
        seqkey = (0x0029, 0x1202)
        if seqkey not in im:
            logger.debug("get_rois: no presentation sequence (0x0029, 0x1202)")
            return rois, content
        presentation_sq = im[seqkey]

    try:
        measurement = presentation_sq[0][(0x0029, 0x10a7)]
    except KeyError:
        logger.debug("get_rois: no measurement sequence (0x0029, 0x10a7)")
        # return (rois, content)
        raise KeyError('No measurement sequence (0x0029, 0x10a7) in input data')

    try:
        findings = measurement[0][(0x0029, 0x1031)]
    except KeyError:
        logger.error('get_rois: no findings attribute (0x0029, 0x1031) in input data')
        raise KeyError('No findings attribute (0x0029, 0x1031) in input data')

    content_sq = presentation_sq[0][(0x0029, 0x10a9)]
    content['label'] = content_sq[0][(0x0070, 0x0080)].value
    content['description'] = content_sq[0][(0x0070, 0x0081)].value
    content['date'] = content_sq[0][(0x0070, 0x0082)].value
    content['time'] = content_sq[0][(0x0070, 0x0083)].value
    content['creator'] = '%s' % presentation_sq[0][(0x0070, 0x0084)].value

    for finding in findings:
        referenced_frame_seq = finding[(0x0029, 0x109a)][0]
        if referenced_frame_seq[(0x0029, 0x1038)].VM > 1:
            referenced_syngo_uid = []
            for u in referenced_frame_seq[(0x0029, 0x1038)].value:
                referenced_syngo_uid.append(u)
            referenced_syngo_uid = referenced_frame_seq[(0x0029, 0x1038)].value
        else:
            referenced_syngo_uid =\
                referenced_frame_seq[(0x0029, 0x1038)].value.decode().split('\\')
        if uid_table:
            stu_ins_uid = uid_table[referenced_syngo_uid[0]]
            ser_ins_uid = uid_table[referenced_syngo_uid[1]]
            sop_ins_uid = uid_table[referenced_syngo_uid[2]]
        else:
            stu_ins_uid = referenced_syngo_uid[0]
            ser_ins_uid = referenced_syngo_uid[1]
            sop_ins_uid = referenced_syngo_uid[2]
        # sop_cla_uid = referenced_syngo_uid[3] # '1.2.840.10008.5.1.4.1.1.4'
        # ROI number
        meas_appl_number = int(finding[(0x0029, 0x1035)].value)
        roi_type_value = finding[(0x0029, 0x1032)].value.strip()
        try:
            roi_name = finding[(0x0029, 0x1030)].value.decode()
        except AttributeError:
            roi_name = finding[(0x0029, 0x1030)].value
        except KeyError:
            roi_name = 'NONAME'
        logger.info("Finding: {} {} {}".format(content['creator'], roi_name, roi_type_value))
        if roi_type_value == 'PolygonApplication3D' or roi_type_value == 'FreehandApplication3D':
            output = finding[(0x0029, 0x1096)]
            if output.VR == "UN":
                meas_data_points = np.fromstring(output.value, dtype='float32')
            else:
                meas_data_points = np.array(output.value)
            polygon = meas_data_points.reshape((meas_data_points.size // 3, 3))
            logger.debug("XYZ: {}".format(polygon[0]))
            polygon = xyz_to_zyx(polygon)
            logger.debug("ZYX: {}".format(polygon[0]))

            rois.append(PolygonROI(polygon, roi_name, stu_ins_uid, ser_ins_uid, sop_ins_uid))
            logger.debug('ROI {}: {} points'.format(meas_appl_number, len(polygon) // 3))
        elif roi_type_value == 'EllipseApplication3D':
            # Ellipsis centre
            output = finding[(0x0029, 0x1096)]
            if output.VR == "UN":
                meas_data_points = np.fromstring(output.value, dtype='float32')
            else:
                meas_data_points = np.array(output.value)
            centre = meas_data_points.reshape((meas_data_points.size // 3, 3))
            centre = xyz_to_zyx(centre)
            # Ellipsis angles
            output = finding[(0x0029, 0x1097)]
            if output.VR == "UN":
                meas_data_points = np.fromstring(output.value, dtype='float32')
            else:
                meas_data_points = np.array(output.value)
            angles = meas_data_points.reshape((meas_data_points.size // 3, 3))
            # Ellipsis thickness
            output = finding[(0x0029, 0x1099)]
            if output.VR == "UN":
                meas_data_points = np.fromstring(output.value, dtype='float32')
            else:
                meas_data_points = np.array(output.value)
            thickness = meas_data_points

            rois.append(EllipseROI(
                centre, angles, thickness, roi_name, stu_ins_uid, ser_ins_uid, sop_ins_uid))
        elif roi_type_value == 'StandaloneTextApplication3D':
            logger.warning("Standalone Text ROI not implemented.")
            pass
        elif roi_type_value == 'DistanceLineApplication3D':
            logger.warning("DistanceLineApplication3D ROI not implemented.")
            pass
        else:
            raise ValueError("ROI type %s not implemented." % roi_type_value)

    return rois, content


def make_mask_in_slice(roi_type, si, points, shape):
    """Make a 3D mask [nz,ny,nx]
    """

    if roi_type == 'polygon':
        # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
        # width = ?
        # height = ?

        # for p in points:
        #     save_x(p)
        polygon = transform_data_points_to_voxels(si, points)
        roi = None  # TODO
        points_matrix = roi.get_points_matrix(si)  # TODO
        """
        print("polygon: points cm    {}".format(points))
        print("polygon: points matrix", polygon)
        """
        slice = polygon.verify_all_voxels_in_slice(points_matrix)  # TODO

        mask = np.zeros(shape[1:], dtype=np.bool)

        polygon2D = []
        for p in points_matrix:
            z, y, x = p
            polygon2D.append((x, y))

        img = Image.new('L', (shape[3], shape[2]), 0)
        ImageDraw.Draw(img).polygon(polygon2D, outline=0, fill=1)
    elif roi_type == 'ellipse':
        (centre_cm, angles, thickness) = points
        centre = transform_data_points_to_voxels(si, centre_cm)[0]
        slice = centre[0]
        mask = np.zeros(shape[1:], dtype=np.bool)

        angle1 = angles[1]
        radius_cm = math.sqrt(angle1[0] * angle1[0] +
                              angle1[1] * angle1[1] +
                              angle1[2] * angle1[2])
        adjacent_cm = centre_cm + np.array((0, 0, radius_cm))
        # save_x((centre_cm + np.array((0, 0, radius_cm)))[0])
        # save_x((centre_cm - np.array((0, 0, radius_cm)))[0])
        # save_x((centre_cm + np.array((radius_cm, 0, 0)))[0])
        # save_x((centre_cm - np.array((radius_cm, 0, 0)))[0])
        adjacent = transform_data_points_to_voxels(si, adjacent_cm)[0]
        radius = abs(centre[1] - adjacent[1])

        yc, xc = centre[1], centre[2]
        ellipse2D = (xc - radius, yc - radius, xc + radius, yc + radius)
        img = Image.new('L', (shape[3], shape[2]), 0)
        ImageDraw.Draw(img).ellipse(ellipse2D, outline=0, fill=1)
    mask[slice, :, :] = np.array(img)
    return mask


def transform_data_points_to_voxels(si, meas_data_points):
    polygon = []
    for point in meas_data_points:
        x, y, z = si.getVoxelForPosition(point)
        polygon.append((z, y, x))
    return polygon
