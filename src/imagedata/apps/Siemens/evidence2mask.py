"""Read Siemens Syngo.VIA Basic Reading
"""

# Copyright (c) 2017-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import numpy as np
import logging
from .ROI import PolygonROI, EllipseROI

logger = logging.getLogger(__name__)


def _xyz_to_zyx(polygon):
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


def _get_measurement_points(output, point=True, zyx=False):
    if output.VR == "UN":
        meas_data_points = np.fromstring(output.value, dtype='float32')
    else:
        meas_data_points = np.array(output.value)
    if point:
        points = meas_data_points.reshape((meas_data_points.size // 3, 3))
    else:
        points = meas_data_points
    if point and zyx:
        logger.debug("XYZ: {}".format(points[0]))
        points = _xyz_to_zyx(points)
        logger.debug("ZYX: {}".format(points[0]))
    return points


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
        measurements = presentation_sq[0][(0x0029, 0x10a7)]
    except (TypeError, KeyError):
        try:
            p0_sq = presentation_sq[0][(0x0029, 0x10ee)]
            measurements = p0_sq[0][(0x0029, 0x10a7)]
        except (TypeError, KeyError):
            logger.debug("get_rois: no measurement sequence (0x0029, 0x10a7)")
            raise KeyError('No measurement sequence (0x0029, 0x10a7) in input data')

    findings = None
    for measurement in measurements:
        if (0x0029, 0x1031) in measurement:
            findings = measurement[(0x0029, 0x1031)]
    if findings is None:
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
        if uid_table is not None:
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
            roi_type_value = roi_type_value.decode('ascii')
        except AttributeError:
            pass
        try:
            roi_name = finding[(0x0029, 0x1030)].value.decode()
        except AttributeError:
            roi_name = finding[(0x0029, 0x1030)].value
        except KeyError:
            roi_name = 'NONAME'
        if roi_type_value not in (
            'PolygonApplication3D', 'FreehandApplication3D',
            'MarkerApplication3D', 'EllipseApplication3D',
            'StandaloneTextApplication3D', 'DistanceLineApplication3D'
        ):
            # Attempt to find roi type in advanced presentation sequence (0029,1091)
            try:
                adv_pres_seq = finding[(0x0029, 0x1091)][0]
                try:
                    roi_type_value = adv_pres_seq[(0x0029, 0x108e)].value.decode()
                    roi_name = finding[(0x0029, 0x1032)].value.decode()
                except AttributeError:
                    roi_type_value = adv_pres_seq[(0x0029, 0x108e)].value
                    roi_name = finding[(0x0029, 0x1032)].value
            except KeyError:
                pass
        logger.info("Finding: {} {} {}".format(content['creator'], roi_name, roi_type_value))
        if roi_type_value in ('PolygonApplication3D', 'FreehandApplication3D', 'ROI2D'):
            polygon = _get_measurement_points(finding[(0x0029, 0x1096)], zyx=True)
            rois.append(PolygonROI(polygon, roi_name, stu_ins_uid, ser_ins_uid, sop_ins_uid))
            logger.debug('ROI {}: {} points'.format(meas_appl_number, len(polygon) // 3))
        elif roi_type_value == 'ROI2D':
            polygon = _get_measurement_points(finding[(0x0029, 0x1096)], zyx=True)
            rois.append(PolygonROI(polygon, roi_name, stu_ins_uid, ser_ins_uid, sop_ins_uid))
            logger.debug('ROI {}: {} points'.format(meas_appl_number, len(polygon) // 3))
        elif roi_type_value == 'MarkerApplication3D':
            # marker = _get_measurement_points(finding[(0x0029, 0x1096)], zyx=True)
            logger.warning("MarkerApplication3D ROI not implemented.")
        elif roi_type_value == 'EllipseApplication3D':
            # Ellipsis centre
            centre = _get_measurement_points(finding[(0x0029, 0x1096)], zyx=True)
            # Ellipsis angles
            angles = _get_measurement_points(finding[(0x0029, 0x1097)])
            # Ellipsis thickness
            thickness = _get_measurement_points(finding[(0x0029, 0x1099)], point=False)
            rois.append(EllipseROI(
                centre, angles, thickness, roi_name, stu_ins_uid, ser_ins_uid, sop_ins_uid))
        elif roi_type_value == 'StandaloneTextApplication3D':
            logger.warning("Standalone Text ROI not implemented.")
            pass
        elif roi_type_value == 'DistanceLineApplication3D':
            logger.warning("DistanceLineApplication3D ROI not implemented.")
            pass
        else:
            logging.warning(
                "ROI type {} ({}) not implemented.".format(
                    roi_type_value, type(roi_type_value))
            )

    return rois, content
