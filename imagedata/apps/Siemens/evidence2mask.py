"""Read Siemens Syngo.VIA Basic Reading
"""

# Copyright (c) 2017-2020 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import warnings, math, sys, os
import numpy as np
import logging
import scipy.cluster.vq
import matplotlib.pyplot as plt
#import bitarray
import scipy.misc
from imagedata.apps.Siemens.ROI import PolygonROI, EllipseROI

"""
def save_rois(rois, filename):
    f = open("/home/root/x.txt", "w")
    for roi in rois:
        for p in roi.get_points_cm():
            save_point(f, p)
    f.close()

def save_point(f, x):
    f.write("%9.4f %9.4f %9.4f\n" % (x[0], x[1], x[2]))

def save_x(x):
    f = open("/home/root/xx.txt", "a")
    f.write("%9.4f %9.4f %9.4f\n" % (x[0], x[1], x[2]))
    f.close()

def scan_local_options():
    global separate_laterality

    separate = "combined"
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == "-laterality":
            del sys.argv[i]
            separate = sys.argv[i]
            del sys.argv[i]
        else:
            i += 1
    if separate not in ("combined", "separate"):
        print("Unknown laterality argument: {}".format(separate))
        sys.exit(1)
    separate_laterality = (separate == "separate")
"""

def xyz_to_zyx(polygon):
    """Swap (x,y,z) with (z,y,x) for each point
    Input:
    - polygon: np.array shape (n,3)
    Output:
    - rearranged np.array shape (n,3)
    """
    assert polygon.ndim == 2, "Array ndim should be 2, not {}".format(polygon.ndim)
    newPolygon = np.zeros_like(polygon)
    for i in range(polygon.shape[0]):
        x,y,z = polygon[i,:]
        newPolygon[i,:] = z,y,x
    return newPolygon

def evidence2roi(im, separate_laterality=False, uid_table=None, content={}):
    # def get_rois(im, separate_laterality=False, uid_table=None, content={}):

    """
    overrows = im[(0x6000, 0x0010)].value
    overcols = im[(0x6000, 0x0011)].value
    overkey=(0x6000, 0x3000)
    overlay=im[overkey].value

    a=bitarray.bitarray(overlay, endian='little')

    while len(a) < len(overlay)*8:
            a.append(0)

    c = np.array(a.tolist())
    c = c.astype(np.uint8)
    #c = c.reshape((overcols, overrows))
    c = c.reshape((overrows, overcols))
    """

    #print("get_rois: {}".format(im.InstanceNumber))
    #content = {}
    rois = []	# List of ROI tuples

    seqkey = (0x0029, 0x1102)
    if seqkey in im:
        presentationSQ = im[seqkey]
    else:
        seqkey = (0x0029, 0x1202)
        if seqkey not in im:
            logging.debug("get_rois: no presentation sequence (0x0029, 0x1202)")
            return (rois, content)
        presentationSQ = im[seqkey]

    try:
        measurement = presentationSQ[0][(0x0029, 0x10a7)]
    except KeyError:
        logging.debug("get_rois: no measurement sequence (0x0029, 0x10a7)")
        #return (rois, content)
        raise KeyError('No measurement sequence (0x0029, 0x10a7) in input data')

    try:
        findings = measurement[0][(0x0029, 0x1031)]
    except KeyError:
        logging.error('get_rois: no findings attribute (0x0029, 0x1031) in input data')
        raise KeyError('No findings attribute (0x0029, 0x1031) in input data')

    contentSQ = presentationSQ[0][(0x0029,0x10a9)]
    content['label'] = contentSQ[0][(0x0070,0x0080)].value
    content['description'] = contentSQ[0][(0x0070,0x0081)].value
    content['date'   ] = contentSQ[0][(0x0070,0x0082)].value
    content['time'   ] = contentSQ[0][(0x0070,0x0083)].value
    content['creator'] = presentationSQ[0][(0x0070,0x0084)].value
    #print("get_rois: creator: {}".format(content['creator']))

    for finding in findings:
        referenced_frame_seq = finding[(0x0029, 0x109a)][0]
        if referenced_frame_seq[(0x0029, 0x1038)].VM > 1:
            referenced_syngo_uid = []
            for u in referenced_frame_seq[(0x0029, 0x1038)].value:
                referenced_syngo_uid.append(u)
            referenced_syngo_uid = referenced_frame_seq[(0x0029, 0x1038)].value
        else:
            referenced_syngo_uid = referenced_frame_seq[(0x0029, 0x1038)].value.decode().split('\\')
        if uid_table:
            stu_ins_uid = uid_table[referenced_syngo_uid[0]]
            ser_ins_uid = uid_table[referenced_syngo_uid[1]]
            sop_ins_uid = uid_table[referenced_syngo_uid[2]]
        else:
            stu_ins_uid = referenced_syngo_uid[0]
            ser_ins_uid = referenced_syngo_uid[1]
            sop_ins_uid = referenced_syngo_uid[2]
        #sop_cla_uid = referenced_syngo_uid[3] # '1.2.840.10008.5.1.4.1.1.4'
        # ROI number
        #if finding[(0x0029, 0x1035)].VR == 'IS':
        meas_appl_number = int(finding[(0x0029, 0x1035)].value)
        roi_type_value   = finding[(0x0029, 0x1032)].value.strip()
        try:
            roi_name         = finding[(0x0029, 0x1030)].value.decode()
        except KeyError:
            roi_name         = 'NONAME'
        logging.info("Finding: {} {} {}".format(content['creator'], roi_name, roi_type_value))
        if roi_type_value == 'PolygonApplication3D' or roi_type_value == 'FreehandApplication3D':
            output = finding[(0x0029,0x1096)]
            if output.VR == "UN":
                meas_data_points = np.fromstring(output.value, dtype='float32')
            else:
                meas_data_points = np.array(output.value)
            polygon = meas_data_points.reshape((meas_data_points.size//3, 3))
            logging.debug("XYZ: {}".format(polygon[0]))
            polygon = xyz_to_zyx(polygon)
            logging.debug("ZYX: {}".format(polygon[0]))

            rois.append(PolygonROI(polygon, roi_name, stu_ins_uid, ser_ins_uid, sop_ins_uid))
            logging.debug('ROI {}: {} points'.format(meas_appl_number, len(polygon)//3))
            #print('ROI SOPInsUID: {}'.format(sop_ins_uid))
        elif roi_type_value == 'EllipseApplication3D':
            # Ellipsis centre
            output = finding[(0x0029,0x1096)]
            if output.VR == "UN":
                meas_data_points = np.fromstring(output.value, dtype='float32')
            else:
                meas_data_points = np.array(output.value)
            centre = meas_data_points.reshape((meas_data_points.size//3, 3))
            centre = xyz_to_zyx(centre)
            #print("ellipse centre:", centre)
            # Ellipsis angles
            output = finding[(0x0029,0x1097)]
            if output.VR == "UN":
                meas_data_points = np.fromstring(output.value, dtype='float32')
            else:
                meas_data_points = np.array(output.value)
            angles = meas_data_points.reshape((meas_data_points.size//3, 3))
            # Ellipsis thickness
            output = finding[(0x0029,0x1099)]
            if output.VR == "UN":
                meas_data_points = np.fromstring(output.value, dtype='float32')
            else:
                meas_data_points = np.array(output.value)
            thickness = meas_data_points

            rois.append(EllipseROI(centre, angles, thickness, roi_name, stu_ins_uid, ser_ins_uid, sop_ins_uid))
        elif roi_type_value == 'StandaloneTextApplication3D':
            logging.warning("Standalone Text ROI not implemented.")
            pass
        elif roi_type_value == 'DistanceLineApplication3D':
            logging.warning("DistanceLineApplication3D ROI not implemented.")
            pass
        else:
            raise ValueError("ROI type %s not implemented." % roi_type_value)

    return (rois, content)

def make_mask_in_slice(roi_type, si, points, shape):
    # Make a 3D mask [nz,ny,nx]

    if roi_type == 'polygon':
        # polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
        # width = ?
        # height = ?

        for p in points:
            save_x(p)
        polygon = transform_data_points_to_voxels(si, points)
        points_matrix = roi.get_points_matrix(si)
        """
        print("polygon: points cm    {}".format(points))
        print("polygon: points matrix", polygon)
        """
        slice = verify_all_voxels_in_slice(points_matrix)

        mask = np.zeros(shape[1:], dtype=np.bool)

        polygon2D = []
        for p in points_matrix:
            z,y,x = p
            polygon2D.append((x,y))

        img = Image.new('L', (shape[3], shape[2]), 0)
        ImageDraw.Draw(img).polygon(polygon2D, outline=0, fill=1)
    elif roi_type == 'ellipse':
        #print("make_mask_in_slice: points", len(points), points)
        (centre_cm, angles, thickness) = points
        centre = transform_data_points_to_voxels(si, centre_cm)[0]
        #print("make_mask_in_slice: centre (cm)", centre_cm)
        #print("make_mask_in_slice: centre     ", centre)
        slice = centre[0]
        #print("make_mask_in_slice: slice ", slice)
        mask = np.zeros(shape[1:], dtype=np.bool)

        #print("make_mask_in_slice: angles", angles.shape, angles)
        #print("make_mask_in_slice: angles", transform_data_points_to_voxels(si, angles))

        angle1=angles[1]
        radius_cm = math.sqrt(angle1[0]*angle1[0]+angle1[1]*angle1[1]+angle1[2]*angle1[2])
        #print("make_mask_in_slice: radius_cm", radius_cm)
        adjacent_cm = centre_cm + np.array((0,0,radius_cm))
        save_x((centre_cm + np.array((0,0,radius_cm)))[0])
        save_x((centre_cm - np.array((0,0,radius_cm)))[0])
        save_x((centre_cm + np.array((radius_cm,0,0)))[0])
        save_x((centre_cm - np.array((radius_cm,0,0)))[0])
        #print("make_mask_in_slice: adjacent_cm", adjacent_cm)
        adjacent = transform_data_points_to_voxels(si, adjacent_cm)[0]
        #print("make_mask_in_slice: adjacent ", adjacent)
        radius = abs(centre[1]-adjacent[1])
        #print("make_mask_in_slice: radius   ", radius)

        """
        ellipse2D = []
        for p in ellipse3D:
            z,y,x = p
            ellipse2D.append((x,y))
        print("make_mask_in_slice: ellipse2D {}".format(ellipse2D))
        """

        yc,xc = centre[1],centre[2]
        ellipse2D = (xc-radius, yc-radius, xc+radius, yc+radius)
        img = Image.new('L', (shape[3], shape[2]), 0)
        ImageDraw.Draw(img).ellipse(ellipse2D, outline=0, fill=1)
    mask[slice,:,:] = np.array(img)
    return mask

def transform_data_points_to_voxels(si, meas_data_points):
    polygon = []
    for point in meas_data_points:
        x,y,z = si.getVoxelForPosition(point)
        polygon.append((z,y,x))
    return polygon

#def evidence2roi(im, separate_laterality=False, uid_table=None, content={}):
#    #tg,fname,im = reading.DicomHeaderDict[0][0]
#
#    # Extract ROIs from syngo.via object
#    rois,content = get_rois(im, separate_laterality, uid_table, content)
#
#    save_rois(rois, '/home/root/x.txt')
#
#    return rois,content

'''
def roi2mask(mask, content, roi, hdr, si, separate_laterality=False):
    """Convert each roi into mask.
    """

    timeline = hdr.getTagList()

    # Create masks for each laterality
    for laterality in labelled_rois.keys():
        if not laterality in mask:
            # Create mask by building on one slice at a time
            mask[laterality] = np.zeros(si.shape[1:], dtype=np.bool)
            # mask[laterality] = np.zeros(si.shape[1:], dtype=np.uint8)
        for key in labelled_rois[laterality].keys():
            roi_type, meas_data_points, stu_ins_uid, ser_ins_uid, sop_ins_uid = labelled_rois[laterality][key]
            try:
                slice,tag,fname,im = hdr.search_sop_ins_uid(sop_ins_uid)
            except:
                continue
            #print("SOPInsUID: %s" % sop_ins_uid)
            print("  Laterality: %s, slice: %d, tag %s, timepoint %d" % (laterality, slice, tag, timeline.index(tag)))
            print("  Date %s, time %s, creator %s" % (content['date'], content['time'], content['creator']))
            mask[laterality] = np.logical_or(mask[laterality], make_mask_in_slice(roi_type, hdr, meas_data_points, si.shape))
            content['timepoint'] = timeline.index(tag)
'''
