"""ROI objects
"""

# Copyright (c) 2017-2024 Erling Andersen, Haukeland University Hospital, Bergen, Norway

import logging
import numpy as np
import math
from abc import ABCMeta, abstractmethod
from .draw_antialiased import draw_circle_mask, draw_polygon_mask

logger = logging.getLogger(__name__)


class ROI(object, metaclass=ABCMeta):
    """General ROI object.

    Attributes:
        points
        points_matrix
        slice
        label
        stu_ins_uid
        ser_ins_uid
        sop_ins_uid
    """

    def __init__(self, label, stu_ins_uid, ser_ins_uid, sop_ins_uid):
        object.__init__(self)
        self.points = None
        self.points_matrix = None
        self.slice = None
        self.label = label
        self.stu_ins_uid = stu_ins_uid
        self.ser_ins_uid = ser_ins_uid
        self.sop_ins_uid = sop_ins_uid

    def __str__(self):
        return "ROI {} points: {}\n".format(self.label, len(self.points)) + \
            "StudyInstanceUID: {}\n".format(self.stu_ins_uid) + \
            "SeriesInstanceUID: {}\n".format(self.ser_ins_uid) + \
            "SOPInstanceUID: {}\n".format(self.sop_ins_uid)

    @abstractmethod
    def get_points_cm(self):
        """Get ROI points as Numpy array in cm real coordinates

        Returns:
            np.array([points,3]) where each row is (z,y,x) in cm
        """
        pass

    @abstractmethod
    def get_points_matrix(self, si):
        """Get ROI points as Numpy array in matrix coordinates

        Args:
            si: Series with transformation matrix
        Returns:
            np.array([points,3]) where each row is (z,y,x) in matrix coordinates
        """
        pass

    @staticmethod
    def create_canvas(shape, mode=bool):
        """Make a 2D [ny,nx] canvas

        Args:
            shape:
                4D [nt,nz,ny,nx] or 3D [nz,ny,nx] shape
            mode:
                bool for binary image, np.uint8 for 8-bit grayscale image
        """
        if len(shape) == 3:
            nz, ny, nx = shape
        elif len(shape) == 4:
            nt, nz, ny, nx = shape
        else:
            raise ValueError("Shape has wrong length: {}".format(len(shape)))

        return np.zeros((ny, nx), dtype=mode)

    @abstractmethod
    def draw_roi_on_canvas(self, canvas, colour=1, threshold=0.5, fill=False):
        """Make a 2D mask [ny,nx] on canvas

        Args:
            canvas: 2D [ny,nx]
            colour: mask color
            fill: whether to fill interior of ROI
            self.points_matrix: polygon points
            self._slice
        Returns:
            binary img of shape [ny,nx]
        """
        raise ValueError("Abstract ROI::draw_roi_on_canvas should not be called!")
        pass

    def draw_roi_on_numpy(self, mask, colour=1, threshold=0.5, fill=True):
        """Draw the ROI on an existing numpy array

        Args:
            mask: numpy array [nz,ny,nx]
            colour: mask colour (int)
            threshold: alpha blend (float)
            fill: whether to fill interior of ROI (bool)
        Returns:
            mask: modified numpy array [nz,ny,nx]
        """
        canvas = self.create_canvas(mask.shape)
        self.draw_roi_on_canvas(canvas, colour=colour, threshold=threshold, fill=fill)
        if self.slice:
            logger.debug('mask: {}, canvas: {}, slice: {}'.format(
                mask.shape, canvas.shape, self.slice))
            mask[self.slice, :, :] = np.logical_or(mask[self.slice, :, :], canvas)
        return mask

    def verify_all_voxels_in_slice(self):
        if self.points_matrix is None:
            raise ValueError("Matrix points have not been calculated.")
        iz, y, x = self.points_matrix[0]
        for p in self.points_matrix:
            z, y, x = p
            if z != iz:
                logger.debug("Point %d,%d,%d is not in _slice %d." % (z, y, x, iz))
                # raise ValueError("Point %d,%d,%d is not in _slice %d." % (z,y,x,iz))
        logger.debug('verify_all_voxels_in_slice: {}'.format(iz))
        return iz

    @staticmethod
    def transform_data_points_to_voxels(si, points):
        voxels = []
        for point in points:
            t = si.getVoxelForPosition(np.array(point))
            voxels.append(t)
        return np.asarray(voxels).reshape(len(points), 3)

    def get_timepoint(self, si):
        _index = None
        for key in si.SOPInstanceUIDs:
            if self.sop_ins_uid == si.SOPInstanceUIDs[key]:
                _index = key
                break
        if _index is not None:
            return _index[0]
        raise IndexError("SOP Instance UID {} not found".format(self.sop_ins_uid))


class PolygonROI(ROI):
    """Polygon ROI object.

    Attributes:
        points
    """

    def __init__(self, polygon, label, stu_ins_uid, ser_ins_uid, sop_ins_uid):
        super(PolygonROI, self).__init__(label, stu_ins_uid, ser_ins_uid, sop_ins_uid)
        self.points = polygon

    def get_points_cm(self):
        """Get ROI points as Numpy array in cm real coordinates

        Returns:
            np.array((points,3)) where each row is (z,y,x) in cm
        """
        return self.points

    def get_points_matrix(self, si):
        """Get ROI points as Numpy array in matrix coordinates

        Args:
            si: Series with transformation matrix
        Returns:
            np.array((points,3)) where each row is (z,y,x) in matrix coordinates
        """
        if self.points is None:
            _ = self.get_points_cm()
        if self.points_matrix is None:
            self.points_matrix = self.transform_data_points_to_voxels(si, self.points)
        return self.points_matrix

    def draw_roi_on_canvas(self, canvas, colour=1, threshold=0.5, fill=False):
        """Make a 2D mask [ny,nx] on canvas

        Args:
            canvas: 2D [ny,nx]
            self.points_matrix: polygon points
            self._slice
            fill: whether to fill polygon interior
        Returns:
            canvas
        """

        self.slice = self.verify_all_voxels_in_slice()

        polygon2D = []
        for p in self.points_matrix:
            z, y, x = p
            polygon2D.append((x, y))

        draw_polygon_mask(canvas, polygon2D, colour, threshold, fill)


class EllipseROI(ROI):
    """Ellipse ROI object.

    Attributes:
        centre_cm
        angles_cm
        thickness_cm
        radius_cm
        radius_matrix
    """

    def __init__(self, centre, angles, thickness, label, stu_ins_uid, ser_ins_uid, sop_ins_uid):
        super(EllipseROI, self).__init__(label, stu_ins_uid, ser_ins_uid, sop_ins_uid)
        assert centre.shape == (1, 3), "Wrong shape of centre: " + str(centre.shape)
        self.centre_cm = centre
        self.angles_cm = angles
        self.thickness_cm = thickness
        angle1 = angles[1]
        self.radius_cm = math.sqrt(angle1[0] * angle1[0] +
                                   angle1[1] * angle1[1] +
                                   angle1[2] * angle1[2])
        self.radius_matrix = None

    def get_points_cm(self):
        """Get ROI points as Numpy array in cm real coordinates

        Returns:
            np.array((points,3)) where each row is (z,y,x) in cm
        """
        if self.points is None:
            self.points = np.zeros([25, 3])
            self.points[0, :] = self.centre_cm[0, :]
            i = 1
            for angle in np.arange(0.0, 2 * math.pi, math.pi / 12):
                self.points[i, :] =\
                    self.centre_cm +\
                    self.radius_cm * np.array((math.sin(angle), 0, math.cos(angle)))
                i += 1
        return self.points

    def get_points_matrix(self, si):
        """Get ROI points as Numpy array in matrix coordinates

        Args:
            si: Series with transformation matrix
        Returns:
            np.array((points,3)) where each row is (z,y,x) in matrix coordinates
        """
        if self.points is None:
            _ = self.get_points_cm()
        if self.points_matrix is None:
            self.points_matrix = self.transform_data_points_to_voxels(si, self.points)
            self.slice = self.verify_all_voxels_in_slice()
            # Point on circle along x axis in cm ...
            adjacent_cm = self.centre_cm + self.radius_cm * np.array((0, 0, 1))
            # ... and in voxel coordinates
            adjacent_matrix = self.transform_data_points_to_voxels(si, adjacent_cm)
            # Radius in voxel coordinates is distance in x between centre and periphery
            self.radius_matrix = abs(adjacent_matrix[0][2] - self.points_matrix[0][2])
        return self.points_matrix

    def draw_roi_on_canvas(self, canvas, colour=1, threshold=0.5, fill=False):
        """Make a 2D mask [ny,nx] on canvas

        Args:
            canvas: 2D [ny,nx]
            fill: fill color
            self.points_matrix: polygon points
            self._slice
        Returns:
            canvas
        """

        self.slice = self.verify_all_voxels_in_slice()

        centre = self.points_matrix[0]
        yc, xc = centre[1], centre[2]
        radius = self.radius_matrix

        if radius > 0:
            draw_circle_mask(canvas, xc, yc, radius, colour, threshold, fill=fill)
        else:
            canvas[yc, xc] = colour
