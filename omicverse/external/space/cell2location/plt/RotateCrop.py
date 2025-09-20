r"""Class for rotating and cropping images."""
import cv2 as cv
import numpy as np


# +
def find_rectangle_corners(bottom_left, width):
    r"""Find rectangle corners given positions of left side points and width of the plot.

    Parameters
    ----------
    bottom_left :
        array with x/y positions (columns) of top left and bottom left corners (in rows).
    width :
        width of the rectangle.

    Returns
    -------

    """

    # We identify the position of remaining corners of the rectangle using basic geometry
    x1 = bottom_left[0, 1] - bottom_left[1, 1]
    y1 = bottom_left[0, 0] - bottom_left[1, 0]
    z1 = np.sqrt((np.array([x1, y1]) ** 2).sum())
    x = y1 / z1 * width
    y = x1 / z1 * width

    return np.array(
        [
            bottom_left[0, :],
            [bottom_left[0, 0] - y, bottom_left[0, 1] + x],
            [bottom_left[1, 0] - y, bottom_left[1, 1] + x],
        ]
    )


class RotateCrop:
    def __init__(self, img, corners, rotate90=0, flip_axes=True):
        r"""

        Parameters
        ----------
        img :
            Image to crop
        corners :
            Crop rectangle corners, 3 is enough
        rotate90 : int
            If you want to rotate cropped frame several times
        flip_axes : bool
            Try to switch it if cropped frame looks weird


        """

        self.img = img
        self.corners = np.array(corners).astype(np.float32)
        self.rotation_matrix = None
        self.rotate90 = rotate90
        self.flip_axes = slice(None, None, (-1) ** self.rotate90 * (1 if flip_axes else -1))

    def compute_distances(self):
        distances = []
        for i in self.corners:
            for j in self.corners:
                distances.append(np.linalg.norm(i - j).astype(np.float32))

        distances = sorted(set(distances))[1:]
        if len(distances) == 1:
            return distances[0], distances[0]
        else:
            return distances[0], distances[1]

    def get_rotation_matrix(self):
        self.width, self.height = self.compute_distances()[self.flip_axes]
        self.width, self.height = self.width.astype(np.int32), self.height.astype(np.int32)

        self.dst_pts = [[0, self.height], [0, 0], [self.width, 0], [self.width, self.height]]

        self.dst_pts = [
            self.dst_pts[self.rotate90 % 4],
            self.dst_pts[(self.rotate90 + 1) % 4],
            self.dst_pts[(self.rotate90 + 2) % 4],
            self.dst_pts[(self.rotate90 + 3) % 4],
        ]

        self.dst_pts = np.array(self.dst_pts).astype(np.float32)

        self.rotation_matrix = cv.getAffineTransform(self.corners[:3], self.dst_pts[:3])

    def crop_rotate(self):
        if self.rotation_matrix is None:
            self.get_rotation_matrix()

        return cv.warpAffine(self.img, self.rotation_matrix, (self.width, self.height))

    def rotate_points(self, points, return_mask=False):
        """

        Parameters
        ----------
        points :

        return_mask :
             (Default value = False)

        Returns
        -------

        """
        if self.rotation_matrix is None:
            self.get_rotation_matrix()

        points = np.array(points)
        new_points = []

        for p in points:
            new_p = (np.vstack((self.rotation_matrix, [0, 0, 1])) @ np.hstack((p, [1])))[:2]
            new_points.append(new_p)

        new_points = np.array(new_points)

        valid_mask = (
            (new_points[:, 0] > 0)
            & (new_points[:, 1] > 0)
            & (new_points[:, 0] < self.width)
            & (new_points[:, 1] < self.height)
        )

        valid_points = new_points[valid_mask]

        if return_mask:
            return valid_points, valid_mask
        return valid_points
