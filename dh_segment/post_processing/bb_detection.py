#!/usr/bin/env python

import cv2
import numpy as np
import math
from shapely import geometry

def validate_box(boxes_mask, box: np.array,min_area) -> (np.array, float):
    """

    :param box: array of 4 coordinates with format [[x1,y1], ..., [x4,y4]]
    :return: (box, area)
    """
    polygon = geometry.Polygon([point for point in box])
    if polygon.area > min_area * boxes_mask.size:

        # Correct out of range corners
        box = np.maximum(box, 0)
        box = np.stack((np.minimum(box[:, 0], boxes_mask.shape[1]),
                        np.minimum(box[:, 1], boxes_mask.shape[0])), axis=1)

        # return box
        return box, polygon.area


def find_bounding_boxes(image_mask: np.ndarray,
                           min_area: float=0.0,
                           n_max_polygons: int=math.inf) -> list:
    """
    Finds the shapes in a binary mask and returns their coordinates as polygons.

    :param image_mask: Uint8 binary 2D array
    :param min_area: minimum area the polygon should have in order to be considered as valid
                (value within [0,1] representing a percent of the total size of the image)
    :param n_max_polygons: maximum number of boxes that can be found (default inf).
                        This will select n_max_boxes with largest area.
    :return: list of length n_max_polygons containing polygon's n coordinates [[x1, y1], ... [xn, yn]]
    """

    contours, _ = cv2.findContours(image_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours is None:
        print('No contour found')
        return None
    boxes = []

    for c in contours:
        if len(c) < 4:  # A polygon cannot have less than 3 points
            continue
        
        # rect = cv2.minAreaRect(c)
        # box = cv2.boxPoints(rect).astype(np.int32)
        x,y,w,h = cv2.boundingRect(c)
        box = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=int)
        if validate_box(image_mask, box,min_area) != None:
            boxes.append(box)


    # sort by area
    # boxes = [bb for bb in boxes if bb is not None]
    # boxes = sorted(boxes, key=lambda x: x[1], reverse=True)

    if boxes:
        return [bb for i, bb in enumerate(boxes) if i <= n_max_polygons]
    else:
        return None
