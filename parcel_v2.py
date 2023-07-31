import numpy as np
import cv2


# 0: Land, 1: Road, 3: Building, 4:Occupied 5: Unavailable
Landmarks: np.ndarray
Buildings: []


class Building:
    def __init__(self, id, points):
        self.id = id
        self.points = points