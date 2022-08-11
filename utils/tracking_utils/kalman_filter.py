"""
code from
"""

import numpy as np
import scipy.linalg

class KalmanFilter(object):
    """
    A simple Kalman filter for tracking bound boxes in lidar space
    The N-dimensional state space
        x, y, vx, vy
    """

    def __init__(self):
        pass

    def initiate(self, measurement):
        """
            Create track form unassociated measurement
            Parameters
            ---------------
            measurement : ndarray
                Bounding box corrdinates (x, y, a, h)
            ---------------
        """
        means_pos = measurement
