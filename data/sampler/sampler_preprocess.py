import abc
import sys
import time
from collections import OrderedDict
from functools import reduce

import numba
import numpy as np

import data.data_utils as box_np_ops

def random_flip_both(gt_boxes, points, pc, x_enable, y_enable, flip_coor=None):
    # x flip
    if x_enable:
        gt_boxes[:, 1] = -gt_boxes[:, 1]   # y
        gt_boxes[:, 6] = -gt_boxes[:, 6] + np.pi  # yaw
        points[:, 1] = -points[:, 1]
        pc[:, 1] = -pc[:, 1]
        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, r, vx, vy, cat
            gt_boxes[:, 8] = -gt_boxes[:, 8]
    #
    # y flip
    if y_enable:
        if flip_coor is None:
            gt_boxes[:, 0] = -gt_boxes[:, 0]
            points[:, 0] = -points[:, 0]
            pc[:, 0] = -pc[:, 0]
        else:
            gt_boxes[:, 0] = flip_coor * 2 - gt_boxes[:, 0]
            points[:, 0] = flip_coor * 2 - points[:, 0]

        gt_boxes[:, 6] = -gt_boxes[:, 6] + 2 * np.pi  # TODO: CHECK THIS

        if gt_boxes.shape[1] > 7:  # y axis: x, y, z, w, h, l, r, vx, vy, cat
            gt_boxes[:, 7] = -gt_boxes[:, 7]

    return gt_boxes, points, pc


def global_rotation(gt_boxes, points, pc, noise_rotation):
    points[:, :3] = box_np_ops.rotation_points_single_angle(
        points[:, :3], noise_rotation, axis=2
    )
    pc[:, :3] = box_np_ops.rotation_points_single_angle(
        pc[:, :3], noise_rotation, axis=2
    )
    gt_boxes[:, :3] = box_np_ops.rotation_points_single_angle(
        gt_boxes[:, :3], noise_rotation, axis=2
    )
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] = box_np_ops.rotation_points_single_angle(
            np.hstack([gt_boxes[:, 7:9], np.zeros((gt_boxes.shape[0], 1))]),
            noise_rotation,
            axis=2,
        )[:, :2]
    gt_boxes[:, 6] -= noise_rotation
    return gt_boxes, points, pc


def global_scaling_v2(gt_boxes, points, pc, noise_scale):
    points[:, :3] *= noise_scale
    pc[:, :3] *= noise_scale
    gt_boxes[:, :6] *= noise_scale  # x, y, z, h, w, l
    if gt_boxes.shape[1] > 7:
        gt_boxes[:, 7:9] *= noise_scale  # vx, vy
    return gt_boxes, points, pc


def global_translate_(gt_boxes, points, pc, noise_translate):
    """
    Apply global translation to gt_boxes and points.
    """
    points[:, :3] += noise_translate
    pc[:, :3] += noise_translate
    gt_boxes[:, :3] += noise_translate

    return gt_boxes, points, pc