import torch
import torch.nn as nn
from torch.autograd import Function
from utils import common_utils
import torch.nn.functional as F
from . import roiaware_pool3d_cuda


def points_in_boxes_cpu(points, boxes):
    """
    Args:
        points: (num_points, 3)
        boxes: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    Returns:
        point_indices: (N, num_points)
    """
    assert boxes.shape[1] == 7
    assert points.shape[1] == 3
    points, is_numpy = common_utils.check_numpy_to_torch(points)
    boxes, is_numpy = common_utils.check_numpy_to_torch(boxes)

    point_indices = points.new_zeros((boxes.shape[0], points.shape[0]), dtype=torch.int)
    roiaware_pool3d_cuda.points_in_boxes_cpu(boxes.float().contiguous(), points.float().contiguous(), point_indices)

    return point_indices.numpy() if is_numpy else point_indices


def points_in_boxes_gpu(points, boxes):
    """
    :param points: (B, M, 3)
    :param boxes: (B, T, 7), num_valid_boxes <= T
    :return box_idxs_of_pts: (B, M), default background = -1
    """
    assert boxes.shape[0] == points.shape[0]
    assert boxes.shape[2] == 7 and points.shape[2] == 3
    batch_size, num_points, _ = points.shape

    box_idx_of_pts = points.new_zeros((batch_size, num_points), dtype=torch.int).fill_(-1)
    roiaware_pool3d_cuda.points_in_boxes_gpu(boxes.contiguous(), points.contiguous(), box_idx_of_pts)

    return box_idx_of_pts


def build_voxel_feature(points, extents, rows, cols, voxel_num, feature_num, device):
    """
    :param points: (B, M, 3)
    :param extents: point cloud range
    :param rows int, how many pixel per row and col
    :param cols int, how many pixel per row and col
    :param voxel_num, number of voxel number in z axis
    :param feature_num, number of feature pre voxel
    :param device: gpu cuda device
    :return
        voxel_feature, all voxel feature
    """
    batch_size = points.shape[0]
    num_points = points.shape[1]
    # assert batch_size == points.shape[0], "points batch_size is not equal to batch_size"
    voxel_feature_num = voxel_num * feature_num
    voxel_feature = torch.zeros(size=(batch_size, rows, cols, voxel_feature_num), device=device, dtype=torch.float)
    pts_in_voxel_position = torch.zeros(size=(batch_size, num_points), device=device, dtype=torch.int32).fill_(-1)
    #  GPU algorithm for build voxel features
    roiaware_pool3d_cuda.build_voxel_feature(points.contiguous(), rows, cols, extents.contiguous(),
                                             voxel_num, feature_num, pts_in_voxel_position, voxel_feature)

    return voxel_feature


def build_view_index(curr_frame_points,
                     extents,
                     rows,
                     cols,
                     boxes,
                     device):
    """
    @attention all of this batch size is 1
    @param curr_frame_points: point: (B, M, 3) curr frame points
    @param extents: point cloud range
    @param rows: rows int, how many pixel per row and col
    @param cols: int, how many pixel per row and col
    @param device: gpu cuda device
    @param boxes: (B, T, 10) boxes
    @return: float five channel float map (count, sum_x, sum_y, center_x, center_y, objectness_weight_map)
    @return: objectness_weight_map: (B, row, col) float
        1: gird_map count
        2: gird_map_sum_x_y
        3: grid_map_center_x_y
        4: objectness_weight_map
    @return: points_in_which_bbox, [B, M]. record points in which bbox
    @return: count_points_in_bboxes [B, T], count of points in one box
    """
    batch_size = curr_frame_points.shape[0]
    num_points = curr_frame_points.shape[1]
    num_bboxes = boxes.shape[1]
    num_channel = 6
    view_index_map = torch.zeros(size=(batch_size, rows, cols, num_channel), device=device, dtype=torch.float)
    points_in_which_bbox = torch.zeros(size=(batch_size, num_points), device=device, dtype=torch.int32).fill_(-1)
    count_points_in_bboxes = torch.zeros(size=(batch_size, num_bboxes), device=device, dtype=torch.int32)
    pts_in_grid_position = torch.zeros(size=(batch_size, num_points), device=device, dtype=torch.int32).fill_(-1)
    #  GPU algorithm for build view index
    roiaware_pool3d_cuda.build_view_index(curr_frame_points.contiguous(),
                                          rows, cols, num_channel,
                                          boxes, extents.contiguous(),
                                          pts_in_grid_position,
                                          view_index_map,
                                          points_in_which_bbox,
                                          count_points_in_bboxes)
    return view_index_map


def build_pixel_offset_velocity_target(rows, cols, extents, boxes, device):
    """
    :param rows int, how many pixel per row and col
    :param cols int, how many pixel per row and col
    :param device : gpu cuda device
    :param extents: points range in axis x, y, z
    :param boxes: (B, T, 10) boxes
    :param boxes: velocity is the last two element of the boxes
    :return pixel_target_map: (B, rows, cols)
    :return offset_target_map: (B, rows, cols, 2)
    :return size_target_map: (B, rows, cols) cos(2 * yaw), sin(2 * yaw)
    :return yaw_target_map: (B, rows, cols)  center - dz / 2, center + dy / 2
    :return height_target_map: (B, rows, cols)
    :return category_target_map: (B, rows, cols) 10 num of category objects
    :return offset_weight_map: (B, rows, cols)
    :return number pixel in every bboxes: (B, N)
    :record_box_map_idx_map, record pixel belong to which bboxes idx: (B, rows, cols)
    """
    batch_size = boxes.shape[0]
    num_bboxes = boxes.shape[1]  # pay attention to those who`s num_bboxes is zero, it will cause illegal memory
    record_box_map_idx_map = torch.zeros(size=(batch_size, rows, cols), device=device, dtype=torch.int).fill_(-1)
    pixel_target_map = torch.zeros(size=(batch_size, rows, cols), device=device, dtype=torch.float)
    offset_target_map = torch.zeros(size=(batch_size, 2, rows, cols), device=device, dtype=torch.float)
    offset_weight_map = torch.zeros(size=(batch_size, 2, rows, cols), device=device, dtype=torch.float)
    velocity_target_map = torch.zeros(size=(batch_size, 2, rows, cols), device=device, dtype=torch.float)
    size_target_map = torch.zeros(size=(batch_size, 2, rows, cols), device=device, dtype=torch.float)
    yaw_target_map = torch.zeros(size=(batch_size, 2, rows, cols), device=device, dtype=torch.float)
    height_target_map = torch.zeros(size=(batch_size, 2, rows, cols), device=device, dtype=torch.float)
    category_target_map_idx = torch.zeros(size=(batch_size, rows, cols), device=device, dtype=torch.float)
    count_pixels_in_bboxes = torch.zeros(size=(batch_size, num_bboxes), device=device, dtype=torch.int32)
    roiaware_pool3d_cuda.build_pixel_offset_velocity_target(
        rows, cols, extents.contiguous(), boxes, pixel_target_map, offset_target_map, offset_weight_map,
        velocity_target_map, size_target_map, yaw_target_map, height_target_map, category_target_map_idx,
        count_pixels_in_bboxes, record_box_map_idx_map)
    category_target_map = F.one_hot(category_target_map_idx.long(), num_classes=11).float()
    return pixel_target_map, offset_target_map, offset_weight_map, velocity_target_map, size_target_map, \
           yaw_target_map, height_target_map, category_target_map, count_pixels_in_bboxes


if __name__ == '__main__':
    pass