import time
import pickle
import numpy as np
import torch
import os
import os.path as path
from pyquaternion import Quaternion
from functools import reduce
from ops.roiaware_pool3d import roiaware_pool3d_utils
import matplotlib.pyplot as plt
A = np.array([[4.0, 3.0, 1.0, -1.0],
              [5.0, 1.0, 0.0,  0.0],
              [0.0, 6.0, 7.0, 0.0]], np.float)

A = torch.from_numpy(A)

# find where more than 1
pos = torch.where(A > 1.0)
val = A[pos[0], pos[1]]
print("val: ", A[pos[0], pos[1]])
val_sort, val_idx = torch.sort(val, descending=True)
print("idx: ", val_sort, val_idx)
print(pos)


# def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
#                      rotation: Quaternion = Quaternion([1, 0, 0, 0]),
#                      inverse: bool = False) -> np.ndarray:
#     """
#     Convert pose to transformation matrix.
#     :param translation: <np.float32: 3>. Translation in x, y, z.
#     :param rotation: Rotation in quaternions (w ri rj rk).
#     :param inverse: Whether to compute inverse transform matrix.
#     :return: <np.float32: 4, 4>. Transformation matrix.
#     """
#     tm = np.eye(4)
#
#     if inverse:
#         rot_inv = rotation.rotation_matrix.T
#         trans = np.transpose(-np.array(translation))
#         tm[:3, :3] = rot_inv
#         tm[:3, 3] = rot_inv.dot(trans)
#     else:
#         tm[:3, :3] = rotation.rotation_matrix
#         tm[:3, 3] = np.transpose(np.array(translation))
#
#     return tm
#
# def convert_pickle_boxes_to_torch_box(curr_boxes_gt):
#     """
#         args:
#             convert boxes which read from pickle to cuda format
#     """
#     curr_boxes = np.zeros((len(curr_boxes_gt), 9), np.float32)
#     for i, elem in enumerate(curr_boxes_gt):
#         whl = elem.wlh
#         velocity = elem.velocity[:2]
#         if np.isnan(velocity).any():
#             velocity = [0.0, 0.0]
#         curr_boxes[i, :] = ([*elem.center, whl[1], whl[0], whl[2], elem.orientation.yaw_pitch_roll[0],
#                             velocity[0], velocity[1]])
#     return curr_boxes
#
# data_root = "/media/user/C14D581BDA18EBFA/nuScenesGenData"
#
# with open(path.join(data_root, "trainlist.pkl"), "rb") as f:
#     train_list = pickle.load(f)
#
# num_past_lidar = 5
# num_future_lidar = 25
#
# example = train_list[79957]
# value = example.split("_")
# scene_idx = int(value[0])
# sweep_idx = int(value[1])
# ref_lidar_file_name = os.path.join(data_root, "scene_{}/lidars/{}.bin".format(scene_idx, sweep_idx))
# ref_boxes_file_name = os.path.join(data_root, "scene_{}/boxes/{}.pkl".format(scene_idx, sweep_idx))
# ref_calibration_file_name = os.path.join(data_root, "scene_{}/calibration/{}.pkl".format(scene_idx, sweep_idx))
# #  Get refernce pose and timestamp
# with open(ref_calibration_file_name, "rb") as f:
#     ref_pose_rec, ref_cs_rec = pickle.load(f)
# # Homogeneous transform from ego car frame to reference frame
# ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
# # Homogeneous transformation matrix from global to _current_ ego car frame
# car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)
#
# assert path.isfile(ref_lidar_file_name), "{} is not exist".format(ref_lidar_file_name)
# assert path.isfile(ref_boxes_file_name), "{} is not exist".format(ref_boxes_file_name)
# assert path.isfile(ref_calibration_file_name), "{} is not exist".format(ref_calibration_file_name)
#
# #  merge num past lidar sweep for lidar input
# #  first zeros pts
# last_time_stamp = ref_pose_rec['timestamp']
# all_pc = np.fromfile(ref_lidar_file_name, np.float32).reshape((-1, 5))[:, :4].T
# all_pc = np.vstack((all_pc, np.zeros(all_pc.shape[1])))
# for idx in range(1, num_past_lidar):
#     curr_lidar_file_name = os.path.join(data_root, "scene_{}/lidars/{}.bin".format(scene_idx, sweep_idx - idx))
#     curr_calibration_file_name = os.path.join(data_root, "scene_{}/calibration/{}.pkl"
#                                               .format(scene_idx, sweep_idx - idx))
#     print("[INFO] read curr {} lidar name".format(curr_lidar_file_name))
#     curr_pc = np.fromfile(curr_lidar_file_name, np.float32).reshape((-1, 5))[:, :4].T
#
#     # Get past pose
#     with open(curr_calibration_file_name, "rb") as f:
#         current_pose_rec, current_cs_rec = pickle.load(f)
#     global_from_car = transform_matrix(current_pose_rec['translation'],
#                                        Quaternion(current_pose_rec['rotation']), inverse=False)
#     car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
#                                         inverse=False)
#     # Fuse four transformation matrices into one and perform transform.
#     trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
#     curr_pc[:3, :] = trans_matrix.dot(np.vstack((curr_pc[:3, :], np.ones(curr_pc.shape[1]))))[:3, :]
#     curr_time_stamp = current_pose_rec['timestamp']
#     time_diff = 1e-6 * (last_time_stamp - curr_time_stamp)
#     #  hstask timestamp to pc
#     curr_pc = np.vstack([curr_pc, time_diff * np.ones(curr_pc.shape[1])])
#     all_pc = np.hstack((all_pc, curr_pc))
#
# # trans multi lidar sweep to voxel grid
# """
#     voxel: W, H, C, num_of_pts_pre_voxel
#     non_zeros_map:
#     pixel_target:
#     offset_target:
# """
# #  read bot gt
#
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# with open(ref_boxes_file_name, "rb") as f:
#     curr_boxes_gt = pickle.load(f)
# curr_boxes_gt = convert_pickle_boxes_to_torch_box(curr_boxes_gt)
# # torch.cuda.synchronize()
# start = time.time()
# input_pc = torch.from_numpy(np.ascontiguousarray(all_pc.transpose())).unsqueeze(dim=0).cuda().float()
# # torch.cuda.synchronize()
# print("convert cpu pc to gpu has cost about: {} ms".format((time.time() - start) * 1000))
#
# input_gt_boxes = torch.from_numpy(curr_boxes_gt).unsqueeze(dim=0).float().cuda()
# rows = 256
# cols = 256
# extents_cpu = torch.Tensor([[-32., 32.], [-32., 32.], [-3., 2.]]).float()
# extents = extents_cpu.cuda()
# start = time.time()
# pixel_target_map, offset_target_map, velocity_target_map = roiaware_pool3d_utils.\
#     build_pixel_offset_velocity_target(rows, cols, extents, input_gt_boxes, device)
# print("build_pixel_offset_velocity_target has cost about: {} ms".format((time.time() - start) * 1000))
#
# start = time.time()
# voxel_num = 10
# feature_num = 8
# voxel_feature = roiaware_pool3d_utils.build_voxel_feature(input_pc, extents, rows, cols, voxel_num, feature_num, device)
# print("build voxel feature has cost about: {} ms".format((time.time() - start) * 1000))
#
# #  draw pixel and offset
# fig, ax = plt.subplots(2, 3, figsize=(50, 50))
# #  draw points
# ax[0, 0].scatter(all_pc[0, :], all_pc[1, :])
# ax[0, 0].axis('off')
# ax[0, 0].set_aspect('equal')
# ax[0, 0].title.set_text('Lidar data')
#
# #  draw pixel gt
# pixel_gt_numpy = pixel_target_map.cpu().numpy().squeeze()
# pixel_mask = (pixel_gt_numpy > 0.5)
# idx_x = np.arange(rows)
# idx_y = np.arange(cols)
# idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
# X = idx_x[pixel_mask]
# Y = idx_y[pixel_mask]
# ax[0, 1].plot(X, Y, '.')
# ax[0, 1].set_aspect('equal')
# ax[0, 1].axis('off')
# ax[0, 1].title.set_text('pixel gt')
#
# #  draw offset gt
# pos_nonzeros_tuple = np.where(pixel_mask == True)
# #  too mush positive pixel need to show, we sample it
# pox_x, pos_y = pos_nonzeros_tuple[0], pos_nonzeros_tuple[1]
# pos_selected_mask = np.zeros_like((pixel_gt_numpy > 0.5)).astype(np.bool)
# pos_selected_mask[pox_x, pos_y] = True
# offset_target_map_numpy = offset_target_map.cpu().numpy().squeeze()
# X = idx_x[pos_selected_mask]
# Y = idx_y[pos_selected_mask]
# U = -offset_target_map_numpy[0, :, :][pos_selected_mask]
# V = -offset_target_map_numpy[1, :, :][pos_selected_mask]
# ax[0, 2].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r')
# ax[0, 2].set_aspect('equal')
# ax[0, 2].title.set_text('offset Prediction')
# ax[0, 2].axis('off')
#
# # draw velocity
# #  draw offset gt
# grid_size = (extents_cpu[0, 1] - extents_cpu[0, 0]) / rows
# pos_nonzeros_tuple = np.where(pixel_mask == True)
# #  too mush positive pixel need to show, we sample it
# pox_x, pos_y = pos_nonzeros_tuple[0], pos_nonzeros_tuple[1]
# pos_selected_mask = np.zeros_like((pixel_gt_numpy > 0.5)).astype(np.bool)
# # pos_selected_mask[pox_x[::10], pos_y[::10]] = True
# pos_selected_mask[pox_x, pos_y] = True
# velocity_target_map_numpy = velocity_target_map.cpu().numpy().squeeze()
# X = idx_x[pos_selected_mask]
# Y = idx_y[pos_selected_mask]
# U = velocity_target_map_numpy[0, :, :][pos_selected_mask] / grid_size.numpy()
# V = velocity_target_map_numpy[1, :, :][pos_selected_mask] / grid_size.numpy()
#
# ax[1, 0].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r')
# ax[1, 0].set_aspect('equal')
# ax[1, 0].title.set_text('velocity Prediction')
# ax[1, 0].axis('off')
#
# #  draw features
# ax[1, 1].imshow(voxel_feature[0, :, :, 53].cpu().numpy())
# ax[1, 1].set_aspect('equal')
# ax[1, 1].title.set_text('build voxel feature')
# ax[1, 1].axis('off')
# plt.show()
# # vistool=vis(all_pc)
# # vispy.app.run()
# print("[INFO] code finished")
# debug = 1
# plt.show()

# start_t = time.time()
# tensor_cpu = torch.zeros([300000, 5], dtype=torch.float32)
# print("from here 0 cost : {} ms".format((time.time() - start_t) * 1000))
# tensor_cpu = tensor_cpu.cuda()
# print("from here 1 cost : {} ms".format((time.time() - start_t) * 1000))
# print("unitTest finished.")

# # torch.cuda.synchronize()
# start_t = time.time()
# # for idx in range(10):
# tensor_cpu1 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu1.cuda()
# tensor_cpu2 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu2.cuda()
# tensor_cpu3 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu3.cuda()
# tensor_cpu4 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu4.cuda()
# tensor_cpu5 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu5.cuda()
# tensor_cpu6 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu6.cuda()
# tensor_cpu7 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu7.cuda()
# tensor_cpu8 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu8.cuda()
# tensor_cpu9 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu9.cuda()
# tensor_cpu10 = torch.zeros([300000, 5], dtype=torch.float32)
# tensor_cpu10.cuda()
# print("from here 1 cost : {} ms".format((time.time() - start_t) * 1000))

# # torch.cuda.synchronize()
# time.sleep(5)
print("start code process")

