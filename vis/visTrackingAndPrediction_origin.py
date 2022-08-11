"""Copyright (c) 2021, by sun jing
brief: vis tracking result and prediction result by vispy"""

import torch
import torch.nn as nn
import itertools
import configs
import os
from model import OffsetThree
from data.nuscenes_dataloader import TrainDatasetMultiSeq
from utils.tracker import TrackingModel
from utils.common_utils import RANDOR_COLORS

import numpy as np
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion


import vispy
from vispy import scene
from vispy.color import get_colormaps
from vispy import app
from vispy.io import load_data_file, read_png
from vispy import app, visuals
from vispy.scene import visuals, SceneCanvas
from vispy.scene.visuals import Text
from vispy.visuals import TextVisual
import pickle
from data_loader import transform_matrix
from functools import reduce

resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/det_da_6epoch_0724/epoch_6.pth"
#resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/det_da_6epoch_0724/epoch_6.pth"
resume_tracking = "/media/user/C14D581BDA18EBFA/code/OffsetThree/checkpoint/tracking_ass_new_motion/epoch_7.pth"

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_nuscenes_base = TrainDatasetMultiSeq(batch_size=1, devices=devices, tracking=True, split='val', vispy_seq=True)
data_nuscenes = TrainDatasetMultiSeq(batch_size=1, devices=devices, tracking=True, split='val', vispy_seq=True)
# trainloader = torch.utils.data.DataLoader(data_nuscenes,
#                                           batch_size=configs.data.batch_size,
#                                           shuffle=False,
#                                           num_workers=configs.data.num_worker,
#                                           collate_fn=data_nuscenes.collate_batch)

model_detect = OffsetThree(num_feature_channel=configs.bird.num_feature_channel, batch_size=2, device=devices, is_training=False)
model_detect = nn.DataParallel(model_detect)
model_detect = model_detect.to(devices)
checkpoint = torch.load(resume_det)
model_detect.load_state_dict(checkpoint['model_state_dict'])


model_tracking = TrackingModel()
model_tracking = model_tracking.to(devices)
checkpoint_tracking = torch.load(resume_tracking)
model_tracking.load_state_dict(checkpoint_tracking['model_state_dict'])

num_path_lidar = configs.tracker.num_past_lidar
num_feature_lidar = configs.tracker.num_feature_lidar
num_interval_lidar = num_path_lidar + num_feature_lidar

np.set_printoptions(suppress=True)

data_root = "/media/user/C14D581BDA18EBFA/nuScenesGenData"


class VisTrackingPrediction(object):
    def __init__(self, offset=0):
        self.seq_idx = 0
        self.offset = offset
        self.reset()
        self.last_gt_boxes = None
        self.last_ego_motion = None
        self.to_next_sample()

    def reset(self):
        self.action = "no"  #, no , next, back, quit
        self.canvas = SceneCanvas(keys='interactive', size=(1600, 1200), show=True, bgcolor='k')
        self.canvas.events.key_press.connect(self.key_press_event)

        # interface(n next, b back, q quit, very simple)
        self.lidar_view = self.canvas.central_widget.add_view()
        self.lidar_view.camera = 'turntable'
        visuals.XYZAxis()
        self.lidar_vis = visuals.Markers()
        self.lidar_view.add(self.lidar_vis)
        self.extents = np.array(configs.bird.extents)
        self.rows = configs.bird.rows
        self.cols = configs.bird.cols
        self.coor = np.array([self.extents[0, 0], self.extents[1, 0]])
        self.resolution = configs.bird.resolution
        self.last_scene_idx = 0
        self.last_sweep_idx = 0
        self.curr_num_scenes_sweep = -1
        # draw lidar boxes
        self.line_vis = visuals.Line(color='r', method='gl', connect="segments", name="boxes line", width=3)
        self.velocity_vis = visuals.Line(color='r', method='gl', connect="segments", name="velocity line", width=3)
        self.box_info_text = scene.Text(color='w', parent=self.velocity_vis, font_size=540)
        self.lidar_view.add(self.line_vis)
        self.lidar_view.add(self.velocity_vis)

    def get_trans_matrix2(self, data_dict):
        """
        brief: get transform matrix
        @return:
        """
        scene_idx = data_dict['scene_idx']
        sweep_idx = data_dict['sweep_idx']
        ref_lidar_file_name = os.path.join(data_root, "scene_{}/lidars/{}.bin".format(scene_idx, sweep_idx))
        ref_boxes_file_name = os.path.join(data_root, "scene_{}/boxes/{}.pkl".format(scene_idx, sweep_idx))
        ref_calibration_file_name = os.path.join(data_root, "scene_{}/calibration/{}.pkl".format(scene_idx, sweep_idx))
        with open(ref_calibration_file_name, "rb") as f:
            ref_pose_rec, ref_cs_rec = pickle.load(f)
        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # for last sweep prep
        curr_calibration_file_name = os.path.join(data_root, "scene_{}/calibration/{}.pkl"
                                                  .format(self.last_scene_idx, self.last_sweep_idx))
        with open(curr_calibration_file_name, "rb") as f:
            current_pose_rec, current_cs_rec = pickle.load(f)
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)
        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        return trans_matrix

    def get_trans_matrix(self, data_dict, lastSweepBoxes):
        """
            brief: get transform matrix
            bboxes: last frame bboxes
            @return:
        """
        res = []
        scene_idx = data_dict['scene_idx']
        sweep_idx = data_dict['sweep_idx']
        ref_lidar_file_name = os.path.join(data_root, "scene_{}/lidars/{}.bin".format(scene_idx, sweep_idx))
        ref_boxes_file_name = os.path.join(data_root, "scene_{}/boxes/{}.pkl".format(scene_idx, sweep_idx))
        ref_calibration_file_name = os.path.join(data_root, "scene_{}/calibration/{}.pkl".format(scene_idx, sweep_idx))
        with open(ref_calibration_file_name, "rb") as f:
            ref_pose_rec, ref_cs_rec = pickle.load(f)
        # Homogeneous transform from ego car frame to reference frame
        # for last sweep prep
        curr_calibration_file_name = os.path.join(data_root, "scene_{}/calibration/{}.pkl"
                                                  .format(self.last_scene_idx, self.last_sweep_idx))
        with open(curr_calibration_file_name, "rb") as f:
            last_ref_pose_rec, last_ref_cs_rec = pickle.load(f)
        for lastSweepBox in lastSweepBoxes:
            # move coord to global
            lastSweepBox.rotate(Quaternion(last_ref_cs_rec['rotation']))
            lastSweepBox.translate(np.array(last_ref_cs_rec['translation']))
            lastSweepBox.rotate(Quaternion(last_ref_pose_rec['rotation']))
            lastSweepBox.translate(np.array(last_ref_pose_rec['translation']))

            # Move box to ego vehicle coord system
            lastSweepBox.translate(-np.array(ref_pose_rec['translation']))
            lastSweepBox.rotate(Quaternion(ref_pose_rec['rotation']).inverse)
            # Move box to sensor coord system
            lastSweepBox.translate(-np.array(ref_cs_rec['translation']))
            lastSweepBox.rotate(Quaternion(ref_cs_rec['rotation']).inverse)
            res.append(lastSweepBox)
        return res

    def to_next_sample(self):
        print("[to_netxt_sample] [" + str(self.offset) + "] to_next_sample")
        data_dict_add_base = data_nuscenes_base[0]
        data_dict = data_nuscenes[self.offset]
        batch_list_data = [data_dict, data_dict_add_base]
        data_dict_input = TrainDatasetMultiSeq.collate_batch(batch_list_data)
        # if data_dict['scene_idx'] != self.last_scene_idx:
        #     self.last_scene_idx = data_dict['scene_idx']
        #     self.curr_num_scenes_sweep = len([d for d in os.listdir(os.path.join(
        #         data_root, "scene_" + str(self.last_scene_idx) + "/lidars"))]) - num_path_lidar
        with torch.no_grad():
            det_res_dict, data_dict = model_detect(data_dict_input)  # get detection results
            track_result = model_tracking(data_dict, det_res_dict, is_training=False)
        self.seq_idx += 1
        print("[INFO] {}: {}".format(self.seq_idx, self.curr_num_scenes_sweep))
        # if self.seq_idx >= self.curr_num_scenes_sweep:
        #     self.seq_idx = 0
        if data_dict['scene_idx'][0] != self.last_scene_idx:
            model_tracking.reset_tracker()
            self.last_scene_idx = data_dict['scene_idx']
        pts = data_dict['pc'][0].detach().cpu().numpy()
        extents = self.extents
        # prep confidence pred
        filter_idx = np.where((extents[0, 0] < pts[:, 0]) & (pts[:, 0] < extents[0, 1]) &
                              (extents[1, 0] < pts[:, 1]) & (pts[:, 1] < extents[1, 1]) &
                              (extents[2, 0] < pts[:, 2]) & (pts[:, 2] < extents[2, 1]))[0]
        pts = pts[filter_idx]
        self.lidar_vis.set_gl_state('translucent', depth_test=False)
        self.lidar_vis.set_data(pts[:, :3],
                                edge_color=None, size=2)
        self.lidar_view.add(self.lidar_vis)
        self.lidar_view.camera = 'turntable'

        #  draw box
        boxes = self.get_tracking_box(track_result)
        # boxes = self.get_detecting_box(boxes)

        # finished add last gt box in curr frame
        num_pts_pre_boxes = 24
        all_bboxes_pts = np.zeros((len(boxes) * num_pts_pre_boxes, 3), dtype=np.float32)
        color_lines = np.zeros((len(boxes) * num_pts_pre_boxes, 3), dtype=np.float32)
        box_idx = 0
        velocity_pts = []
        display_content = []
        display_position = []
        for box in boxes:
            box_pt = box.corners().transpose()
            all_bboxes_pts[box_idx * num_pts_pre_boxes: (box_idx + 1) * num_pts_pre_boxes, :] =\
                box_pt[[0, 1, 4, 5, 7, 6, 3, 2, 0, 3, 3, 7, 7, 4, 4, 0, 2, 6, 6, 5, 5, 1, 1, 2], :]
            curr_dip_text = f"{box.id}_{box.age}"
            curr_position = (tuple(box.center))
            if not np.isnan(box.velocity).any():
                # show velocity
                center = box.center
                target = box.center + box.velocity
                velocity_pts.append(center)
                velocity_pts.append(target)
                # curr_dip_text += (": " + str(box.velocity))
                # all_bboxes_pts = np.vstack([all_bboxes_pts, center, target])
            color_lines[box_idx * num_pts_pre_boxes: (box_idx + 1) * num_pts_pre_boxes, :] = \
                RANDOR_COLORS[box.id * 5 % RANDOR_COLORS.shape[0]] / 255.0
            box_idx += 1
            display_content.append(curr_dip_text)
            display_position.append(curr_position)
        if boxes.__len__() != 0:
            self.box_info_text.text = display_content
            self.box_info_text.pos = display_position
            self.line_vis.set_data(all_bboxes_pts, color=color_lines)

        #  add ego motion data
        self.last_gt_boxes = data_dict['gt_boxes']
        curr_ego_motion = data_dict['ego_motion']
        self.last_scene_idx = data_dict['scene_idx']
        self.last_sweep_idx = data_dict['sweep_idx']
        # curr_ego_motion[0] = np.sqrt(curr_ego_motion[0] * curr_ego_motion[0] +
        #                              curr_ego_motion[1] * curr_ego_motion[1])
        # curr_ego_motion[1] = 0
        # self.last_ego_motion = curr_ego_motion

        if velocity_pts.__len__() > 0:
            self.velocity_vis.set_data(np.array(velocity_pts))
        else:
            self.velocity_vis.set_data(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    @staticmethod
    def get_tracking_box(track_result):
        curr_track_box = track_result['track_box']
        curr_track_ids = track_result['track_ids']
        track_boxes_age = track_result['track_boxes_age']
        boxes = []
        for idx in range(curr_track_ids.__len__()):
            curr_id = curr_track_ids[idx]
            curr_box = curr_track_box[idx]
            inst = curr_box.detach().cpu().numpy()
            if np.isnan(inst).any():
                continue
            size_box = [inst[4], inst[3], inst[5]]
            box = Box(center=inst[0:3], size=size_box, orientation=Quaternion(
                axis=[0, 0, 1], angle=inst[6]))
            box.id = curr_id
            box.velocity = np.array([inst[7], inst[8], 0.0])
            box.age = track_boxes_age[idx]
            boxes.append(box)
        return boxes

    @staticmethod
    def get_detecting_box(boxes_np):
        """
        @brief For debug dataAugmentation
        @param boxes_np:
        @return:
        """
        boxes = []
        for idx in range(len(boxes_np)):
            elem = boxes_np[idx]
            size_box = elem[4], elem[3], elem[5]
            box = Box(center=elem[:3], size=size_box, orientation=Quaternion(
                axis=[0, 0, 1], angle=elem[6]))
            box.id = 0
            box.velocity = np.array([elem[7], elem[8], 0.0])
            box.age = 0
            boxes.append(box)
        return boxes

    def update_scenes_choose(self):
        self.to_next_sample()

    def draw_boxes(self, boxes):
        num_pts_pre_boxes = 24
        all_bboxes_pts = np.zeros((len(boxes) * num_pts_pre_boxes, 3), dtype=np.float32)
        box_idx = 0
        color_lines = np.zeros((len(boxes) * num_pts_pre_boxes, 3), dtype=np.float32)
        # velocity_pts = []
        for box in boxes:
            box_pt = box.corners().transpose()
            all_bboxes_pts[box_idx * num_pts_pre_boxes: (box_idx + 1) * num_pts_pre_boxes, :] =\
                box_pt[[0, 1, 4, 5, 7, 6, 3, 2, 0, 3, 3, 7, 7, 4, 4, 0, 2, 6, 6, 5, 5, 1, 1, 2], :]
            if not np.isnan(box.velocity).any():
                # show velocity
                center = box.center
                target = box.center + box.velocity
                # velocity_pts.append(center)
                # velocity_pts.append(target)
            color_lines[box_idx * num_pts_pre_boxes: (box_idx + 1) * num_pts_pre_boxes, :] = \
                RANDOR_COLORS[box.id % RANDOR_COLORS.shape[0]] / 255.0
            box_idx += 1
        self.line_vis.set_data(all_bboxes_pts, color=color_lines)
        # if velocity_pts.__len__() > 0:
        #     self.velocity_vis.set_data(np.array(velocity_pts))
        # else:
        #     self.velocity_vis.set_data(np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]))

    def key_press_event(self, event):
        if event.key == 'N' or event.key == 'Right':
            self.offset += 1
            # self.update_canvas()
            self.to_next_sample()
        # elif event.key == "B":
        #     self.offset -= 1
        #     print("[EVENT] B")
        #     # self.update_canvas()
        #     self.to_next_sample()
        elif event.key == "H":
            self.scenes_idx += 1
            if self.scenes_idx >= 849:
                self.scenes_idx = 849
            self.offset = 0
            print("Processing scene {} ...".format(self.scenes_idx))
            self.lastSweepBoxes.clear()
            self.update_scenes_choose()
        elif event.key == "G":
            self.scenes_idx -= 1
            if self.scenes_idx <= 0:
                self.scenes_idx = 0
            self.offset = 0
            print("Processing scene {} ...".format(self.scenes_idx))
            self.lastSweepBoxes.clear()
            self.update_scenes_choose()


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    visTool = VisTrackingPrediction(offset=6000)
    vispy.app.run()
