import os
import time

import torch
import configs
import pickle
import operator
import warnings
import numpy as np
import os.path as path
from functools import reduce
from collections import defaultdict
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from pathlib import Path
import json
import itertools
from ops.roiaware_pool3d import roiaware_pool3d_utils
from utils.data_utils import transform_matrix, convert_pickle_boxes_to_torch_box
import torch.multiprocessing
from data.datasets.nuscenes.nusc_common import *
from data.nuscenes_base import train_folder_list, val_folder_list

cfg = configs

try:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.eval.detection.config import config_factory
except:
    print("[ERROR] nuScenes devkit not found! ")

from data.datasets.nuscenes.nusc_common import (
    general_to_detection,
    cls_attr_dist,
    _second_det_to_nusc_box,
    _lidar_nusc_box_to_global,
    eval_main
)

class TrainDatasetMultiSeq(Dataset):
    def __init__(self, batch_size=1, devices=None, tracking=False, split='train', vispy_seq=False):
        """
        cache_size: The cache size for storing parts of data in the memory (for reducing the IO cost)
        """
        self.split = split
        if cfg.data.data_root is None:
            raise ValueError("The dataset root is None. Should specify its value.")

        self.data_root = cfg.data.data_root
        self.vispy_seq = vispy_seq
        print("data root:", self.data_root)
        #  first synchronize cuda prep
        #  # torch.cuda.synchronize()
        self.batch_size = batch_size
        self.num_past_lidar = cfg.data.num_past_lidar
        self.num_future_lidar = cfg.data.num_future_lidar
        if tracking:
            self.train_list = self.get_tracking_train_list()
        else:
            if self.split == "train":
                # with open(path.join(self.data_root, "trainlist.pkl"), "rb") as f:
                #     self.train_list = pickle.load(f)
                #     self.train_list = pickle.load(f)
                with open(path.join(self.data_root, "val_list.pkl"), "rb") as f:
                    self.train_list = pickle.load(f)
            else:
                with open(path.join(self.data_root, "val_list.pkl"), "rb") as f:
                    # self.train_list = pickle.load(f)[4591:]
                    self.train_list = pickle.load(f)
                    # self.train_list = self.train_list
        # self.train_list = self.train_list[1200:1400]  # over-fitting
        # self.train_list = self.train_list  # over-fitting
        self.num_sweeps_seqs = len(self.train_list)
        self.extents = torch.Tensor(np.array(cfg.bird.extents).astype(np.float)).float().cuda()
        self.rows = cfg.bird.rows
        self.cols = cfg.bird.cols
        self.voxel_num = cfg.bird.voxel_num
        self.feature_num = cfg.bird.feature_num
        self.devices = devices
        self.TRACKING_NAMES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']
        if self.devices is None:
            warnings.warn(">> you devices is None, please be careful, ok?")
        if tracking:
            if self.split == "train":
                if self.num_sweeps_seqs != 181920:
                    warnings.warn(">> The size of training dataset is not 178858.\n")
            elif self.split == "val":
                if self.num_sweeps_seqs != 6019:
                    warnings.warn(">> The size of training dataset is not 6019.\n")
            else:
                warnings.warn(">> The size of training dataset is testset.\n")
        else:
            if self.split == "train":
                if self.num_sweeps_seqs != 181920:
                    warnings.warn(">> The size of training dataset is not 178858.\n")
            elif self.split == "val":
                if self.num_sweeps_seqs != 6019:
                    warnings.warn(">> The size of training dataset is not 6019.\n")
            else:
                warnings.warn(">> The size of training dataset is testset.\n")

        # for eval and test
        self.version = "v1.0-trainval"
        tasks = [
            dict(num_class=1, class_names=["car"]),
            dict(num_class=2, class_names=["truck", "construction_vehicle"]),
            dict(num_class=2, class_names=["bus", "trailer"]),
            dict(num_class=1, class_names=["barrier"]),
            dict(num_class=2, class_names=["motorcycle", "bicycle"]),
            dict(num_class=2, class_names=["pedestrian", "traffic_cone"]),
        ]

        self._class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))
        self._name_mapping = general_to_detection
        self.eval_version = "detection_cvpr_2019"
        self._info_path = configs.data.info_path


    def __len__(self):
        return self.num_sweeps_seqs

    def __get_value(self, scene_idx, sweep_idx, is_key_frame):
        """
            scene_idx
            sweep_idx
            is_key_frame
        """
        ref_lidar_file_name = os.path.join(self.data_root, "scene_{}/lidars/{}.bin".format(scene_idx, sweep_idx))
        ref_boxes_file_name = os.path.join(self.data_root, "scene_{}/boxes/{}.pkl".format(scene_idx, sweep_idx))
        ref_calibration_file_name = os.path.join(self.data_root,
                                                 "scene_{}/calibration/{}.pkl".format(scene_idx, sweep_idx))
        ego_motion_file_name = os.path.join(self.data_root, "scene_{}/ego_motion/{}.pkl".format(scene_idx, sweep_idx))
        #  Get references pose and timestamp
        with open(ref_calibration_file_name, "rb") as f:
            ref_pose_rec, ref_cs_rec = pickle.load(f)
        sweep_token_file_name = os.path.join(self.data_root, "scene_{}/token/{}.txt".format(scene_idx, sweep_idx))
        with open(sweep_token_file_name, "r") as f:
            data_token = f.readline()
        # with open(ego_motion_file_name, "rb") as f:
        #     ego_motion = pickle.load(f)
        # Homogeneous transform from ego car frame to reference frame
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
        # Homogeneous transformation matrix from global to _current_ ego car frame
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # assert path.isfile(ref_lidar_file_name), "{} is not exist".format(ref_lidar_file_name)
        # assert path.isfile(ref_boxes_file_name), "{} is not exist".format(ref_boxes_file_name)
        # assert path.isfile(ref_calibration_file_name), "{} is not exist".format(ref_calibration_file_name)
        #  merge num past lidar sweep for lidar input, first zeros pts
        # torch.cuda.synchronize()
        time_trans = time.time()
        last_time_stamp = ref_pose_rec['timestamp']
        all_pc = np.fromfile(ref_lidar_file_name, np.float32).reshape((-1, 5))[:, :4].T
        all_pc = np.vstack((all_pc, np.zeros(all_pc.shape[1])))
        curr_frame_pc = all_pc.copy()
        curr_ego_trans_matrix = None
        num_past_lidar = 0
        if sweep_idx >= self.num_past_lidar:
            num_past_lidar = self.num_past_lidar
        else:
            num_past_lidar = sweep_idx
        for idx in range(0, num_past_lidar):
            sweep_multi_idx = sweep_idx - 1 - idx
            # print("[DEBUG] {}, {}, {}".format(idx, sweep_idx, sweep_multi_idx))
            curr_lidar_file_name = os.path.join(self.data_root, "scene_{}/lidars/{}.bin"
                                                .format(scene_idx, sweep_multi_idx))
            curr_calibration_file_name = os.path.join(self.data_root, "scene_{}/calibration/{}.pkl"
                                                      .format(scene_idx, sweep_multi_idx))
            time_read_lidar = time.time()
            curr_pc = np.fromfile(curr_lidar_file_name, np.float32).reshape((-1, 5))[:, :4].T
            # print('[TIME] read lidar has cost: {}'.format((time.time() - time_read_lidar) * 1000))
            # Get past pose
            with open(curr_calibration_file_name, "rb") as f:
                current_pose_rec, current_cs_rec = pickle.load(f)
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)
            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
            lidar_dot = time.time()
            curr_pc[:3, :] = trans_matrix.dot(np.vstack((curr_pc[:3, :], np.ones(curr_pc.shape[1]))))[:3, :]
            # print('[TIME] lidar dot has cost: {}'.format((time.time() - lidar_dot) * 1000))
            curr_time_stamp = current_pose_rec['timestamp']
            time_diff = 1e-6 * (last_time_stamp - curr_time_stamp)
            #  hstask timestamp to pc
            curr_pc = np.vstack([curr_pc, time_diff * np.ones(curr_pc.shape[1])])
            # lidar_stack = time.time()
            all_pc = np.hstack((all_pc, curr_pc))
            # print('[TIME] lidar stack has cost: {}'.format((time.time() - lidar_stack) * 1000))
            if idx == 0:
                curr_ego_trans_matrix = trans_matrix
        # torch.cuda.synchronize()
        # print('[TIME] trans cloud has cost: {}'.format((time.time() - time_trans) * 1000))
        with open(ref_boxes_file_name, "rb") as f:
            curr_boxes_gt_orig = pickle.load(f)
        curr_boxes_gt = convert_pickle_boxes_to_torch_box(curr_boxes_gt_orig)
        # input_pc = torch.from_numpy(np.ascontiguousarray(all_pc.transpose())).unsqueeze(dim=0).cuda().float()
        # curr_frame_pc = torch.from_numpy(np.ascontiguousarray(curr_frame_pc.transpose())).\
        #    unsqueeze(dim=0).cuda().float()
        # input_gt_boxes_tensor = torch.from_numpy(curr_boxes_gt).unsqueeze(dim=0).float()
        # input_gt_boxes = input_gt_boxes_tensor.cuda().contiguous()
        ret = {
            'points': all_pc.transpose(),
            'pc': curr_frame_pc.transpose(),
            'gt_boxes': curr_boxes_gt,
            'scene_idx': scene_idx,
            'sweep_idx': sweep_idx,
            'is_key_frame': is_key_frame,
            'ego_motion': curr_ego_trans_matrix,
            'gt_boxes_orig': curr_boxes_gt_orig,
            'token': data_token,
        }
        return ret

    def __getitem__(self, idx):
        example = self.train_list[idx]
        value = example.split("_")
        scene_idx = int(value[0])
        sweep_idx = int(value[1])
        is_key_frame = int(value[2])
        # torch.cuda.synchronize()
        time_0 = time.time()
        data_dict = self.__get_value(scene_idx, sweep_idx, is_key_frame)
        # torch.cuda.synchronize()
        # print('[TIME] get_value: {}'.format((time.time() - time_0) * 1000))
        return data_dict

    def get_tracking_train_list(self):
        """
            online get tracking training list
        @return:
        """
        # from data.create_data_with_ego_motion import get_sweep_keyframe_train_val_test_attr
        # from data import splits
        # splits_dict = splits.create_splits_scenes()
        data_root = configs.data.data_root
        seq_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if
                    os.path.isdir(os.path.join(data_root, d))]
        num_past_lidar = self.num_past_lidar
        num_feature_lidar = self.num_future_lidar
        num_interval_lidar = num_past_lidar + num_feature_lidar
        train_list = []
        assert len(seq_dirs) == 850, "number scenes has not 850"
        # train_idx = []
        # val_idx = []
        for dir_idx in range(seq_dirs.__len__()):
            curr_scene_file_names = [d for d in os.listdir(os.path.join(data_root, "scene_"
                                                                        + str(dir_idx) + "/lidars"))]
            num_curr_scens_sweep = len(curr_scene_file_names)
            first_curr_scenes_name = curr_sweep_token_filename = os.path.join(data_root, "scene_" + str(dir_idx)
                                                                              + "/token/{}.txt".format(0))
            # data_split_name, _, _, _ = get_sweep_keyframe_train_val_test_attr(first_curr_scenes_name, splits_dict)
            # if data_split_name == "train" and self.split == "train":
            #     curr_scene_file_names = [d for d in
            #                              os.listdir(os.path.join(data_root, "scene_" + str(dir_idx) + "/lidars"))]
            #     num_curr_scens_sweep = len(curr_scene_file_names)
            #     for sweep_idx in range(num_past_lidar, num_curr_scens_sweep - num_interval_lidar, num_interval_lidar):
            #         for idx in range(num_interval_lidar):
            #             train_list.append("{}_{}_{}".format(dir_idx, sweep_idx + idx, 0))
            # print("[INFO]:{}: {}".format(dir_idx, data_split_name))
            # if data_split_name == "train":
            #     train_idx.append(dir_idx)
            # else:
            #     val_idx.append(dir_idx)
            if dir_idx in train_folder_list:
                data_split_name = "train"
            else:
                data_split_name = "val"
            if data_split_name == "val" and self.split == "train":
                curr_scene_file_names = [d for d in
                                         os.listdir(os.path.join(data_root, "scene_" + str(dir_idx) + "/lidars"))]
                num_curr_scens_sweep = len(curr_scene_file_names)
                for sweep_idx in range(num_past_lidar, num_curr_scens_sweep - num_interval_lidar, num_interval_lidar):
                    for idx in range(num_interval_lidar):
                        train_list.append("{}_{}_{}".format(dir_idx, sweep_idx + idx, 0))
                    # print("scense {} has {} file".format(scene_idx, num_curr_scens_sweep))
            elif data_split_name == "val" and self.split == "val":
                curr_scene_file_names = [d for d in
                                         os.listdir(os.path.join(data_root, "scene_" + str(dir_idx) + "/lidars"))]
                num_curr_scens_sweep = len(curr_scene_file_names)
                for sweep_idx in range(num_past_lidar, num_curr_scens_sweep - num_interval_lidar, num_interval_lidar):
                    for idx in range(num_interval_lidar):
                        train_list.append("{}_{}_{}".format(dir_idx, sweep_idx + idx, 0))
        # print(train_idx)
        # print('---------------------------')
        # print(val_idx)
        train_list_np = np.array(train_list)
        if not self.vispy_seq:
            train_list_np_reshape = train_list_np.reshape(-1, num_interval_lidar)
            np.random.shuffle(train_list_np_reshape)
        else:
            train_list_np_reshape = train_list_np.reshape(-1, 1)
        tracking_train_list = []
        # prep tracking train list for num of per batch size by bai xu yang
        for seq_idx in range(0, train_list_np_reshape.shape[0] - self.batch_size, self.batch_size):
            for sub_seq_idx in range(train_list_np_reshape.shape[1]):
                for idx in range(self.batch_size):
                    tracking_train_list.append(train_list_np_reshape[seq_idx + idx, sub_seq_idx])
        return tracking_train_list

    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)
        for cur_sample in batch_list:
            for key, val in cur_sample.items():
                data_dict[key].append(val)
        batch_size = len(batch_list)
        ret = {}
        for key, val in data_dict.items():
            try:
                if key in ['pixel_target_map', 'offset_target_map', 'velocity_target_map',
                           'voxel_feature', 'voxel_count_map']:
                    ret[key] = torch.cat(val, dim=0)
                elif key in ['points', 'pc']:
                    # coors = []
                    # for idx, coor in enumerate(val):
                    #     num_pts = coor.shape[0]
                    #     pad_val = idx * torch.ones(size=(num_pts, 1), device=coor.device, dtype=torch.float)
                    #     cood_pad = torch.cat((coor, pad_val), dim=1)
                    #     coors.append(cood_pad)
                    # ret[key] = torch.cat(coors, dim=0)
                    ret[key] = val
                # elif key in ['gt_boxes']:
                #     max_gt = max([x.shape[1] for x in val])
                #     batch_gt_boxes3d = torch.zeros(size=(batch_size, max_gt, val[0].shape[-1]),
                #                                    device=val[0].device, dtype=val[0].dtype)
                #     for k in range(batch_size):  # val[x] shape is [1, NBoxes, dim]
                #         batch_gt_boxes3d[k, :val[k].shape[1], :] = val[k]
                #     ret[key] = batch_gt_boxes3d
                elif key in ['scene_idx', 'sweep_idx', 'ego_motion', 'gt_boxes_orig', 'token', 'is_key_frame', 'gt_boxes']:
                    ret[key] = val
                else:
                    ret[key] = torch.cat(val, dim=0)
            except:
                print('Error in collate_batch: key={}'.format(key))
                raise TypeError
        ret['batch_size'] = batch_size
        return ret

    def evaluation(self, detections, output_dir=None, gen_root_dir=None, testset=False):
        version = self.version
        eval_set_map = {
            "v1.0-mini": "mini_val",
            "v1.0-trainval": "val",
            "v1.0-test": "test",
        }

        print("[INFO] Start read nuscenes ...")
        with open(os.path.join(gen_root_dir, "nusc.pkl"), "rb") as readNusc:
            nusc = pickle.load(readNusc)

        if version == "v1.0-trainval":
            train_scenes = splits.train
            # random.shuffle(train_scenes)
            # train_scenes = train_scenes[:int(len(train_scenes)*0.2)]
            val_scenes = splits.val
        elif version == "v1.0-test":
            train_scenes = splits.test
            val_scenes = []
        elif version == "v1.0-mini":
            train_scenes = splits.mini_train
            val_scenes = splits.mini_val
        else:
            raise ValueError("unknown")

        test = "test" in version

        available_scenes = get_available_scenes(nusc)
        available_scene_names = [s["name"] for s in available_scenes]
        train_scenes = list(filter(lambda x: x in available_scene_names, train_scenes))
        val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
        train_scenes = set(
            [
                available_scenes[available_scene_names.index(s)]["token"]
                for s in train_scenes
            ]
        )
        val_scenes = set(
            [available_scenes[available_scene_names.index(s)]["token"] for s in val_scenes]
        )

        if test:
            print(f"[INFO] test scene: {len(train_scenes)}")
        else:
            print(f"[INFO] train scene: {len(train_scenes)}, val scene: {len(val_scenes)}")

        if not testset:
            val_sample_data_tokens = []
            val_sample_tokens = []
            for sample in nusc.sample:
                if sample['scene_token'] in val_scenes:
                    curr_sample_data_token = nusc.get('sample_data', sample['data']['LIDAR_TOP'])['token']
                    val_sample_data_tokens.append(curr_sample_data_token)
                    val_sample_tokens.append(sample['token'])
                    # val_tokens.append(sample['token'])
            dets = []
            miss = 0
            for idx, val_gt_token in enumerate(val_sample_data_tokens):
                if val_gt_token in detections:
                    detections[val_gt_token].update({'sample_token': val_sample_tokens[idx]})
                    dets.append(detections[val_gt_token])
                else:
                    miss += 1
            # assert miss == 0
            print("[WARN] miss {}:".format(miss))
        else:
            dets = [v for _, v in detections.items()]
            assert len(detections) == 6008

        nusc_annos = {
            "results": {},
            "meta": None,
        }

        mapped_class_names = []
        for n in self._class_names:
            if n in self._name_mapping:
                mapped_class_names.append(self._name_mapping[n])
            else:
                mapped_class_names.append(n)

        for det in dets:
            annos = []
            boxes = _second_det_to_nusc_box(det)
            boxes = _lidar_nusc_box_to_global(nusc, boxes, det["sample_token"])
            # boxes = _lidar_nusc_box_to_global(nusc, boxes, det["token"])
            for i, box in enumerate(boxes):
                name = mapped_class_names[box.label]
                if np.sqrt(box.velocity[0] ** 2 + box.velocity[1] ** 2) > 0.2:
                    if name in [
                        "car",
                        "construction_vehicle",
                        "bus",
                        "truck",
                        "trailer",
                    ]:
                        attr = "vehicle.moving"
                    elif name in ["bicycle", "motorcycle"]:
                        attr = "cycle.with_rider"
                    else:
                        attr = None
                else:
                    if name in ["pedestrian"]:
                        attr = "pedestrian.standing"
                    elif name in ["bus"]:
                        attr = "vehicle.stopped"
                    else:
                        attr = None

                nusc_anno = {
                    #"sample_token": det["token"],
                    "sample_token": det["sample_token"],
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": box.velocity[:2].tolist(),
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr
                    if attr is not None
                    else max(cls_attr_dist[name].items(), key=operator.itemgetter(1))[
                        0
                    ],
                }
                annos.append(nusc_anno)
            # nusc_annos["results"].update({det["token"]: annos})
            nusc_annos["results"].update({det["sample_token"]: annos})

        nusc_annos["meta"] = {
            "use_camera": False,
            "use_lidar": True,
            "use_radar": False,
            "use_map": False,
            "use_external": False,
        }

        name = self._info_path.split("/")[-1]
        res_path = str(Path(output_dir) / Path(name + ".json"))
        with open(res_path, "w") as f:
            json.dump(nusc_annos, f)

        print(f"Finish generate predictions for testset, save to {res_path}")

        if not testset:
            eval_main(
                nusc,
                self.eval_version,
                res_path,
                eval_set_map[self.version],
                output_dir,
            )

            with open(Path(output_dir) / "metrics_summary.json", "r") as f:
                metrics = json.load(f)

            detail = {}
            result = f"Nusc {version} Evaluation\n"
            for name in mapped_class_names:
                detail[name] = {}
                for k, v in metrics["label_aps"][name].items():
                    detail[name][f"dist@{k}"] = v
                threshs = ", ".join(list(metrics["label_aps"][name].keys()))
                scores = list(metrics["label_aps"][name].values())
                mean = sum(scores) / len(scores)
                scores = ", ".join([f"{s * 100:.2f}" for s in scores])
                result += f"{name} Nusc dist AP@{threshs}\n"
                result += scores
                result += f" mean AP: {mean}"
                result += "\n"
            res_nusc = {
                "results": {"nusc": result},
                "detail": {"nusc": detail},
            }
        else:
            res_nusc = None

        if res_nusc is not None:
            res = {
                "results": {"nusc": res_nusc["results"]["nusc"],},
                "detail": {"eval.nusc": res_nusc["detail"]["nusc"],},
            }
        else:
            res = None

        return res, None


if __name__ == "__main__":
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.multiprocessing.set_start_method("spawn")
    data_nuscenes = TrainDatasetMultiSeq(devices=devices)
    # a = data_nuscenes[79957]
    trainloader = torch.utils.data.DataLoader(data_nuscenes, batch_size=2, shuffle=True,
                                              num_workers=2, collate_fn=data_nuscenes.collate_batch)
    for i, data in enumerate(trainloader, 0):
        debug = 1
    debug = 1


