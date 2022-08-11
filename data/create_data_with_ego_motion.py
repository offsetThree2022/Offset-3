from data import splits
# import splits
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from typing import Tuple, List, Dict
import os
import os.path as osp
from nuscenes.utils.data_classes import LidarPointCloud, Box
import numpy as np
import argparse
import pickle
import numpy as np
from pyquaternion import Quaternion

from multiprocessing import Pool
import json
parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='/media/user/C14D581BDA18EBFA/nuScenesFull', type=str, help='Root path to nuScenes dataset')
parser.add_argument('-s', '--split', default='train', type=str, help='The data split [train/val/test]')
parser.add_argument('-p', '--savepath', default='/media/user/C14D581BDA18EBFA/nuScenesGenData', type=str, help='Directory for save nuscenes gen Dataset')

args = parser.parse_args()

num_worker = 30  # num worker to multi process
num_total_scenes = 850
class_map = {'vehicle.car': 1, 'vehicle.bus.rigid': 1, 'vehicle.bus.bendy': 1, 'human.pedestrian': 2,
             'vehicle.bicycle': 3}  # background: 0, other: 4

with open("/media/user/C14D581BDA18EBFA/nuScenesFull/nusc.pkl", "rb") as readNusc:
    nusc = pickle.load(readNusc)
print("Total number of scenes:", len(nusc.scene))

def mkdir(target):
    if not os.path.exists(target):
        print("mkdir: " + str(target))
        os.mkdir(target)


def make_target_dir(savePath, scenes_idx):
    # make scenes dir
    scenes_dir_name = os.path.join(savePath, "scene_" + str(scenes_idx))
    mkdir(scenes_dir_name)
    # make lidar dir
    lidar_dir_name = os.path.join(scenes_dir_name, "lidars")
    mkdir(lidar_dir_name)
    # make boxes dir
    box_dir_name = os.path.join(scenes_dir_name, "boxes")
    mkdir(box_dir_name)
    # make calibration
    calibration_dir_name = os.path.join(scenes_dir_name, "calibration")
    mkdir(calibration_dir_name)

    ego_motion_dir = os.path.join(scenes_dir_name, "ego_motion")
    mkdir(ego_motion_dir)

    token_dir = os.path.join(scenes_dir_name, "token")
    mkdir(token_dir)

def read_sweep_for_sample_data(self, nusc,
                               ref_sd_rec: Dict,
                               min_distance: float=1.0):
    """
        get sample data point cloud
    """
    current_pc = LidarPointCloud.from_file(osp.join(nusc.dataroot, ref_sd_rec['filename']))
    current_pc.remove_close(min_distance)
    return current_pc.points


def get_sweep_box_and_save_id(nusc,
                              ref_sd_rec,
                              instance_to_id_dict,
                              id_idx,
                              last_ref_pose_rec,
                              last_ref_cs_rec,
                              lastSweepBoxes):
    """
        get sweep box and save box id, if box is the sample instances
        make sure sweep not sample
    """
    corresponding_sample_rec = nusc.get('sample', ref_sd_rec['sample_token'])
    # Map the bounding boxes to the local sensor coordinate
    # Get reference pose and timestamp
    ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
    box_list = list()
    attr_list = list()
    cat_list = list()
    id_list = list()

    for curr_sweep_box_taken in corresponding_sample_rec['anns']:
        ann_rec = nusc.get('sample_annotation', curr_sweep_box_taken)
        category_name = ann_rec['category_name']
        instance_token = ann_rec['instance_token']
        box, attr, cat = nusc.get_instance_box(ref_sd_rec['token'], instance_token)
        if box is not None:
            attr_list.append(attr)
            cat_list.append(cat)
            if instance_token in instance_to_id_dict.keys():
                id_list.append(instance_to_id_dict[instance_token])
            else:
                instance_to_id_dict.update({instance_token: id_idx})
                id_list.append(id_idx)
                id_idx += 1
            # Move box to ego vehicle coord system
            box.translate(-np.array(ref_pose_rec['translation']))
            box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)
            # Move box to sensor coord system
            box.translate(-np.array(ref_cs_rec['translation']))
            box.rotate(Quaternion(ref_cs_rec['rotation']).inverse)
            # convert to self define Bbox
            box.id = id_list[-1]
            # row = np.array([*box.center, *box.wlh, box.orientation.yaw_pitch_roll[0]], dtype=np.float32)
            # box_save = Bbox(*row, id_list[-1])
            # convert category to self define
            flag = False
            for c, v in class_map.items():
                if category_name.startswith(c):
                    box.category = v
                    flag = True
                    break
            if not flag:
                box.category = 4  # Other category
            box_list.append(box)

    time_diff = None
    if last_ref_pose_rec is not None:
        time_diff = (1e-6 * ref_pose_rec['timestamp']) - (1e-6 * last_ref_pose_rec['timestamp'])
    # calculate velocity
    for idx, currBox in enumerate(box_list):
        box_list[idx].velocity = np.array([np.nan, np.nan, np.nan])
        for lastSweepBox in lastSweepBoxes:
            if currBox.id == lastSweepBox.id:
                ## move coord to global
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

                # box_list.append(lastSweepBox)
                ## set velocity
                pos_diff = currBox.center - lastSweepBox.center
                box_list[idx].velocity = pos_diff / time_diff
                break

    # update lastSweepBoxes
    lastSweepBoxes = box_list
    last_ref_cs_rec = ref_cs_rec
    last_ref_pose_rec = ref_pose_rec

    return box_list, attr_list, cat_list, id_list, ref_pose_rec, ref_cs_rec, id_idx, \
           lastSweepBoxes, last_ref_cs_rec, last_ref_pose_rec

def gen_scene_data(scenes_idx):
    print("[INFO] gen {} scenes start...".format(scenes_idx))
    instance_to_id_dict = {}
    id_idx = 0
    curr_scenes = nusc.scene[scenes_idx]
    first_sample_token = curr_scenes['first_sample_token']
    curr_sample = nusc.get('sample', first_sample_token)
    curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
    print("Processing scene {} ...".format(scenes_idx))
    offset = 0
    curr_sample_data_index = 0
    savePath = args.savepath
    make_target_dir(savePath, scenes_idx)
    lastSweepBoxes = list()
    last_ref_pose_rec = None
    last_ref_cs_rec = None

    # to_next_sample
    # load ego_pose.json
    last_yaw = None
    last_ts = None
    last_trans = None
    while curr_sample_data is not '':
        # curr_pc = read_sweep_for_sample_data(nusc, curr_sample_data)
        # link nuscenes lidar data to ln -s dir
        src_lidar_link_name = os.path.join(args.root, curr_sample_data['filename'])
        target_lidar_link_name = os.path.join(savePath, "scene_{}/lidars/{}.bin".format(scenes_idx, offset))
        os.symlink(src_lidar_link_name, target_lidar_link_name)
        # read box
        boxes, attr_list, cat_list, id_list, ref_pose_rec, ref_cs_rec, id_idx, lastSweepBoxes, last_ref_cs_rec, \
            last_ref_pose_rec = get_sweep_box_and_save_id(nusc, curr_sample_data, instance_to_id_dict, id_idx,
                                                          last_ref_pose_rec, last_ref_cs_rec, lastSweepBoxes)
        # save box as pkl to targe dir name
        target_box_pkl_name = os.path.join(savePath, "scene_{}/boxes/{}.pkl".format(scenes_idx, offset))
        with open(target_box_pkl_name, "wb") as f:
            pickle.dump(boxes, f)
        # save calibration to target dir name
        target_calibration_pkl_name = os.path.join(savePath, "scene_{}/calibration/{}.pkl".format(scenes_idx, offset))
        with open(target_calibration_pkl_name, "wb") as f:
            pickle.dump([ref_pose_rec, ref_cs_rec], f)

        # curr_token = curr_sample_data["sample_token"]  # error code, pay more attention
        curr_token = curr_sample_data["token"]
        target_token_txt_name = os.path.join(savePath, "scene_{}/token/{}.txt".format(scenes_idx, offset))
        with open(target_token_txt_name, "w") as file:
            file.writelines(curr_token)
        curr_ego_info = ego_pose_json_dict[curr_token]
        curr_yaw = curr_ego_info[0]
        curr_trans = curr_ego_info[1]
        curr_ts = curr_ego_info[2]

        if last_yaw is None:
            last_yaw = curr_yaw
            last_ts = curr_ts
            last_trans = curr_trans
            deltaX = [0, 0]
            deltaTheta = 0
        else:
            deltaT = 1e-6 * (curr_ts - last_ts)
            curr_delta_yaw = (curr_yaw - last_yaw) / deltaT
            curr_delta_trans = [(curr_trans[0] - last_trans[0]) / deltaT, (curr_trans[1] - last_trans[1]) / deltaT]
            last_trans = curr_trans
            last_yaw = curr_yaw
            last_ts = curr_ts
            deltaX = curr_delta_trans
            deltaTheta = curr_delta_yaw
        target_ego_motion_pkl_name = os.path.join(savePath, "scene_{}/ego_motion/{}.pkl".format(scenes_idx, offset))
        with open(target_ego_motion_pkl_name, "wb") as file:
            pickle.dump([deltaX[0], deltaX[1], deltaTheta], file)
        if curr_sample_data['next'] is not '':
            curr_sample_data = nusc.get('sample_data', curr_sample_data['next'])
        else:
            curr_sample_data = ''
        offset += 1  # self add
    print("[INFO] curr data has reach the end")

def get_sweep_keyframe_train_val_test_attr(sweep_token_filename, splits_dict):
    """
    @param sweep_token_filename: sweep token filename
    @param splits_dict: split scenes for train val test
    @return: None
    """
    with open(sweep_token_filename, "r") as file:
        curr_sweep_token = file.readline()
    curr_sweep_sample_token = nusc.get('sample_data', curr_sweep_token)['sample_token']
    curr_scene_token = nusc.get('sample', curr_sweep_sample_token)['scene_token']
    curr_scene_record = nusc.get('scene', curr_scene_token)
    curr_scene_name = curr_scene_record['name']
    is_key_frame = int(nusc.get('sample_data', curr_sweep_token)['is_key_frame'])
    if curr_scene_name in splits_dict['train']:
        name_data = "train"
    elif curr_scene_name in splits_dict["val"]:
        name_data = "val"
    elif curr_scene_name in splits_dict["test"]:
        name_data = "test"
    else:
        print("[ERROR] curr_scene_name not in splits")
    return name_data, is_key_frame, curr_sweep_sample_token, curr_sweep_token


with open(os.path.join(args.root, "v1.0-trainval/ego_pose.json"), "r") as f:
    ego_pose_json = json.load(f)

ego_pose_json_dict = {}
for data in ego_pose_json:
    token = data["token"]
    rotation = data["rotation"]
    translation = data["translation"][:2]
    timestamp = data["timestamp"]
    yaw = Quaternion(rotation).yaw_pitch_roll[0]
    ego_pose_json_dict.update({token: [yaw, translation, timestamp]})


if __name__ == "__main__":
    data_root = "/media/user/C14D581BDA18EBFA/nuScenesGenData"
    num_past_lidar = 9
    num_feature_lidar = 25
    num_feature_lidar_val = 0
    if False:  # gen data and mkdir bbox calibration motion token etc.
        pool = Pool(num_worker)
        res_scenes = list()
        for s in range(num_total_scenes):
           res_scenes.append(int(s))
        with pool:
           pool.map(gen_scene_data, res_scenes)
        #############################################################
        # for scense_idx in range(num_total_scenes):
        #     print("Start process {} data".format(scense_idx))
        #     gen_scene_data(scense_idx)
        print("Finished process all dataset.")

    if True:  # gen train list pkl
        # For a Test
        scene2dir = {}
        for idx in range(num_total_scenes):
            curr_scenes = nusc.scene[idx]
            scene2dir.update({curr_scenes['name']: idx})
        sample_tokens_all = [s['token'] for s in nusc.sample]
        sample_tokens = []
        splits_dict = splits.create_splits_scenes()
        val_sample_data_list = []
        for sample_token in sample_tokens_all:
            curr_sample = nusc.get('sample', sample_token)
            scene_token = curr_sample['scene_token']
            scene_record = nusc.get('scene', scene_token)
            if scene_record['name'] in splits_dict['val']:
                sample_tokens.append(sample_token)
                curr_sample_data_token = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])['token']
                val_sample_data_list.append(curr_sample_data_token)

        # prepare train list for dataloader
        print("[INFO] start create train list")
        seq_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        train_list = []
        val_list = []
        val_list_key_frame = []
        total_key_val_frame = 0
        assert len(seq_dirs) == 850, "number scenes has not 850"
        # splits_dict = splits.create_splits_scenes()
        val_dict_debug = {}
        for dir_idx in range(seq_dirs.__len__()):
            curr_scene_file_names = [d for d in os.listdir(os.path.join(data_root, "scene_"
                                                                        + str(dir_idx) + "/lidars"))]
            num_curr_scens_sweep = len(curr_scene_file_names)
            first_curr_scenes_name = curr_sweep_token_filename = os.path.join(data_root, "scene_" + str(dir_idx)
                                                                              + "/token/{}.txt".format(0))
            data_split_name, _, _, _ = get_sweep_keyframe_train_val_test_attr(first_curr_scenes_name, splits_dict)
            if data_split_name == "train":
                for sweep_idx in range(num_past_lidar, num_curr_scens_sweep - num_feature_lidar):
                    curr_sweep_token_filename = os.path.join(data_root, "scene_" + str(dir_idx)
                                                             + "/token/{}.txt".format(sweep_idx))
                    data_split_name, is_key_frame, curr_sample_token, curr_sweep_token = \
                        get_sweep_keyframe_train_val_test_attr(curr_sweep_token_filename, splits_dict)
                    train_list.append("{}_{}_{}".format(dir_idx, sweep_idx, is_key_frame))
            elif data_split_name == "val":
                # for sweep_idx in range(num_past_lidar, num_curr_scens_sweep - num_feature_lidar_val):
                # for sweep_idx in range(num_past_lidar, num_curr_scens_sweep):
                for sweep_idx in range(num_curr_scens_sweep):
                    curr_sweep_token_filename = os.path.join(data_root, "scene_" + str(dir_idx)
                                                             + "/token/{}.txt".format(sweep_idx))
                    data_split_name, is_key_frame, curr_sample_token, curr_sweep_token = \
                        get_sweep_keyframe_train_val_test_attr(curr_sweep_token_filename, splits_dict)
                    val_dict_debug.update({curr_sweep_token: [dir_idx, sweep_idx]})
                    if is_key_frame:
                        total_key_val_frame += 1
                        val_list.append("{}_{}_{}".format(dir_idx, sweep_idx, is_key_frame))
            else:
                print("[WARN] data_split_name: {}".format(data_split_name))
        print("[INFO] Has {} val frame. ".format(total_key_val_frame))
        # print("scenes {} has {} file".format(scene_idx, num_curr_scenes_sweep))
        # save train list
        # for elem in val_sample_data_list:
        #     if elem not in val_dict_debug:
        #         print("WARN ==================: {}".format(elem))
        #     else:
        #         print("INFO: {}: {}".format(elem, val_dict_debug[elem]))
        # idx = 0
        # for elem in val_dict_debug.keys():
        #     if elem in sample_tokens:
        #         idx += 1
        #         print("{} : {}".format(elem, val_dict_debug[elem]))
        with open(data_root + "/trainlist.pkl", "wb") as f:
            pickle.dump(train_list, f)
        print("Finished create train list")
        # save val list
        with open(data_root + "/val_list.pkl", "wb") as f:
            pickle.dump(val_list, f)
        print("Finished create train, val list")