from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from typing import Tuple, List, Dict
import os
import os.path as osp
from nuscenes.utils.data_classes import LidarPointCloud, Box
import numpy as np
import argparse
from data.data_utils import voxelize_occupy, gen_2d_grid_gt
import pickle
import numpy as np
from pyquaternion import Quaternion
from multiprocessing import Pool

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='/media/pwu/Data/3D_data/nuscene/all_nuscene', type=str, help='Root path to nuScenes dataset')
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
    print("gen {} scenes start...".format(scenes_idx))
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
    while curr_sample_data['next'] is not '':
        # curr_pc = read_sweep_for_sample_data(nusc, curr_sample_data)
        src_lidar_link_name = os.path.join(args.root, curr_sample_data['filename'])
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
        curr_sample_data = nusc.get('sample_data', curr_sample_data['next'])
        offset += 1  # self add
    print("[INFO] curr data has reach the end")


if __name__ == "__main__":
    if True:
        pool = Pool(num_worker)
        res_scenes = list()
        for s in range(num_total_scenes):
           res_scenes.append(int(s))
        with pool:
           pool.map(gen_scene_data, res_scenes)
        ###############################################################
        for scense_idx in range(num_total_scenes):
            print("Start process {} data".format(scense_idx))
            gen_scene_data(scense_idx)
        print("Finished process all dataset.")

    if False:
        # prepare train list for dataloader
        print("start create train list")
        data_root = "/media/user/C14D581BDA18EBFA/nuScenesGenData"
        seq_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
        num_past_lidar = 5
        num_feature_lidar = 25
        train_list = []
        assert len(seq_dirs) == 850, "number scenes has not 850"
        for scene_idx in range(500):
            curr_scene_file_names = [d for d in os.listdir(os.path.join(data_root, "scene_" + str(scene_idx) + "/lidars"))]
            num_curr_scens_sweep = len(curr_scene_file_names)
            for sweep_idx in range(num_past_lidar, num_curr_scens_sweep - num_feature_lidar):
                train_list.append("{}_{}".format(scene_idx, sweep_idx))
            # print("scense {} has {} file".format(scene_idx, num_curr_scens_sweep))
        # check all file
        for elem in train_list:
            value = elem.split("_")
            scene_idx = int(value[0])
            sweep_idx = int(value[1])

        # save train list
        with open(data_root + "/trainlist.pkl", "wb") as f:
            pickle.dump(train_list, f)
        print("Finished create train list")