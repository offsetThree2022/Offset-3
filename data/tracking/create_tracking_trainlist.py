import pickle
import numpy as np
import os
import random
import os.path as osp
# data_root = "/media/user/C14D581BDA18EBFA/nuScenesGenData"
# with open(osp.join(data_root, "trainlist.pkl"), "rb") as f:
#     train_list = pickle.load(f)

print("start create tracking train list")
data_root = "/media/user/C14D581BDA18EBFA/nuScenesGenData"

seq_dirs = [os.path.join(data_root, d) for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]
num_past_lidar = 5
num_feature_lidar = 25
batch_size = 4
num_interval_lidar = num_past_lidar + num_feature_lidar
train_list = []
assert len(seq_dirs) == 850, "number scenes has not 850"
for scene_idx in range(500):
    curr_scene_file_names = [d for d in os.listdir(os.path.join(data_root, "scene_" + str(scene_idx) + "/lidars"))]
    num_curr_scens_sweep = len(curr_scene_file_names)
    if scene_idx == 414:
        debug = 1
    for sweep_idx in range(num_past_lidar, num_curr_scens_sweep - num_interval_lidar, num_interval_lidar):
        for idx in range(num_interval_lidar):
            train_list.append("{}_{}".format(scene_idx, sweep_idx + idx))
    # print("scense {} has {} file".format(scene_idx, num_curr_scens_sweep))

# check all file
train_list_np = np.array(train_list)
train_list_np_reshape = train_list_np.reshape(-1, num_interval_lidar)
len_seq = train_list_np_reshape.shape[0]
np.random.shuffle(train_list_np_reshape)

tracking_train_list = []
# prep tracking train list for num of per batch size by bai xu yang
for seq_idx in range(0, train_list_np_reshape.shape[0] - batch_size, batch_size):
    for sub_seq_idx in range(train_list_np_reshape.shape[1]):
        for idx in range(batch_size):
            tracking_train_list.append(train_list_np_reshape[seq_idx + idx, sub_seq_idx])

for elem in tracking_train_list:
    value = elem.split("_")
    scene_idx = int(value[0])
    sweep_idx = int(value[1])

with open(data_root + "/tracking_trainlist.pkl", "wb") as f:
    pickle.dump(tracking_train_list, f)
print("Finished create train list")


