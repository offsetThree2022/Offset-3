import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import sys
import os
from shutil import copytree, copy
from model import OffsetThree
from data.nuscenes_dataloader import TrainDatasetMultiSeq
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

import numpy as np
import configs

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/error_confidence_debug/epoch_5.pth"


need_log = configs.train.log
num_epochs = configs.train.num_epochs

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_num = torch.cuda.device_count()
print("device number", device_num)
data_nuscenes_base = TrainDatasetMultiSeq(batch_size=1, devices=devices, tracking=False, split='train')
data_nuscenes = TrainDatasetMultiSeq(batch_size=1, devices=devices, tracking=False, split='train')

print("Training dataset size:", len(data_nuscenes))

def main():
    start_epoch = 1
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)
    torch.multiprocessing.set_start_method("spawn")
    data_nuscenes = TrainDatasetMultiSeq(devices=devices, tracking=False, split='train')
    need_shuffle = not configs.tracker
    trainloader = torch.utils.data.DataLoader(data_nuscenes, batch_size=2, shuffle=need_shuffle,
                                              num_workers=2, collate_fn=data_nuscenes.collate_batch)
    # for tracking set is trainging for false, when not traing with end to end
    model = OffsetThree(num_feature_channel=configs.bird.num_feature_channel, device=devices, is_training=True)
    model = nn.DataParallel(model)
    model = model.to(devices)

    checkpoint = torch.load(resume_det)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # freeze detect model param
    for param in model.parameters():
        param.requires_grad = False

    eval_one(model, trainloader)
    debug = 1


fig, ax = plt.subplots(2, 4, figsize=(40, 20))


def eval_one(model, trainloader):
    fig.tight_layout()
    data_idx_0 = 79982
    data_idx2 = 11
    need_shuffle = False
    i = 0
    # for data_idx in range(data_idx_0, data_idx_0 + 3000, 1):
    for i, data_dict in enumerate(trainloader, 0):
        # data_dict = data_nuscenes[data_idx]
        # data_idx2 = 75516
        # data_dict2 = data_nuscenes[data_idx2]
        # for i, data_dict in enumerate(trainloader, 0):
        scene_idxes = data_dict['scene_idx']
        sweep_idxes = data_dict['sweep_idx']
        # gen detect bbox for training
        det_res_dict = model(data_dict)[2]

        confidence_pred = det_res_dict['confidence_pred']
        voxel_count_gt = det_res_dict['voxel_count_gt']
        pixel_pred = det_res_dict['pixel_pred']
        offset_pred = det_res_dict['offset_pred']
        data_idx = i

        all_pc = data_dict['points'][0]
        #  draw points

        ax[0, 0].set_xlim([-1.0 * configs.bird.extents[0][1], configs.bird.extents[0][1]])
        ax[0, 0].set_ylim([-1.0 * configs.bird.extents[1][1], configs.bird.extents[1][1]])
        ax[0, 0].scatter(all_pc[:, 0], all_pc[:, 1])
        ax[0, 0].axis('off')
        ax[0, 0].set_aspect('equal')
        ax[0, 0].title.set_text('Lidar data')
        rows = configs.bird.rows
        cols = configs.bird.cols

        # boxes_tensor = det_res_dict['bboxes'][0]
        # boxes = boxes_tensor.detach().cpu().numpy()
        # for j in range(boxes.shape[0]):
        #     inst = boxes[j, :]
        #     # if np.isnan(inst).any():
        #     #     continue
        #     size_data = [inst[4], inst[3], inst[5]]
        #     box = Box(center=inst[:3], size=size_data, orientation=Quaternion(
        #         axis=[0, 0, 1], angle=inst[6]))
        #     box.render(ax[0, 0])
        #  draw pixel pred
        pixel_pred_numpy = pixel_pred[0].view(-1, rows, cols).contiguous().detach().cpu().numpy()[0, :, :]
        voxel_count_gt_np = voxel_count_gt[0].view(-1, rows, cols).contiguous().detach().cpu().numpy()[0, :, :]
        pixel_mask = (pixel_pred_numpy > 0.05) * (voxel_count_gt_np > 0.5)
        idx_x = np.arange(rows)
        idx_y = np.arange(cols)
        idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')

        ax[0, 1].imshow(pixel_mask.T)
        ax[0, 1].set_aspect('equal')
        ax[0, 1].axis('off')
        ax[0, 1].title.set_text('pixel pred')

        #  draw pixel gt
        pixel_gt_numpy = det_res_dict['pixel_target_map'][0].detach().cpu().numpy().squeeze()
        pixel_gt_mask_new = (pixel_gt_numpy > 0.1) * (voxel_count_gt_np > 0.5)
        ax[0, 2].imshow(pixel_gt_mask_new.T)
        ax[0, 2].set_aspect('equal')
        ax[0, 2].axis('off')
        ax[0, 2].title.set_text('pixel gt')

        #  draw confidence pred
        confidence_pred_numpy = confidence_pred[0].view(-1, rows, cols).contiguous().detach().cpu().numpy()[0, :, :]
        confidence_mask = (confidence_pred_numpy > 0.6) * (voxel_count_gt_np > 0.5)
        idx_x = np.arange(rows)
        idx_y = np.arange(cols)
        idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
        ax[0, 3].imshow(confidence_mask.T)
        ax[0, 3].set_aspect('equal')
        ax[0, 3].axis('off')
        ax[0, 3].title.set_text('confidence pred')

        #  draw offset pred
        pixel_gt_mask = (pixel_gt_numpy > 0.1)
        grid_size = (configs.bird.extents[0][1] - configs.bird.extents[0][0]) / configs.bird.rows
        pos_nonzeros_tuple = np.where(pixel_gt_mask == True)
        #  too mush positive pixel need to show, we sample it
        pox_x, pos_y = pos_nonzeros_tuple[0], pos_nonzeros_tuple[1]
        pos_selected_mask = np.zeros_like((pixel_gt_numpy > 0.5)).astype(np.bool)
        pos_selected_mask[pox_x, pos_y] = True
        offset_pred_map_numpy = offset_pred[0].detach().cpu().numpy().squeeze()
        X = idx_x[pos_selected_mask]
        Y = idx_y[pos_selected_mask]
        U = -offset_pred_map_numpy[0, :, :][pos_selected_mask] / grid_size
        V = -offset_pred_map_numpy[1, :, :][pos_selected_mask] / grid_size
        ax[1, 0].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r')
        ax[1, 0].set_aspect('equal')
        ax[1, 0].title.set_text('offset Prediction')
        ax[1, 0].axis('off')
        ax[1, 0].set_xlim([0, configs.bird.rows])
        ax[1, 0].set_ylim([0, configs.bird.cols])

        #  draw offset gt
        pos_nonzeros_tuple = np.where(pixel_gt_mask == True)
        #  too mush positive pixel need to show, we sample it
        pox_x, pos_y = pos_nonzeros_tuple[0], pos_nonzeros_tuple[1]
        pos_selected_mask = np.zeros_like((pixel_gt_numpy > 0.5)).astype(np.bool)
        pos_selected_mask[pox_x, pos_y] = True
        offset_target_map_numpy = det_res_dict['offset_target_map'][0].detach().cpu().numpy().squeeze()
        X = idx_x[pos_selected_mask]
        Y = idx_y[pos_selected_mask]
        U = -offset_target_map_numpy[0, :, :][pos_selected_mask] / grid_size
        V = -offset_target_map_numpy[1, :, :][pos_selected_mask] / grid_size
        ax[1, 1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r')
        ax[1, 1].set_aspect('equal')
        ax[1, 1].title.set_text('offset ground truth')
        ax[1, 1].axis('off')
        ax[1, 1].set_xlim([0, configs.bird.rows])
        ax[1, 1].set_ylim([0, configs.bird.cols])

        #  draw velocity pred
        velocity_pred = det_res_dict['velocity_pred']
        grid_size = (configs.bird.extents[0][1] - configs.bird.extents[0][0]) / configs.bird.rows
        pos_selected_mask = np.zeros_like((pixel_gt_numpy > 0.5)).astype(np.bool)
        pos_selected_mask[pox_x[::5], pos_y[::5]] = True
        velocity_pred_map_numpy = velocity_pred[0].detach().cpu().numpy().squeeze()
        X = idx_x[pos_selected_mask]
        Y = idx_y[pos_selected_mask]
        U = velocity_pred_map_numpy[0, :, :][pos_selected_mask] / grid_size
        V = velocity_pred_map_numpy[1, :, :][pos_selected_mask] / grid_size
        ax[1, 2].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='g')
        ax[1, 2].set_aspect('equal')
        ax[1, 2].title.set_text('velocity Prediction')
        ax[1, 2].axis('off')
        ax[1, 2].set_xlim([0, configs.bird.rows])
        ax[1, 2].set_ylim([0, configs.bird.cols])

        #  draw velocity pred
        velocity_target_map_numpy = det_res_dict['velocity_target_map'][0].detach().cpu().numpy().squeeze()
        U = velocity_target_map_numpy[0, :, :][pos_selected_mask] / grid_size
        V = velocity_target_map_numpy[1, :, :][pos_selected_mask] / grid_size
        ax[1, 3].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='g')
        ax[1, 3].set_aspect('equal')
        ax[1, 3].title.set_text('velocity gt')
        ax[1, 3].axis('off')
        ax[1, 3].set_xlim([0, configs.bird.rows])
        ax[1, 3].set_ylim([0, configs.bird.cols])

        file_name = os.path.join("img/", str(data_idx) + '.png')
        print("[INFO] save {} finished".format(file_name))
        plt.savefig(file_name)
        i += 1
        ax[0, 0].clear()
        ax[0, 1].clear()
        ax[0, 2].clear()
        ax[0, 3].clear()
        ax[1, 0].clear()
        ax[1, 1].clear()
        ax[1, 2].clear()
        ax[1, 3].clear()
    plt.close()


if __name__ == "__main__":
    main()
