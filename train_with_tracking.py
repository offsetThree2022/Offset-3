import math

import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import configs
import time
import sys
import os
from shutil import copytree, copy
from model import OffsetThree
from data.nuscenes_dataloader import TrainDatasetMultiSeq
import ops.iou3d_nums.iou3d_nus_utils as iou3d_nms_utils

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")
import numpy as np

from utils.tracker import TrackingModel

np.set_printoptions(suppress=True)

need_log = configs.train.log
num_epochs = configs.train.num_epochs


def main():
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)
    torch.multiprocessing.set_start_method("spawn")
    data_nuscenes = TrainDatasetMultiSeq(batch_size=configs.tracker.train.batch_size, devices=devices,
                                         tracking=True, split="train")
    need_shuffle = not configs.tracker
    trainloader = torch.utils.data.DataLoader(data_nuscenes,
                                              batch_size=configs.tracker.train.batch_size,
                                              shuffle=need_shuffle,
                                              num_workers=configs.data.num_worker,
                                              collate_fn=data_nuscenes.collate_batch)
    # for tracking set is training for false, when not training with end to end
    # model_detect = OffsetThree(num_feature_channel=configs.bird.num_feature_channel, device=devices, is_training=False)
    model_detect = OffsetThree(num_feature_channel=configs.bird.num_feature_channel,
                             batch_size=configs.tracker.train.batch_size, device=devices, is_training=False)
    model_detect = nn.DataParallel(model_detect)
    model_detect = model_detect.to(devices)

    model_tracking = TrackingModel()
    # model_tracking = nn.DataParallel(model_tracking)
    model_tracking = model_tracking.to(devices)

    optimizer_detect = optim.Adam(model_detect.parameters(), lr=0.00006)
    scheduler_detect = torch.optim.lr_scheduler.MultiStepLR(optimizer_detect, milestones=[2, 4, 6, 8], gamma=0.5)

    optimizer_tracking = optim.Adam(model_tracking.parameters(), lr=0.0016)
    scheduler_tracking = torch.optim.lr_scheduler.MultiStepLR(optimizer_tracking, milestones=[2, 4, 6, 8], gamma=0.5)

    if configs.train.resume_det is not None:
        checkpoint = torch.load(configs.train.resume_det)
        start_epoch = checkpoint['epoch'] + 1
        model_detect.load_state_dict(checkpoint['model_state_dict'])
        optimizer_detect.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler_detect.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Load model from {}, at epoch {}".format(configs.train.resume_det, start_epoch - 1))

        # freeze detect model param
        for param in model_detect.parameters():
            param.requires_grad = False

    if configs.train.resume_tracking is not None:
        checkpoint_track = torch.load(configs.train.resume_tracking)
        tracking_epoch = checkpoint_track['epoch'] + 1
        model_tracking.load_state_dict(checkpoint_track['model_state_dict'])
        print("Load tracking model from {}, at epoch {}".format(configs.train.resume_det, tracking_epoch - 1))

    # for train tracking
    start_epoch = 0
    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer_tracking.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))
        # model.train()
        model_detect.eval()
        loss = train(model_detect, model_tracking, trainloader, optimizer_tracking, devices, epoch, scheduler_tracking)
        if epoch % 1 == 0 or epoch == num_epochs or epoch == 1:
            save_dict = {'epoch': epoch,
                         'model_state_dict': model_tracking.state_dict(),
                         'optimizer_state_dict': optimizer_tracking.state_dict(),
                         'scheduler_state_dict': scheduler_tracking.state_dict()}
            torch.save(save_dict, os.path.join("checkpoint/tracking_ass_new_motion/", 'epoch_' + str(epoch) + '.pth'))


def train_tracking(data_dict, det_res_dict, model_tracking, frame_idx):
    """
        brief:
            train tracking one iter
        Input:
            seq_idx: 0, num_total_seq_idx, for time sequence data flow in one iter
            data_dict: all input data_dict for one for seq_data
            model_tracking tracker for sequence data
            det_res_dict: detect model result, for tracking input
            optimizer: training optimizer
            scheduler: lr scheduler
            frame_idx: curr frame idx
    """
    # get curr det, and gt boxes iou matrix, both box need convert to iou matrix inputs
    track_loss = model_tracking(data_dict, det_res_dict, is_training=True)
    return track_loss


def train(model_detect,
          model_tracking,
          trainloader,
          optimizer_tracking,
          device,
          epoch,
          scheduler_tracking):
    seq_idx = 0
    num_path_lidar = configs.tracker.num_past_lidar
    num_feature_lidar = configs.tracker.num_feature_lidar
    num_interval_lidar = num_path_lidar + num_feature_lidar
    for i, data_dict in enumerate(trainloader, 0):
        scene_idx = data_dict['scene_idx']
        sweep_idx = data_dict['sweep_idx']
        # gen detect bbox for training
        det_res_dict = model_detect(data_dict)
        #  for training tracking
        loss_tracking = train_tracking(data_dict, det_res_dict, model_tracking, seq_idx)
        optimizer_tracking.zero_grad()
        if seq_idx != 0:
            if isinstance(loss_tracking, torch.Tensor):
                loss_tracking.backward()
                print("{}/{}, [{}_{}] : frame_idx [{}]: loss: {}".format(epoch, i, scene_idx, sweep_idx,
                                                                         seq_idx, loss_tracking.item()))
            else:
                print("0.0 for no box")
                print("{}/{}, [{}_{}] : frame_idx [{}]: loss: {}".format(epoch, i, scene_idx, sweep_idx,
                                                                         seq_idx, loss_tracking))

        else:
            print("{}/{}, [{}_{}] : frame_idx [{}]: loss: {}".format(epoch, i, scene_idx, sweep_idx,
                                                                     seq_idx, 0.0))
        optimizer_tracking.step()
        scheduler_tracking.step()
        seq_idx += 1
        if seq_idx >= num_interval_lidar:
            seq_idx = 0  # reset when reach num_frame_pre_train_tracking_frame
            model_tracking.reset_tracker()

    loss = 0
    return loss


if __name__ == "__main__":
    main()
