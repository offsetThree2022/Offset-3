#!/usr/bin/python
from __future__ import division
from __future__ import print_function
import math


class data:
    data_root = "/media/user/C14D581BDA18EBFA/nuScenesGenData"
    train_info_path = ""
    val_info_path = ""
    info_path = "work_dir/dist_test/val_json"
    batch_size = 3
    num_worker = 3
    num_past_lidar = 9
    num_future_lidar = 25

    class train_preprocessor:
        no_augmentation = False,
        shuffle_points = True,
        global_rot_noise = [-0.7850, 0.7850],
        global_scale_noise = [0.90, 1.10]
        global_translate_std = 0.5

class bird:
    rows = 512
    cols = 512
    # rows = 512
    # cols = 512
    # voxel_num = 20
    voxel_num = 10
    feature_num = 8
    num_feature_channel = voxel_num * feature_num
    extents = [[-51.2, 51.2], [-51.2, 51.2], [-5., 3.]]
    resolution = (extents[0][1] - extents[0][0]) / rows


class train:
    log = "."
    num_epochs = 36
    # resume_det = None
    # resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/det_da_6epoch_0724/epoch_6.pth"
    resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/det_corret_voxel_count_gt_re_train/epoch_5.pth"
    # resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/det_da_6epoch_0724/epoch_5.pth"
    # resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/box_512_baseline/epoch_5.pth"
    # resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/over_fitting_pre_200_2/epoch_20.pth"
    # resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/bbox_model_first/epoch_10.pth"
    # resume_tracking = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/box_512_baseline/epoch_12.pth"
    resume_tracking = None
    gen_bbox = True  # whether gen bbox for train stage
    without_loss = False  # for train stage, not return loss
    # num_epochs = 20
    # resume = None
    # resume = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/det_da_6epoch_0724/epoch_6.pth"
    # resume = None

    class det_stage1_weight:
        pixel = 2.0
        confidence = 1.0
        velocity = 1.0
        offset = 5.0
        size = 1.0
        height = 1.0
        yaw = 1.0
        category = 0.5

class val:
    model_path = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/bbox_model_first/epoch_10.pth"
    batch_size = 1
    num_worker = 1
    shuffle = False


class traj:
    load_model_tarj = None


class tracker:
    class train:
        batch_size = 4
        save_dir = ""
    num_past_lidar = 9  # for data loader batch_size
    num_feature_lidar = 25  # for data loader batch_size
    num_mature_size = 3  # how much age are tracker to be a mature tracker

    tracker = True
    Max_record_frame = 25
    decay = 1.0
    decay2 = 0.01
    max_track_node = 50
    max_object = None
    track_buffer = None
    lstm = None

    TRACKING_NAMES = ['bicycle', 'bus', 'car', 'motorcycle', 'pedestrian', 'trailer', 'truck']
    TRACKING_NAMES_INT = [4, 3, 2, 5, 6, 0, 1]

    class match:
        threshold_gt_match_det = 0.3

    share_rc = [256, 128, 64, 1]  # input feature matrix get score of affinity matrix
    DP_RATIO = 0

    class Loss:
        margin_threshold = 0.2  # max margin

    class val:
        batch_size = 1
        resume = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/det_da_6epoch_0724/epoch_6.pth"
