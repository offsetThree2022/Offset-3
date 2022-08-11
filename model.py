import time
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import configs
from ops.rtree import rtree_utils
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from data.nuscenes_base import Int2Eval

from ops.roiaware_pool3d import roiaware_pool3d_utils
from data.sampler.preprocess import Preprocess
from utils.data_utils import convert_pickle_boxes_to_torch_box


class MotionPrediction(nn.Module):
    def __init__(self, seq_len):
        super(MotionPrediction, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 2 * seq_len, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class Conv3D(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding):
        super(Conv3D, self).__init__()
        self.conv3d = nn.Conv3d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn3d = nn.BatchNorm3d(out_channel)

    def forward(self, x):
        # input x: (batch, seq, c, h, w)
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, c, seq_len, h, w)
        x = F.relu(self.bn3d(self.conv3d(x)))
        x = x.permute(0, 2, 1, 3, 4).contiguous()  # (batch, seq_len, c, h, w)

        return x

class CategoryEstimation(nn.Module):
    """ CategoryEstimation """
    def __init__(self, category_dim=10):
        super(CategoryEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, category_dim, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn(self.conv(x))
        x = F.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, 32)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x


class pixelClassification(nn.Module):
    def __init__(self, category_num=1):
        super(pixelClassification, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x


class ConfidenceClassification(nn.Module):
    def __init__(self, category_num=1):
        super(ConfidenceClassification, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, category_num, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)
        x = torch.sigmoid(x)
        return x


class OffsetsEstimation(nn.Module):
    def __init__(self, offset_dimention=2):
        super(OffsetsEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, offset_dimention, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class VelocityEstimation(nn.Module):
    def __init__(self, velocity_dimention=2):
        super(VelocityEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, velocity_dimention, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class HeightEstimation(nn.Module):
    def __init__(self, height_dimention=2):
        super(HeightEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, height_dimention, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class SizeEstimation(nn.Module):
    def __init__(self, size_dimention=2):
        super(SizeEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, size_dimention, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class YawEstimation(nn.Module):
    def __init__(self, yaw_dimention=2):
        super(YawEstimation, self).__init__()
        self.conv1 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, yaw_dimention, kernel_size=1, stride=1, padding=0)

        self.bn1 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv2(x)

        return x


class VFE(nn.Module):
    def __init__(self, devices):
        super(VFE, self).__init__()
        self.rows = configs.bird.rows
        self.cols = configs.bird.cols

        self.extents = torch.Tensor(np.array(configs.bird.extents).astype(np.float)).float().cuda()
        self.voxel_num = configs.bird.voxel_num
        self.feature_num = configs.bird.feature_num
        self.devices = devices

    def forward(self, batch_dict):
        batch_dict['points'] = [torch.from_numpy(np.ascontiguousarray(
            elem)).to(self.devices).float() for elem in batch_dict['points']]
        batch_dict['pc'] = [torch.from_numpy(np.ascontiguousarray(
            elem)).to(self.devices).float() for elem in batch_dict['pc']]
        inputs_pc_list = batch_dict['points']
        inputs_cur_pc_list = batch_dict['pc']
        batch_size = inputs_pc_list.__len__()
        val = batch_dict['gt_boxes_orig']
        max_gt = max([x.shape[0] for x in val])
        batch_gt_boxes3d = torch.zeros(size=(batch_size, max_gt, 10),
                                       device=self.devices, dtype=torch.float)
        for k in range(batch_size):  # val[x] shape is [1, NBoxes, dim]
            batch_gt_boxes3d[k, :val[k].shape[0], :] = \
                torch.from_numpy(val[k]).float().to(self.devices).contiguous()
        batch_dict['gt_boxes_torch'] = batch_gt_boxes3d
        voxel_feature_list = []
        view_index_list = []
        voxel_count_map_list = []
        for bs_idx in range(batch_size):
            voxel_feature_tmp = roiaware_pool3d_utils.build_voxel_feature(inputs_pc_list[bs_idx].unsqueeze(dim=0),
                                                                          self.extents, self.rows, self.cols,
                                                                          self.voxel_num, self.feature_num,
                                                                          self.devices)
            view_index_tmp = roiaware_pool3d_utils.build_view_index(inputs_cur_pc_list[bs_idx].unsqueeze(dim=0),
                                                                    self.extents, self.rows, self.cols,
                                                                    batch_gt_boxes3d[bs_idx].unsqueeze(dim=0),
                                                                    self.devices)
            view_index_list.append(view_index_tmp)
            voxel_feature_list.append(voxel_feature_tmp)
            # voxel_count_map_tmp = voxel_feature_tmp[:, :, :, 7]  # 7 means voxel count channel
            # for idx in range(1, configs.bird.voxel_num):
            #     voxel_count_map_tmp += voxel_feature_tmp[:, :, :, 7 + idx * self.feature_num]
            # voxel_count_map_list.append(voxel_count_map_tmp)
            voxel_count_map_list.append(view_index_tmp[:, :, :, 0])
        voxel_feature = torch.cat(voxel_feature_list, dim=0)
        voxel_count_map = torch.cat(voxel_count_map_list, dim=0)
        mask = (voxel_count_map >= 0.01)
        voxel_count_map = torch.log1p(voxel_count_map) + 1.0
        voxel_count_map *= mask
        view_index_map = torch.cat(view_index_list, dim=0)
        batch_dict.update({'voxel_feature': voxel_feature,
                           'voxel_count_map': voxel_count_map,
                           'view_index_map': view_index_map})
        return batch_dict

class DataPreprocess(nn.Module):
    def __init__(self, devices, is_training):
        super(DataPreprocess, self).__init__()
        self.devices = devices
        self.prep = Preprocess(is_training)
        self.is_training = is_training

    def forward(self, batch_dict):
        self.prep.re_random()
        if self.is_training:
            batch_dict = self.prep(batch_dict, is_training=True)
        else:
            batch_dict = self.prep(batch_dict, is_training=False)
        return batch_dict

class AssignTarget(nn.Module):
    def __init__(self, devices):
        super(AssignTarget, self).__init__()
        self.rows = configs.bird.rows
        self.cols = configs.bird.cols

        self.extents = torch.Tensor(np.array(configs.bird.extents).astype(np.float)).float().cuda()
        self.devices = devices

    def forward(self, batch_dict):
        input_gt_boxes = batch_dict['gt_boxes_torch']
        pixel_target_map, offset_target_map, offset_weight_map, velocity_target_map, size_target_map, \
            yaw_target_map, height_target_map, category_target_map_idx, count_pixels_in_bboxes =\
            roiaware_pool3d_utils.build_pixel_offset_velocity_target(
                self.rows, self.cols, self.extents, input_gt_boxes, self.devices)
        batch_dict.update({
            'pixel_target_map': pixel_target_map,
            'offset_target_map': offset_target_map,
            'velocity_target_map': velocity_target_map,
            'size_target_map': size_target_map,
            'yaw_target_map': yaw_target_map,
            'height_target_map': height_target_map,
            'category_target_map': category_target_map_idx,
            'offset_weight_map': offset_weight_map,
        })
        return batch_dict


class OffsetThree(nn.Module):
    def __init__(self, num_feature_channel, batch_size=1, device=None, is_training=True, dist_test=False):
        super(OffsetThree, self).__init__()
        self.unet = UNet(n_channels=num_feature_channel)
        self.confidence_classify = ConfidenceClassification(category_num=1)
        self.pixel_classify = pixelClassification(category_num=1)
        self.offsets_estimate = OffsetsEstimation(offset_dimention=2)
        self.velocity_estimate = VelocityEstimation(velocity_dimention=2)

        self.size_estimate = SizeEstimation(size_dimention=2)
        self.yaw_estimate = YawEstimation(yaw_dimention=2)
        self.height_estimate = HeightEstimation(height_dimention=2)
        self.category_estimate = CategoryEstimation(category_dim=11)
        self.dist_test = dist_test
        self.filter_name = configs.tracker.TRACKING_NAMES_INT

        if device is None:
            print("Device should not be None. ")
        self.device = device
        self.is_training = is_training
        self.rtree2 = rtree_utils.RTree(rows=configs.bird.rows, cols=configs.bird.cols, devices=device,
                                        isTrain=is_training, batch_size=batch_size, numChannel=7)
        #  for save forward dict result
        self.l1_loss_fun = nn.L1Loss(reduction='none')
        self.smooth_l1_fun = nn.SmoothL1Loss(reduction='none')
        self.extents = configs.bird.extents
        self.rows = configs.bird.rows
        self.cols = configs.bird.cols
        self.grid_size_row = (self.extents[0][1] - self.extents[0][0]) / self.rows
        self.grid_size_col = (self.extents[1][1] - self.extents[1][0]) / self.cols

        self.forward_ret_dict = {}

        # if self.is_training:
        self.data_preprocess = DataPreprocess(device, is_training)

        self.vfe = VFE(device)

        if self.is_training:
            self.assign_target = AssignTarget(device)

    def forward(self, data_dict):
        # torch.cuda.synchronize()
        time_0 = time.time()

        # if self.is_training:
        data_dict = self.data_preprocess(data_dict)
        data_dict = self.vfe(data_dict)
        if self.is_training:
            data_dict = self.assign_target(data_dict)
        bevs = data_dict['voxel_feature']

        bevs = bevs.permute(0, 3, 1, 2)  # Batch, feature_channel, rows, cols
        # torch.cuda.synchronize()
        # print('[TIME] permute cost: {}'.format((time.time() - time_0) * 1000))
        # Backbone network
        x = self.unet(bevs)
        # torch.cuda.synchronize()
        # print('[TIME] unet cost: {}'.format((time.time() - time_0) * 1000))
        #  pixel_confidence_pred
        pixel_pred = self.pixel_classify(x)

        #  For offset Estimate head
        offset_pred = self.offsets_estimate(x)

        #  For confidence head
        confidence_pred = self.confidence_classify(x)

        #  For Velocity head
        velocity_pred = self.velocity_estimate(x)

        # For Box Size pred
        size_pred = self.size_estimate(x)

        #  For Yaw Pred
        yaw_pred = self.yaw_estimate(x)

        #  For height pred
        height_pred = self.height_estimate(x)

        #  For category pred
        category_pred = self.category_estimate(x)

        # torch.cuda.synchronize()
        # print('[TIME] other head cost: {}'.format((time.time() - time_0) * 1000))
        #  get pixel_weight and offset weight, when training propose, when eval bbox
        if self.is_training:
            with torch.no_grad():
                pixel_gt = data_dict['pixel_target_map']
                voxel_count_gt = data_dict['voxel_count_map']
                view_index = data_dict['view_index_map']
                if not configs.train.without_loss:
                    # torch.cuda.synchronize()
                    time_0 = time.time()
                    pixel_weight, offset_weight, confidence_weight, velocity_weight, boxes_result_map \
                        = self.rtree2(voxel_count_gt, pixel_pred, confidence_pred, offset_pred, velocity_pred,
                                      view_index, pixel_gt)
                    # torch.cuda.synchronize()
                    # print('[TIME] rtree has cost: {}'.format((time.time() - time_0) * 1000))

                    self.forward_ret_dict.update({'pixel_pred': pixel_pred,
                                                  'confidence_pred': confidence_pred,
                                                  'offset_pred': offset_pred,
                                                  'velocity_pred': velocity_pred,
                                                  'pixel_weight': pixel_weight,
                                                  'offset_weight': offset_weight,
                                                  'confidence_weight': confidence_weight,
                                                  'velocity_weight': velocity_weight,
                                                  'size_pred': size_pred,
                                                  'yaw_pred': yaw_pred,
                                                  'height_pred': height_pred,
                                                  'category_pred': category_pred,
                                                  'pixel_target_map': pixel_gt,
                                                  'voxel_count_gt': voxel_count_gt,
                                                  'view_index': view_index,
                                                  'offset_target_map': data_dict['offset_target_map'],
                                                  'velocity_target_map': data_dict['velocity_target_map']})
                else:
                    boxes_result_map = self.rtree2(voxel_count_gt, pixel_pred, confidence_pred, offset_pred)
                    self.forward_ret_dict.update({'pixel_pred': pixel_pred,
                                                  'confidence_pred': confidence_pred,
                                                  'offset_pred': offset_pred,
                                                  'velocity_pred': velocity_pred,
                                                  'size_pred': size_pred,
                                                  'yaw_pred': yaw_pred,
                                                  'height_pred': height_pred,
                                                  'category_pred': category_pred,
                                                  'boxes_result_map': boxes_result_map,
                                                  'category_pred': category_pred})
            if not configs.train.without_loss:
                # self.save_bin_file_for_debug(data_dict, self.forward_ret_dict)
                loss, detect_loss_dict = self.get_loss(data_dict)
                return loss, detect_loss_dict, self.forward_ret_dict
            if configs.train.gen_bbox and configs.train.without_loss:
                boxes_result_map = self.rtree2(voxel_count_gt, pixel_pred, confidence_pred, offset_pred)
                bboxes, box_center_in_pixel = self.gen_batch_pred_bbox()
                self.forward_ret_dict.update({'bboxes': bboxes,
                                              'box_center_in_pixel': box_center_in_pixel})
                return self.forward_ret_dict
        else:
            view_index = data_dict['view_index_map']
            voxel_count_gt = data_dict['voxel_count_map']
            boxes_result_map = self.rtree2(voxel_count_gt, pixel_pred, confidence_pred,
                                           offset_pred, view_index, velocity_pred)
            self.forward_ret_dict.update({'pixel_pred': pixel_pred,
                                          'confidence_pred': confidence_pred,
                                          'offset_pred': offset_pred,
                                          'velocity_pred': velocity_pred,
                                          'size_pred': size_pred,
                                          'yaw_pred': yaw_pred,
                                          'height_pred': height_pred,
                                          'boxes_result_map': boxes_result_map,
                                          'category_pred': category_pred,
                                          'voxel_count_gt': voxel_count_gt})

            if not configs.train.without_loss:
                bboxes, box_center_in_pixel, box_scores = self.gen_pred_bbox()
            else:
                bboxes, box_center_in_pixel = self.gen_batch_pred_bbox()
            self.forward_ret_dict.update({'tokens': data_dict['token'],
                                          'gt_boxes_torch': data_dict['gt_boxes_torch']})
            self.forward_ret_dict.update({'box_center_in_pixel': box_center_in_pixel})
            self.forward_ret_dict.update({"batch_det_feature_map": x,
                                          'bboxes': bboxes,
                                          'box_scores': box_scores,
                                          'gt_boxes_orig': data_dict['gt_boxes_orig']})
            if self.dist_test:
                self.post_processing()

            # return self.forward_ret_dict, data_dict
            return self.forward_ret_dict

    @torch.no_grad()
    def post_processing(self):
        """
        @brief for nuscenes eval
        @return:
        """
        batch_size = len(self.forward_ret_dict['bboxes'])
        batch_box_pred = self.forward_ret_dict['bboxes']
        batch_bbox_scores = self.forward_ret_dict['box_scores']
        batch_token = self.forward_ret_dict['tokens']
        prediction_dicts = []
        for i in range(batch_size):
            box_preds = batch_box_pred[i]
            box_scores = batch_bbox_scores[i]
            box_labels = box_preds[:, -1].long()
            box_ignore_mask = torch.ones_like(box_labels, dtype=torch.long)
            for idx, elem in enumerate(box_labels):
                new_class = Int2Eval[elem.item()]
                box_labels[idx] = new_class
                if new_class < 0:  # ignore
                    box_ignore_mask[idx] = 0
                    continue
            box_ignore_mask_idx = torch.where(box_ignore_mask > 0)
            prediction_dict = {
                'box3d_lidar': box_preds[box_ignore_mask_idx],
                'scores': box_scores[box_ignore_mask_idx],
                'box_preds': box_labels[box_ignore_mask_idx],
                'token': batch_token[i]
            }
            prediction_dicts.append(prediction_dict)

        self.forward_ret_dict.update({"output": prediction_dicts})

    @staticmethod
    def save_bin_file_for_debug(data_dict, forward_dict):
        """
        :brief: save some bin file for debug
        @param forward_dict: all result
        @param data_dict: all data dict
        @return: None
        """
        data_idx = 0
        confidence_pred = forward_dict['confidence_pred']
        voxel_count_gt = data_dict['voxel_count_map']
        view_index_map = data_dict['view_index_map']
        pixel_pred = forward_dict['pixel_pred']
        offset_pred = forward_dict['offset_pred']
        pixel_gt = data_dict['pixel_target_map']
        offset_target = data_dict['offset_target_map']
        offset_weight = data_dict['offset_weight_map']
        # voxel_count_gt, pixel_pred, offset_pred, pixel_gt
        confidence_pred.detach().cpu().contiguous().numpy().tofile(
            "/media/user/C14D581BDA18EBFA/code/"
            "simpleViewer/binFile/confidence_pred_{}.bin".format(
                data_idx))
        voxel_count_gt.detach().cpu().contiguous().numpy().tofile(
            "/media/user/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "voxel_count_gt_{}.bin".format(data_idx))
        pixel_pred.detach().cpu().contiguous().numpy().tofile(
            "/media/user/C14D581BDA18EBFA/code/simpleViewer/"
            "binFile/pixel_pred_{}.bin".format(data_idx))
        pixel_gt.detach().cpu().contiguous().numpy().tofile(
            "/media/user/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "pixel_gt_{}.bin".format(data_idx))
        offset_pred.detach().cpu().contiguous().numpy().tofile(
            "/media/user/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "offset_pred_{}.bin".format(data_idx))
        offset_target.detach().cpu().contiguous().numpy().tofile(
            "/media/user/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "offset_target_{}.bin".format(data_idx))
        offset_weight.detach().cpu().contiguous().numpy().tofile(
            "/media/user/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "offset_weight_{}.bin".format(data_idx))
        view_index_map.detach().cpu().contiguous().numpy().tofile(
            "/media/user/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "view_index_{}.bin".format(data_idx))
        offset_pred.detach().cpu().contiguous().numpy().tofile(
            "/media/user/C14D581BDA18EBFA/code/simpleViewer/binFile/"
            "offset_pred_{}.bin".format(data_idx))
        debug = 1

    def gen_batch_pred_bbox(self):
        """
            # gen batch bbox for train with tracking
        """
        boxes_result_map = self.forward_ret_dict['boxes_result_map']
        size_pred = self.forward_ret_dict['size_pred']
        yaw_pred = self.forward_ret_dict['yaw_pred']
        height_pred = self.forward_ret_dict['height_pred']
        offset_pred = self.forward_ret_dict['offset_pred']
        velocity_pred = self.forward_ret_dict['velocity_pred']
        category_pred = self.forward_ret_dict['category_pred']
        offset_weight = self.forward_ret_dict['offset_weight_map']

        batch_size = size_pred.shape[0]
        batch_boxes = []
        box_center_in_pixel = []
        for bs in range(batch_size):
            pos_x, pos_y = torch.where(boxes_result_map[bs] > 0)
            box_size = size_pred[bs, :, pos_x, pos_y]  # B, 2, Num
            box_yaw_sin_cos = yaw_pred[bs, :, pos_x, pos_y]
            box_height = height_pred[bs, :, pos_x, pos_y]
            box_offset = offset_pred[bs, :, pos_x, pos_y]
            box_velocity = velocity_pred[bs, :, pos_x, pos_y]
            pos_x_real = (pos_x + 0.5) * self.grid_size_row - self.extents[0][1]
            pos_y_real = (pos_y + 0.5) * self.grid_size_col - self.extents[1][1]
            box_center_x = pos_x_real - box_offset[0, :]  # make sure here, pay more attention
            box_center_y = pos_y_real - box_offset[1, :]
            box_center_z = (box_height[0, :] + box_height[1, :]) / 2.0
            box_size_z = box_height[1, :] - box_height[0, :]
            box_category = category_pred[bs, :, pos_x, pos_y]
            if box_category.shape[1] > 0:
                box_category_num = torch.argmax(box_category, dim=0)
            else:
                box_category_num = torch.tensor([], device=box_category.device, dtype=torch.float)
            num_bboxes = len(pos_x)
            num_box_elem = 10
            box_yaw = torch.atan2(box_yaw_sin_cos[0, :], box_yaw_sin_cos[1, :]) * 0.5
            bboxes_tensor = torch.zeros(size=(num_bboxes, num_box_elem), dtype=np.float, device=self.device)
            bboxes_tensor[:, :] = torch.stack([box_center_x, box_center_y, box_center_z, box_size[0, :],
                                               box_size[1, :], box_size_z, box_yaw, box_velocity[0, :],
                                               box_velocity[1, :], box_category_num], axis=0).transpose(0, 1)
            box_center_in_pixel.append([pos_x, pos_y])
            batch_boxes.append(bboxes_tensor)
        return batch_boxes, box_center_in_pixel

    def gen_pred_bbox(self):   # need make sure
        """
            Param: gen bbox model train
            Return: bbox
        """
        boxes_result_map = self.forward_ret_dict['boxes_result_map']
        size_pred = self.forward_ret_dict['size_pred']
        yaw_pred = self.forward_ret_dict['yaw_pred']
        height_pred = self.forward_ret_dict['height_pred']
        offset_pred = self.forward_ret_dict['offset_pred']
        velocity_pred = self.forward_ret_dict['velocity_pred']
        category_pred = self.forward_ret_dict['category_pred']
        confidence_pred = self.forward_ret_dict['confidence_pred']

        weighted_confidence_pred = boxes_result_map[:, :, :, 1]
        weighted_velocity_pred = boxes_result_map[:, :, :, 2:4].permute([0, 3, 1, 2]).contiguous()

        condition, pos_x, pos_y = torch.where(weighted_confidence_pred > 0.6)
        # condition, pos_x, pos_y = torch.where(boxes_result_map > 0)
        box_size = size_pred[condition, :, pos_x, pos_y]  # B, 2, Num
        box_yaw_sin_cos = yaw_pred[condition, :, pos_x, pos_y]
        box_height = height_pred[condition, :, pos_x, pos_y]
        box_offset = offset_pred[condition, :, pos_x, pos_y]
        # box_velocity = velocity_pred[condition, :, pos_x, pos_y]
        box_velocity = weighted_velocity_pred[condition, :, pos_x, pos_y]
        box_category = category_pred[condition, :, pos_x, pos_y]
        box_scores = confidence_pred[condition, :, pos_x, pos_y]
        pos_x_real = (pos_x + 0.5) * self.grid_size_row - self.extents[0][1]
        pos_y_real = (pos_y + 0.5) * self.grid_size_col - self.extents[1][1]
        box_center_x = pos_x_real - box_offset[:, 0]  # make sure here, pay more attention
        box_center_y = pos_y_real - box_offset[:, 1]
        box_center_z = (box_height[:, 0] + box_height[:, 1]) / 2.0
        box_size_z = box_height[:, 1] - box_height[:, 0]
        box_yaw = torch.atan2(box_yaw_sin_cos[:, 0], box_yaw_sin_cos[:, 1]) * 0.5

        if box_category.shape[0] > 0:
            box_category_num = torch.argmax(box_category, dim=1)
        else:
            box_category_num = torch.tensor([], device=box_category.device, dtype=torch.float)

        boxes_tensor_list = []
        box_center_in_pixel = []
        box_scores_list = []
        batch_size = boxes_result_map.shape[0]
        for idx in range(batch_size):
            curr_idx = ((condition == idx) * ((box_scores > 0.6)[:, 0]))

            box_center_in_pixel.append([pos_x[curr_idx], pos_y[curr_idx]])
            # boxes_t = torch.zeros(size=(curr_idx.sum(), num_box_elem), dtype=np.float, device=self.device)
            boxes_t = torch.stack([box_center_x[curr_idx], box_center_y[curr_idx], box_center_z[curr_idx],
                                 box_size[curr_idx, 0], box_size[curr_idx, 1], box_size_z[curr_idx],
                                 box_yaw[curr_idx], box_velocity[curr_idx, 0], box_velocity[curr_idx, 1],
                                 box_category_num[curr_idx].float()]).transpose(0, 1)
            box_scores_list.append(box_scores[curr_idx])
            boxes_tensor_list.append(boxes_t)
        return boxes_tensor_list, box_center_in_pixel, box_scores_list

    def get_loss(self, data_dict):
        """
            computer loss for detect model
        """
        pixel_gt = data_dict['pixel_target_map']
        offset_gt = data_dict['offset_target_map']
        velocity_gt = data_dict['velocity_target_map']
        voxel_count_gt = data_dict['voxel_count_map']
        size_gt = data_dict['size_target_map']
        yaw_gt = data_dict['yaw_target_map']
        height_gt = data_dict['height_target_map']
        category_gt = data_dict['category_target_map']
        view_index = data_dict['view_index_map']
        offset_weight = data_dict['offset_weight_map']

        pixel_pred = self.forward_ret_dict['pixel_pred']
        confidence_pred = self.forward_ret_dict['confidence_pred']
        velocity_pred = self.forward_ret_dict['velocity_pred']
        offset_pred = self.forward_ret_dict['offset_pred']
        size_pred = self.forward_ret_dict['size_pred']
        yaw_pred = self.forward_ret_dict['yaw_pred']
        height_pred = self.forward_ret_dict['height_pred']
        category_pred = self.forward_ret_dict['category_pred']

        pixel_weight = self.forward_ret_dict['pixel_weight']
        confidence_weight = self.forward_ret_dict['confidence_weight']
        velocity_weight = self.forward_ret_dict['velocity_weight']

        batch_size = pixel_gt.shape[0]
        # criterion_pixel = torch.nn.BCEWithLogitsLoss(weight=pixel_weight.detach(), reduction='sum')
        criterion_pixel = torch.nn.BCELoss(weight=pixel_weight.detach(), reduction='sum')
        pixel_pred = pixel_pred.view(-1, configs.bird.rows, configs.bird.cols).contiguous()
        loss_pixel = criterion_pixel(pixel_pred, pixel_gt) / batch_size \
                        * configs.train.det_stage1_weight.pixel

        confidence_pred = confidence_pred.view(-1, configs.bird.rows, configs.bird.cols).contiguous()
        criterion_confidence = torch.nn.BCELoss(weight=confidence_weight.detach(), reduction='sum')
        loss_confidence = criterion_confidence(confidence_pred, pixel_gt) / batch_size \
                          * configs.train.det_stage1_weight.confidence

        # todo we need a common weight objects for other labeled relative weight, we also need data augmentation
        objectness_weight_map = view_index[:, :, :, 5].contiguous()
        # weight objects can be get from build_pixel_offset_velocity_target_kernel, all pixel / num_pixel pre bbox
        # also can calculate offset weight, if can do this
        weight_objects = objectness_weight_map.unsqueeze(1).repeat(1, 2, 1, 1)
        resolution = configs.bird.resolution
        weight_pixel_gt = offset_weight / resolution
        # num_valid_pixel = torch.nonzero(pixel_gt).size(0) + 1  # add 1 for nonzero pixel in boxes, get nan
        loss_offset = torch.sum(self.smooth_l1_fun(offset_gt, offset_pred) * offset_weight) / batch_size \
                      * configs.train.det_stage1_weight.offset
        # todo error, using gt objectness
        # velocity_weight_new = velocity_weight.unsqueeze(1).repeat(1, 2, 1, 1).contiguous()
        velocity_weight_new = weight_objects  # need replace by has_velocity map weight
        loss_velocity = torch.sum(self.smooth_l1_fun(velocity_gt, velocity_pred) * velocity_weight_new) / batch_size \
                        * configs.train.det_stage1_weight.velocity
        #  loss box relative
        loss_size = torch.sum(self.l1_loss_fun(size_pred, size_gt) * weight_pixel_gt) / batch_size \
                    * configs.train.det_stage1_weight.size
        loss_yaw = torch.sum(torch.norm(yaw_gt - yaw_pred, p=2, dim=1, keepdim=True) * weight_pixel_gt) / batch_size\
                   * configs.train.det_stage1_weight.yaw
        loss_height = torch.sum(self.l1_loss_fun(height_pred, height_gt) * weight_pixel_gt) / batch_size \
                      * configs.train.det_stage1_weight.height

        weight_category = objectness_weight_map.unsqueeze(1).repeat(1, 11, 1, 1) \
                          * configs.train.det_stage1_weight.category
        category_loss_fun = nn.BCELoss(weight=weight_category, reduction='sum')
        category_gt = category_gt.permute(0, 3, 1, 2)
        loss_category = torch.sum(category_loss_fun(category_pred, category_gt)) / batch_size
        loss = loss_pixel + loss_offset + loss_confidence + loss_velocity + loss_size + \
               loss_yaw + loss_height + loss_category
        detect_loss_dict = {
                'loss_pixel': loss_pixel.item(),
                'loss_offset': loss_offset.item(),
                'loss_confidence': loss_confidence.item(),
                'loss_velocity': loss_velocity.item(),
                'loss_size': loss_size.item(),
                'loss_yaw': loss_yaw.item(),
                'loss_height': loss_height.item(),
                'loss_category': loss_category.item()
        }
        return loss, detect_loss_dict



