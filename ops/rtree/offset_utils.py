import torch
import torch.nn as nn
from torch.autograd import Function

import configs
from . import rtree_cuda


class RTree(nn.Module):
    def __init__(self, rows, cols, devices, isTrain=False, batch_size=1, numChannel=1, pixel_threshold=0.05):
        super(RTree, self).__init__()
        self.Train = isTrain
        self.batch_size = batch_size
        self.numChannel = numChannel
        self.pixel_threshold = pixel_threshold
        self.fixedMem = torch.zeros(size=(self.batch_size, self.numChannel, rows, cols), device=devices,
                                    dtype=torch.int, requires_grad=False)
        self.fixedMem_float = torch.zeros(size=(self.batch_size, 2, rows, cols), device=devices,
                                          dtype=torch.float, requires_grad=False)


    def get_fixed_mem(self):
        return self.fixedMem

    def forward(self, voxel_count_gt, pixel_pred, confidence_pred,
                offset_pred, view_index, velocity_pred, pixel_gt=None):
        """
        Args:
            voxel_count_gt: (B, row, col)
            pixel_pred: (B, row, col)
            confidence_pred: (B, row, col)
            offset_pred: (B, 2, row, col)
            velocity_pred: (B, 2, row, col)
            pixel_gt: (B, row, col)
            view_index: (B, row, col, 5)

            Train ==> true
        Return:
            pixel_weight: (B, row, col)
            offset_weight: (B, 2, row, col)
        not Train Return:
            object: (B, num_instance)
        """
        return RTreeFunction.apply(self.Train, voxel_count_gt, pixel_pred, confidence_pred, offset_pred,
                                   pixel_gt, velocity_pred, self.fixedMem, self.numChannel,
                                   self.fixedMem_float, self.pixel_threshold, view_index)


class RTreeFunction(Function):
    @staticmethod
    def forward(ctx, is_training, voxel_count_gt, pixel_pred,
                confidence_pred, offset_pred, pixel_gt,
                velocity_pred, fixedMem, numChannel, fixedMem_float,
                pixel_threshold, view_index):
        """
            fixedMem: some mem Frequent application and frequent release in cuda, we need keep it fixed
                  it can used to record tmp var, such as traversed map, lock_map, ect
                  we need use memset to zero it, before we use it
        """
        num_view_index_channel = view_index.shape[-1]
        num_boxes_result_map_channel = 4

        if is_training is False:
            assert pixel_gt is None
        assert 2 == offset_pred.shape[1], "offset second channel it not 2"

        batch_size, rows, cols = voxel_count_gt.shape[0], voxel_count_gt.shape[1], voxel_count_gt.shape[2]
        extents = configs.bird.extents
        grid_size_row = (extents[0][1] - extents[0][0]) / rows
        grid_size_col = (extents[1][1] - extents[1][0]) / cols
        voxel_count_gt = voxel_count_gt.contiguous()
        boxes_result_map = pixel_pred.new_zeros((batch_size, rows, cols,
                                                    num_boxes_result_map_channel)).fill_(-0.1)
        if is_training:
            pixel_gt = pixel_gt.contiguous()
            pixel_weight = pixel_pred.new_zeros((batch_size, rows, cols))
            offset_weight = pixel_pred.new_zeros((batch_size, 2, rows, cols))
            confidence_weight = pixel_pred.new_zeros((batch_size, rows, cols))
            velocity_weight = pixel_pred.new_zeros((batch_size, rows, cols))
            rtree_cuda.forward(is_training,
                               rows,
                               cols,
                               grid_size_row,
                               grid_size_col,
                               pixel_threshold,
                               pixel_pred,
                               confidence_pred,
                               voxel_count_gt,
                               offset_pred,
                               pixel_gt,
                               view_index,
                               num_view_index_channel,
                               numChannel,
                               fixedMem,
                               fixedMem_float,
                               pixel_weight,
                               confidence_weight,
                               velocity_weight,
                               offset_weight,
                               boxes_result_map)
            return pixel_weight, offset_weight, confidence_weight, velocity_weight, boxes_result_map
        else:
            rtree_cuda.forward_val(is_training,
                                   rows,
                                   cols,
                                   grid_size_row,
                                   grid_size_col,
                                   pixel_threshold,
                                   pixel_pred,
                                   confidence_pred,
                                   voxel_count_gt,
                                   offset_pred,
                                   velocity_pred,
                                   num_view_index_channel,
                                   view_index,
                                   numChannel,
                                   fixedMem,
                                   num_boxes_result_map_channel,
                                   boxes_result_map)
            return boxes_result_map

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplemented


if __name__ == '__main__':
    pass