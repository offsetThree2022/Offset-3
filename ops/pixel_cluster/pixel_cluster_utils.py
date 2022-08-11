import torch
import torch.nn as nn
from torch.autograd import Function

from ops.pixel_cluster import pixel_cluster_cpu

class PixelCluster(nn.Module):
    def __init__(self, isTrain=False):
        super(PixelCluster, self).__init__()
        self.Train = isTrain

    def forward(self, non_zeros_map, pixel_pred, offset_pred, pixel_gt):
        """
        Args:
            non_zeros_map: (B, row, col)
            pixel_pred: (B, row, col)
            offset_pred: (B, 2, row, col)
            pixel_gt: (B, row, col)

            Train ==> true
        Return:
            pixel_weight: (B, row, col)
            offset_weight: (B, 2, row, col)
            pixel_gt: (B, row, col)
        not Train Return:
            object: (B, num_instance)
        """
        return PixelClusterFunction.apply(self.Train, non_zeros_map, pixel_pred, offset_pred, pixel_gt)


class PixelClusterFunction(Function):
    @staticmethod
    def forward(ctx, is_training, non_zeros_maps, pixel_pred, offset_pred, pixel_gt):
        assert 2 == offset_pred.shape[1], "offset second channel it not 2"
        batch_size, rows, cols = non_zeros_maps.shape[0], non_zeros_maps.shape[1], non_zeros_maps.shape[2]
        pixel_weight = pixel_pred.new_zeros((batch_size, rows, cols))
        offset_weight = pixel_pred.new_zeros((batch_size, 2, rows, cols))

        pixel_cluster_cpu.forward(is_training, non_zeros_maps, pixel_pred, offset_pred, pixel_gt, pixel_weight,
                                  offset_weight)
        return pixel_weight, offset_weight

    @staticmethod
    def backward(ctx, grad_out):
        raise NotImplemented

if __name__ == '__main__':
    pass