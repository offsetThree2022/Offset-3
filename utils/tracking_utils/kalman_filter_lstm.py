"""
kalman_filter_lstm.py as
"""
import numpy as np
import scipy.linalg
from model import OffsetThree
import torch
import torch.nn as nn
import os
import configs


class DecoderRNN(torch.nn.Module):
    def __init__(self, num_hidden):
        super(DecoderRNN, self).__init__()
        self.num_hidden = num_hidden
        self.lstm = nn.LSTM(18, self.num_hidden)
        self.out1 = nn.Linear(self.num_hidden, 64)
        self.out2 = nn.Linear(64, 4 * 4)

    def forward(self, input_tarj):
        """
            Fully connected for predict tarj
        """
        input_tarj = input_tarj.permute(1, 0, 2)

        output, (hn, cn) = self.lstm(input_tarj)
        x = self.out1(output[-1])
        x = self.out2(x)
        return x


class KalmanFilterLSTM(object):
    """
        A simple Kalman filter for tracking bounding boxes in image space.
    """
    def __init__(self):
        self.model = DecoderRNN(128)
        if configs.traj.load_model_tarj is not None:
            pass  # need model
        self.MAX_dist_fut = 4

    def predict(self, h0, c0, new_features):

        new_features = new_features.permuter(1, 0, 2)
        output, (hn, cn) = self.model.lstm(new_features, (h0, c0))
        x = self.model.out1(output[-1])
        x = self.model.out2(x)

        x = x.view(self.MAX_dist_fut, -1).cpu().detach().numpy()
        prediction = {}
        for i in range(self.MAX_dist_fut):
            prediction[1 + i] = x[i]

    def gating_distance(self, mean, covariance, measurements, only_position = False, metric = "maha"):
        if only_position:
            mean, covaritance = mean[:2], covariance[:2, :2]
            measurements = measurements[:, :2]
        d = measurements - mean
        if metric == "gaussian":
            d = measurements[:, 3:-1] - mean[3:-1]
            return np.sqrt(np.sum(d * d, axis=1))
        elif metric == "maha":
            cholesky_factor = np.linalg.cholesky(covariance)
            z = scipy.linalg.solve_triangular(
                cholesky_factor, d.T, lower=True, check_finite=False, overwrite_b=True
            )
            squared_maha = np.sum(z * z, axis=0)
            return squared_maha
        else:
            raise ValueError("invalid distance metric")