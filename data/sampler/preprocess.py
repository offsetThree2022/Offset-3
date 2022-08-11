import numpy as np
import configs as cfg
import data.sampler.sampler_preprocess as prep
from utils.data_utils import convert_pickle_boxes_to_torch_box

class Preprocess(object):
    """
        @brief: one seq data should use one random
    """
    def __init__(self, is_training=True):
        self.mode = is_training
        self.shuffle_points = cfg.data.train_preprocessor.shuffle_points
        self.no_augmentation = cfg.data.train_preprocessor.no_augmentation
        self.probability = 0.5
        self.x_enable_for_flip_both = np.random.choice(
            [False, True], replace=False, p=[1 - self.probability, self.probability])
        self.y_enable_for_flip_both = np.random.choice(
            [False, True], replace=False, p=[1 - self.probability, self.probability])
        self.global_rot_noise = cfg.data.train_preprocessor.global_rot_noise
        self.global_scale_noise = cfg.data.train_preprocessor.global_scale_noise
        self.global_translate_std = cfg.data.train_preprocessor.global_translate_std
        if not isinstance(self.global_rot_noise, (list, tuple, np.ndarray)):
            self.rotation = [-self.global_rot_noise, self.global_rot_noise]
        else:
            self.rotation = self.global_rot_noise[0]
        self.noise_rotation = np.random.uniform(self.rotation[0], self.rotation[1])
        self.scale_factor = cfg.data.train_preprocessor.global_scale_noise
        self.noise_scale = np.random.uniform(self.scale_factor[0], self.scale_factor[1])

        if not isinstance(self.global_translate_std, (list, tuple, np.ndarray)):
            self.noise_translate_std = np.array(
                [self.global_translate_std, self.global_translate_std, self.global_translate_std]
            )
        self.noise_translate = np.array(
            [
                np.random.normal(0, self.noise_translate_std[0], 1),
                np.random.normal(0, self.noise_translate_std[1], 1),
                np.random.normal(0, self.noise_translate_std[0], 1),
            ]
        ).T

    def re_random(self):
        """
        @return:
        """
        self.x_enable_for_flip_both = np.random.choice(
            [False, True], replace=False, p=[1 - self.probability, self.probability])
        self.y_enable_for_flip_both = np.random.choice(
            [False, True], replace=False, p=[1 - self.probability, self.probability])
        self.global_rot_noise = cfg.data.train_preprocessor.global_rot_noise
        self.global_scale_noise = cfg.data.train_preprocessor.global_scale_noise
        self.global_translate_std = cfg.data.train_preprocessor.global_translate_std
        if not isinstance(self.global_rot_noise, (list, tuple, np.ndarray)):
            self.rotation = [-self.global_rot_noise, self.global_rot_noise]
        else:
            self.rotation = self.global_rot_noise[0]
        self.noise_rotation = np.random.uniform(self.rotation[0], self.rotation[1])
        self.scale_factor = cfg.data.train_preprocessor.global_scale_noise
        self.noise_scale = np.random.uniform(self.scale_factor[0], self.scale_factor[1])

        if not isinstance(self.global_translate_std, (list, tuple, np.ndarray)):
            self.noise_translate_std = np.array(
                [self.global_translate_std, self.global_translate_std, self.global_translate_std]
            )
        self.noise_translate = np.array(
            [
                np.random.normal(0, self.noise_translate_std[0], 1),
                np.random.normal(0, self.noise_translate_std[1], 1),
                np.random.normal(0, self.noise_translate_std[0], 1),
            ]
        ).T

    def __call__(self, res, is_training=False):
        """
        @param res: data_dict, only train need db, val do not need
        @return: None
        """
        res["mode"] = self.mode
        if self.no_augmentation[0]:  # not need augmentation
            return
        batch_size = res['points'].__len__()
        for bs_idx in range(batch_size):
            input_gt_bboxes = convert_pickle_boxes_to_torch_box(res["gt_boxes_orig"][bs_idx])
            if not is_training:
                res["gt_boxes_orig"][bs_idx] = input_gt_bboxes
                continue
            #  flip x y
            res["gt_boxes_orig"][bs_idx], res['points'][bs_idx], res['pc'][bs_idx] = \
                prep.random_flip_both(input_gt_bboxes,
                                      res["points"][bs_idx],
                                      res["pc"][bs_idx],
                                      self.x_enable_for_flip_both,
                                      self.y_enable_for_flip_both)
            #  global rotation
            res["gt_boxes_orig"][bs_idx], res['points'][bs_idx], res['pc'][bs_idx] = \
                prep.global_rotation(res["gt_boxes_orig"][bs_idx],
                                      res["points"][bs_idx],
                                      res["pc"][bs_idx],
                                      self.noise_rotation)
            #
            #
            res["gt_boxes_orig"][bs_idx], res['points'][bs_idx], res['pc'][bs_idx] = \
                prep.global_scaling_v2(res["gt_boxes_orig"][bs_idx],
                                      res["points"][bs_idx],
                                      res["pc"][bs_idx],
                                      self.noise_scale)
            #
            res["gt_boxes_orig"][bs_idx], res['points'][bs_idx], res['pc'][bs_idx] = \
                prep.global_translate_(res["gt_boxes_orig"][bs_idx],
                                       res["points"][bs_idx],
                                       res["pc"][bs_idx],
                                       self.noise_translate)
        return res



