import numpy as np
from pyquaternion import Quaternion
import configs
from data.nuscenes_base import NameMapping, Name2Int

def transform_matrix(translation: np.ndarray = np.array([0, 0, 0]),
                     rotation: Quaternion = Quaternion([1, 0, 0, 0]),
                     inverse: bool = False) -> np.ndarray:
    """
    Convert pose to transformation matrix.
    :param translation: <np.float32: 3>. Translation in x, y, z.
    :param rotation: Rotation in quaternions (w ri rj rk).
    :param inverse: Whether to compute inverse transform matrix.
    :return: <np.float32: 4, 4>. Transformation matrix.
    """
    tm = np.eye(4)

    if inverse:
        rot_inv = rotation.rotation_matrix.T
        trans = np.transpose(-np.array(translation))
        tm[:3, :3] = rot_inv
        tm[:3, 3] = rot_inv.dot(trans)
    else:
        tm[:3, :3] = rotation.rotation_matrix
        tm[:3, 3] = np.transpose(np.array(translation))

    return tm

def convert_pickle_boxes_to_torch_box(curr_boxes_gt):
    """
        args:
            convert boxes which read from pickle to cuda format
    """
    # extents = np.array(configs.bird.extents)
    curr_boxes = np.zeros((len(curr_boxes_gt), 10), np.float32)
    for i, elem in enumerate(curr_boxes_gt):
        whl = elem.wlh
        velocity = elem.velocity[:2]
        if np.isnan(velocity).any():
            velocity = [0.0, 0.0]
        curr_boxes[i, :] = ([*elem.center, whl[1], whl[0], whl[2], elem.orientation.yaw_pitch_roll[0],
                            velocity[0], velocity[1], Name2Int[NameMapping[elem.name]]])
    return curr_boxes