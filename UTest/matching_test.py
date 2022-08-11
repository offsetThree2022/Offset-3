import unittest

import numpy as np
from utils import matching


class MyTestCase(unittest.TestCase):
    def test_linear_assignment(self):
        loss_inf = 10000
        cost_mat = np.array(
            [[0.6, 0.01, 0.7, 0.5, 0.2, loss_inf, loss_inf, loss_inf],
             [0.3, 0.7, 0.02, 0.9, loss_inf, 0.3, loss_inf, loss_inf],
             [0.02, 0.8, 0.9, 0.8, loss_inf, loss_inf, 0.1, loss_inf],
             [0.7, 0.9, 0.6, 0.32, loss_inf, loss_inf, loss_inf, 0.1]]
        )

        matches, u_track, u_detection = matching.linear_assignment(cost_matrix=cost_mat, thresh=0.999)
        print("matchs: ", matches)
        print("u_tack:", u_track)
        print("u_detection", u_detection)

    def test_transformtion(self):
        """
        @return:
        """
        pass
        # transform_matrix, lastSweepBox.rotate(Quaternion(last_ref_cs_rec['rotation'])) as points
        # def from_file_multisweep
        # Init
        # points = np.zeros((cls.nbr_dims(), 0))
        # all_pc = cls(points)
        # all_times = np.zeros((1, 0))
        #
        # # Get reference pose and timestamp
        # ref_sd_token = sample_rec['data'][ref_chan]
        # ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        # ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        # ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        # ref_time = 1e-6 * ref_sd_rec['timestamp']
        #
        # # Homogeneous transform from ego car frame to reference frame
        # ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
        #
        # # Homogeneous transformation matrix from global to _current_ ego car frame
        # car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
        #                                    inverse=True)
        #
        # # Aggregate current and previous sweeps.
        # sample_data_token = sample_rec['data'][chan]
        # current_sd_rec = nusc.get('sample_data', sample_data_token)
        # for _ in range(nsweeps):
        #     # Load up the pointcloud.
        #     current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
        #
        #     # Get past pose.
        #     current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
        #     global_from_car = transform_matrix(current_pose_rec['translation'],
        #                                        Quaternion(current_pose_rec['rotation']), inverse=False)
        #
        #     # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
        #     current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
        #     car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
        #                                         inverse=False)
        #
        #     # Fuse four transformation matrices into one and perform transform.
        #     trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        #     current_pc.transform(trans_matrix)
        #
        #     # Remove close points and add timevector.
        #     current_pc.remove_close(min_distance)
        #     time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # positive difference
        #     times = time_lag * np.ones((1, current_pc.nbr_points()))
        #     all_times = np.hstack((all_times, times))
        #
        #     # Merge with key pc.
        #     all_pc.points = np.hstack((all_pc.points, current_pc.points))

        # corresponding_sample_rec = nusc.get('sample', ref_sd_rec['sample_token'])
        # # Map the bounding boxes to the local sensor coordinate
        # # Get reference pose and timestamp
        # ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        # ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        # box_list = list()
        # attr_list = list()
        # cat_list = list()
        # id_list = list()
        #
        # for curr_sweep_box_taken in corresponding_sample_rec['anns']:
        #     ann_rec = nusc.get('sample_annotation', curr_sweep_box_taken)
        #     category_name = ann_rec['category_name']
        #     instance_token = ann_rec['instance_token']
        #     box, attr, cat = nusc.get_instance_box(ref_sd_rec['token'], instance_token)
        #     if box is not None:
        #         attr_list.append(attr)
        #         cat_list.append(cat)
        #         if instance_token in instance_to_id_dict.keys():
        #             id_list.append(instance_to_id_dict[instance_token])
        #         else:
        #             instance_to_id_dict.update({instance_token: id_idx})
        #             id_list.append(id_idx)
        #             id_idx += 1
        #         # Move box to ego vehicle coord system
        #         box.translate(-np.array(ref_pose_rec['translation']))
        #         box.rotate(Quaternion(ref_pose_rec['rotation']).inverse)
        #         # Move box to sensor coord system
        #         box.translate(-np.array(ref_cs_rec['translation']))
        #         box.rotate(Quaternion(ref_cs_rec['rotation']).inverse)
        #         # convert to self define Bbox
        #         box.id = id_list[-1]
        #         # row = np.array([*box.center, *box.wlh, box.orientation.yaw_pitch_roll[0]], dtype=np.float32)
        #         # box_save = Bbox(*row, id_list[-1])
        #         # convert category to self define
        #         flag = False
        #         for c, v in class_map.items():
        #             if category_name.startswith(c):
        #                 box.category = v
        #                 flag = True
        #                 break
        #         if not flag:
        #             box.category = 4  # Other category
        #         box_list.append(box)
        #
        # time_diff = None
        # if last_ref_pose_rec is not None:
        #     time_diff = (1e-6 * ref_pose_rec['timestamp']) - (1e-6 * last_ref_pose_rec['timestamp'])
        # # calculate velocity
        # for idx, currBox in enumerate(box_list):
        #     box_list[idx].velocity = np.array([np.nan, np.nan, np.nan])
        #     for lastSweepBox in lastSweepBoxes:
        #         if currBox.id == lastSweepBox.id:
        #             ## move coord to global
        #             lastSweepBox.rotate(Quaternion(last_ref_cs_rec['rotation']))
        #             lastSweepBox.translate(np.array(last_ref_cs_rec['translation']))
        #             lastSweepBox.rotate(Quaternion(last_ref_pose_rec['rotation']))
        #             lastSweepBox.translate(np.array(last_ref_pose_rec['translation']))
        #
        #             # Move box to ego vehicle coord system
        #             lastSweepBox.translate(-np.array(ref_pose_rec['translation']))
        #             lastSweepBox.rotate(Quaternion(ref_pose_rec['rotation']).inverse)
        #             # Move box to sensor coord system
        #             lastSweepBox.translate(-np.array(ref_cs_rec['translation']))
        #             lastSweepBox.rotate(Quaternion(ref_cs_rec['rotation']).inverse)
        #
        #             # box_list.append(lastSweepBox)
        #             ## set velocity
        #             pos_diff = currBox.center - lastSweepBox.center
        #             box_list[idx].velocity = pos_diff / time_diff
        #             break


if __name__ == '__main__':
    unittest.main()
