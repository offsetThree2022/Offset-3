import time
import pickle
import numpy as np
import torch
import torch.nn as nn
import os
import os.path as path
from pyquaternion import Quaternion
from functools import reduce
from ops.roiaware_pool3d import roiaware_pool3d_utils
import matplotlib.pyplot as plt

#  ################################################### for vis data #################################################
import vispy
from vispy import app, visuals
from vispy.scene import visuals, SceneCanvas
from vispy.scene.visuals import Text
from vispy.visuals import TextVisual
from vispy.color import get_colormap

RANDOR_COLORS = np.array(
     [[104, 109, 253], [125, 232, 153], [158, 221, 134],
      [228, 109, 215], [249, 135, 210], [255, 207, 237],
      [151, 120, 235], [145, 123, 213], [172, 243, 184],
      [105, 131, 110], [217, 253, 154], [250, 102, 109],
      [116, 179, 127], [200, 251, 206], [117, 146, 240],
      [234, 162, 176], [160, 172, 171], [205, 129, 168],
      [197, 167, 238], [234, 248, 101], [226, 240, 119],
      [189, 211, 231], [226, 170, 216], [109, 180, 162],
      [115, 167, 221], [162, 134, 131], [203, 169, 114],
      [221, 138, 114], [246, 146, 237], [200, 167, 244],
      [198, 150, 236], [237, 235, 191], [132, 137, 171],
      [136, 219, 103], [229, 210, 135], [133, 188, 111],
      [142, 144, 142], [122, 189, 120], [127, 142, 229],
      [249, 147, 235], [255, 195, 148], [202, 126, 227],
      [135, 195, 159], [139, 173, 142], [123, 118, 246],
      [254, 186, 204], [184, 138, 221], [112, 160, 229],
      [243, 165, 249], [200, 194, 254], [172, 205, 151],
      [196, 132, 119], [240, 251, 116], [186, 189, 147],
      [154, 162, 144], [178, 103, 147], [139, 188, 175],
      [156, 163, 178], [225, 244, 174], [118, 227, 101],
      [176, 178, 120], [113, 105, 164], [137, 105, 123],
      [144, 114, 196], [163, 115, 216], [143, 128, 133],
      [221, 225, 169], [165, 152, 214], [133, 163, 101],
      [212, 202, 171], [134, 255, 128], [217, 201, 143],
      [213, 175, 151], [149, 234, 191], [242, 127, 242],
      [152, 189, 230], [152, 121, 249], [234, 253, 138],
      [152, 234, 147], [171, 195, 244], [254, 178, 194],
      [205, 105, 153], [226, 234, 202], [153, 136, 236],
      [248, 242, 137], [162, 251, 207], [152, 126, 144],
      [180, 213, 122], [230, 185, 113], [118, 148, 223],
      [162, 124, 183], [180, 247, 119], [120, 223, 121],
      [252, 124, 181], [254, 174, 165], [188, 186, 210],
      [254, 137, 161], [216, 222, 120], [215, 247, 128],
      [121, 240, 179], [135, 122, 215], [255, 131, 237],
      [224, 112, 171], [167, 223, 219], [103, 200, 161],
      [112, 154, 156], [170, 127, 228], [133, 145, 244],
      [244, 100, 101], [254, 199, 148], [120, 165, 205],
      [112, 121, 141], [175, 135, 134], [221, 250, 137],
      [247, 245, 231], [236, 109, 115], [169, 198, 194],
      [196, 195, 136], [138, 255, 145], [239, 141, 147],
      [194, 220, 253], [149, 209, 204], [241, 127, 132],
      [226, 184, 108], [222, 108, 147], [109, 166, 185],
      [152, 107, 167], [153, 117, 222], [165, 171, 214],
      [189, 196, 243], [248, 235, 129], [120, 198, 202],
      [223, 206, 134], [175, 114, 214], [115, 196, 189],
      [157, 141, 112], [111, 161, 201], [207, 183, 214],
      [201, 164, 235], [168, 187, 154], [114, 176, 229],
      [151, 163, 221], [134, 160, 173], [103, 112, 168],
      [209, 169, 218], [137, 220, 119], [168, 220, 210],
      [182, 192, 194], [233, 187, 120], [223, 185, 160],
      [120, 232, 147], [165, 169, 124], [251, 159, 129],
      [182, 114, 178], [159, 116, 158], [217, 121, 122],
      [106, 229, 235], [164, 208, 214], [180, 178, 142],
      [110, 206, 136], [238, 152, 205], [109, 245, 253],
      [213, 232, 131], [215, 134, 100], [163, 140, 135],
      [233, 198, 143], [221, 129, 224], [150, 179, 137],
      [171, 128, 119], [210, 245, 246], [209, 111, 161],
      [237, 133, 194], [166, 157, 255], [191, 206, 225],
      [125, 135, 110], [199, 188, 196], [196, 101, 202],
      [237, 211, 167], [134, 118, 177], [110, 179, 126],
      [196, 182, 196], [150, 211, 218], [162, 118, 228],
      [150, 209, 185], [219, 151, 148], [201, 168, 104],
      [237, 146, 123], [234, 163, 146], [213, 251, 127],
      [227, 152, 214], [230, 195, 100], [136, 117, 222],
      [180, 132, 173], [112, 226, 113], [198, 155, 126],
      [149, 255, 152], [223, 124, 170], [104, 146, 255],
      [113, 205, 183], [100, 156, 216]], dtype=np.float32)

class vis():
    def __init__(self, pc):
        self.pc = pc
        self.reset()
        self.to_next_sample()

    def reset(self):
        self.action = "no" #, no , next, back, quit
        self.canvas = SceneCanvas(keys='interactive', size=(1600, 1200), show=True, bgcolor='k')
        self.canvas.events.key_press.connect(self.key_press_event)

        # interface(n next, b back, q quit, very simple)
        self.lidar_view = self.canvas.central_widget.add_view()
        self.lidar_view.camera = 'turntable'
        visuals.XYZAxis()
        self.lidar_vis = visuals.Markers()
        self.lidar_view.add(self.lidar_vis)

        # draw lidar boxes
        # self.line_vis = visuals.Line(color='r', method='gl', connect="segments", name="boxes line")
        # self.lidar_view.add(self.line_vis)
        # self.flip = 1.0

    def to_next_sample(self):
        """
        to next sample
        """
        #  show lidar
        self.lidar_vis.set_gl_state('translucent', depth_test=False)
        color = np.zeros((self.pc.transpose().shape[0], 3), np.float32)
        pc_t = self.pc.transpose()
        for i in range(self.pc.transpose().shape[0]):
            color[i, :] = RANDOR_COLORS[int(pc_t[i, 3]) % RANDOR_COLORS.shape[0]] / 255.0
        self.lidar_vis.set_data(self.pc.transpose()[:, :3], edge_color=None, face_color=color, size=2)
        self.lidar_view.add(self.lidar_vis)
        self.lidar_view.camera = 'turntable'

    def draw_boxes(self, boxes):
        num_pts_pre_boxes=24
        all_bboxes_pts = np.zeros((len(boxes) * num_pts_pre_boxes, 3), dtype=np.float32)
        box_idx = 0
        color_lines = np.zeros((len(boxes) * num_pts_pre_boxes, 3), dtype=np.float32)
        for box in boxes:
            box_pt = box.get_box_pts().transpose()
            all_bboxes_pts[box_idx * num_pts_pre_boxes : (box_idx + 1) * num_pts_pre_boxes, :] =\
                box_pt[[0, 1, 4, 5, 7, 6, 3, 2, 0, 3, 3, 7, 7, 4, 4, 0, 2, 6, 6, 5, 5, 1, 1, 2], :]
            color_lines[box_idx * num_pts_pre_boxes : (box_idx + 1) * num_pts_pre_boxes, :] = \
                RANDOR_COLORS[box.id % RANDOR_COLORS.shape[0]] / 255.0
            box_idx += 1
        self.line_vis.set_data(all_bboxes_pts, color=color_lines)

    def key_press_event(self, event):
        if event.key == 'N':
            self.offset += 1
            print("[EVENT] N")
            # self.update_canvas()
            self.to_next_sample()

        elif event.key == "B":
            self.offset -= 1
            print("[EVENT] B")
            # self.update_canvas()
            self.to_next_sample()

    def on_draw(selfself, event):
        print("[KEY INFO] draw")


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
    curr_boxes = np.zeros((len(curr_boxes_gt), 10), np.float32)
    for i, elem in enumerate(curr_boxes_gt):
        whl = elem.wlh
        velocity = elem.velocity[:2]
        if np.isnan(velocity).any():
            velocity = [0.0, 0.0]
        curr_boxes[i, :] = ([*elem.center, whl[1], whl[0], whl[2], elem.orientation.yaw_pitch_roll[0],
                            velocity[0], velocity[1], elem.category_num])
    return curr_boxes

data_root = "/media/user/C14D581BDA18EBFA/nuScenesGenData"

# torch.cuda.synchronize()
print("start code process")
with open(path.join(data_root, "trainlist.pkl"), "rb") as f:
    train_list = pickle.load(f)

rows = 256
cols = 256
voxel_num = 10
feature_num = 8
num_past_lidar = 5
num_future_lidar = 25
extents_cpu = torch.Tensor([[-32., 32.], [-32., 32.], [-3., 2.]]).float()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def getitem(idx):
    example = train_list[idx]
    value = example.split("_")
    scene_idx = int(value[0])
    sweep_idx = int(value[1])
    ref_lidar_file_name = os.path.join(data_root, "scene_{}/lidars/{}.bin".format(scene_idx, sweep_idx))
    ref_boxes_file_name = os.path.join(data_root, "scene_{}/boxes/{}.pkl".format(scene_idx, sweep_idx))
    ref_calibration_file_name = os.path.join(data_root, "scene_{}/calibration/{}.pkl".format(scene_idx, sweep_idx))
    #  Get refernce pose and timestamp
    with open(ref_calibration_file_name, "rb") as f:
        ref_pose_rec, ref_cs_rec = pickle.load(f)
    # Homogeneous transform from ego car frame to reference frame
    ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
    # Homogeneous transformation matrix from global to _current_ ego car frame
    car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)

    assert path.isfile(ref_lidar_file_name), "{} is not exist".format(ref_lidar_file_name)
    assert path.isfile(ref_boxes_file_name), "{} is not exist".format(ref_boxes_file_name)
    assert path.isfile(ref_calibration_file_name), "{} is not exist".format(ref_calibration_file_name)
    # print("from here 1 cost : {} ms".format((time.time() - start_g) * 1000))
    #  merge num past lidar sweep for lidar input
    #  first zeros pts
    last_time_stamp = ref_pose_rec['timestamp']
    all_pc = np.fromfile(ref_lidar_file_name, np.float32).reshape((-1, 5))[:, :4].T
    all_pc = np.vstack((all_pc, np.zeros(all_pc.shape[1])))
    for idx in range(1, num_past_lidar):
        curr_lidar_file_name = os.path.join(data_root, "scene_{}/lidars/{}.bin".format(scene_idx, sweep_idx - idx))
        curr_calibration_file_name = os.path.join(data_root, "scene_{}/calibration/{}.pkl"
                                                  .format(scene_idx, sweep_idx - idx))
        # print("[INFO] read curr {} lidar name".format(curr_lidar_file_name))
        curr_pc = np.fromfile(curr_lidar_file_name, np.float32).reshape((-1, 5))[:, :4].T

        # Get past pose
        with open(curr_calibration_file_name, "rb") as f:
            current_pose_rec, current_cs_rec = pickle.load(f)
        global_from_car = transform_matrix(current_pose_rec['translation'],
                                           Quaternion(current_pose_rec['rotation']), inverse=False)
        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                            inverse=False)
        # Fuse four transformation matrices into one and perform transform.
        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
        curr_pc[:3, :] = trans_matrix.dot(np.vstack((curr_pc[:3, :], np.ones(curr_pc.shape[1]))))[:3, :]
        curr_time_stamp = current_pose_rec['timestamp']
        time_diff = 1e-6 * (last_time_stamp - curr_time_stamp)
        #  hstask timestamp to pc
        curr_pc = np.vstack([curr_pc, time_diff * np.ones(curr_pc.shape[1])])
        all_pc = np.hstack((all_pc, curr_pc))
    # print("from here 2 cost : {} ms".format((time.time() - start_g) * 1000))
    # trans multi lidar sweep to voxel grid
    """
        voxel: W, H, C, num_of_pts_pre_voxel
        non_zeros_map: 
        pixel_target:
        offset_target:
    """
    #  read bot gt
    # print("from here 3 cost : {} ms".format((time.time() - start_g) * 1000))
    pickle_s = time.time()
    with open(ref_boxes_file_name, "rb") as f:
        curr_boxes_gt = pickle.load(f)
    print("pickle load ref boxes cost: {} ms".format((time.time() - pickle_s) * 1000))
    # print("from here 4 cost : {} ms".format((time.time() - start_g) * 1000))
    curr_boxes_gt = convert_pickle_boxes_to_torch_box(curr_boxes_gt)
    # print("from here 5 cost : {} ms".format((time.time() - start_g) * 1000))
    start = time.time()
    # print("from here 6 cost : {} ms".format((time.time() - start_g) * 1000))
    input_pc = torch.from_numpy(np.ascontiguousarray(all_pc.transpose())).unsqueeze(dim=0).cuda().float()
    # print("from here 7 cost : {} ms".format((time.time() - start_g) * 1000))
    # print("convert cpu pc to gpu has cost about: {} ms".format((time.time() - start) * 1000))
    input_gt_boxes = torch.from_numpy(curr_boxes_gt).unsqueeze(dim=0).float().cuda()
    # start = time.time()
    # box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(input_pc, input_gt_boxes).long().squeeze(dim=0).cpu().numpy()
    # print("points_in_boxes_gpu has cost about: {} ms".format((time.time() - start) * 1000))
    # all_pc = all_pc[:, box_idxs_of_pts != -1]
    # all_pc[3, box_idxs_of_pts == -1] = 0
    # all_pc[3, box_idxs_of_pts != -1] = 4
    extents = extents_cpu.cuda()
    start = time.time()
    pixel_target_map, offset_target_map, velocity_target_map = roiaware_pool3d_utils.\
        build_pixel_offset_velocity_target(rows, cols, extents, input_gt_boxes, device)
    print("build_pixel_offset_velocity_target has cost about: {} ms".format((time.time() - start) * 1000))
    start = time.time()
    voxel_feature = roiaware_pool3d_utils.build_voxel_feature(input_pc, extents, rows, cols, voxel_num, feature_num, device)
    print("build voxel feature has cost about: {} ms".format((time.time() - start) * 1000))
    voxel_count_map = voxel_feature[:, :, :, 7]
    for idx in range(1, 10):
        voxel_count_map += voxel_feature[:, :, :, 7 + idx * feature_num]
    ret = {
        'points': input_pc.squeeze(dim=0),
        'pixel_target_map': pixel_target_map,
        'offset_target_map': offset_target_map,
        'velocity_target_map': velocity_target_map,
        'voxel_feature': voxel_feature,
        'input_gt_boxes': input_gt_boxes,
        'voxel_count_map': voxel_count_map
    }
    return ret



# for vis
############################### vis nuscenes dataset for doing some practice ###########################################
if __name__ == "__main__":
    # sweep_idx = 79957
    sweep_idx = 0
    for idx in range(0, 5):
        start_g = time.time()
        ret = getitem(sweep_idx + idx)
        all_pc = ret['points']
        pixel_target_map = ret['pixel_target_map']
        offset_target_map = ret['offset_target_map']
        velocity_target_map = ret['velocity_target_map']
        voxel_feature = ret['voxel_feature']
        input_gt_boxes = ret['input_gt_boxes']
        voxel_count_map = ret['voxel_count_map']
        print("{} [TIME COST INFO] the whole process cost about: {} ms"
              .format(idx, (time.time() - start_g) * 1000))  #, torch.nonzero(voxel_feature).size(0)))

    fig, ax = plt.subplots(2, 3, figsize=(50, 50))
    voxel_count_map_numpy = voxel_count_map.cpu().numpy().squeeze()
    pixel_mask = (voxel_count_map_numpy > 0.5)
    ax[1, 0].imshow(pixel_mask.T)
    plt.show()

    #  draw pixel and offset
    # fig, ax = plt.subplots(2, 3, figsize=(50, 50))
    # #  draw points
    # ax[0, 0].scatter(all_pc[0, :], all_pc[1, :])
    # ax[0, 0].axis('off')
    # ax[0, 0].set_aspect('equal')
    # ax[0, 0].title.set_text('Lidar data')
    #
    # #  draw pixel gt
    # pixel_gt_numpy = pixel_target_map.cpu().numpy().squeeze()
    # pixel_mask = (pixel_gt_numpy > 0.5)
    # idx_x = np.arange(rows)
    # idx_y = np.arange(cols)
    # idx_x, idx_y = np.meshgrid(idx_x, idx_y, indexing='ij')
    # # X = idx_x[pixel_mask]
    # # Y = idx_y[pixel_mask]
    # ax[1, 0].imshow(pixel_mask.T)
    # # ax[1, 0].plot(X, Y, '.')
    # ax[1, 0].set_aspect('equal')
    # ax[1, 0].axis('off')
    # ax[1, 0].title.set_text('pixel gt')
    #
    # #  draw offset gt
    # pos_nonzeros_tuple = np.where(pixel_mask == True)
    # #  too mush positive pixel need to show, we sample it
    # pox_x, pos_y = pos_nonzeros_tuple[0], pos_nonzeros_tuple[1]
    # pos_selected_mask = np.zeros_like((pixel_gt_numpy > 0.5)).astype(np.bool)
    # pos_selected_mask[pox_x, pos_y] = True
    # offset_target_map_numpy = offset_target_map.cpu().numpy().squeeze()
    # X = idx_x[pos_selected_mask]
    # Y = idx_y[pos_selected_mask]
    # U = -offset_target_map_numpy[0, :, :][pos_selected_mask]
    # V = -offset_target_map_numpy[1, :, :][pos_selected_mask]
    # ax[0, 2].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='r')
    # ax[0, 2].set_aspect('equal')
    # ax[0, 2].title.set_text('offset Prediction')
    # ax[0, 2].axis('off')
    #
    # # draw velocity
    # #  draw offset gt
    # grid_size = (extents_cpu[0, 1] - extents_cpu[0, 0]) / rows
    # velocity_target_map_numpy = velocity_target_map.cpu().numpy().squeeze()
    # U = velocity_target_map_numpy[0, :, :][pos_selected_mask] / grid_size.numpy()
    # V = velocity_target_map_numpy[1, :, :][pos_selected_mask] / grid_size.numpy()
    # ax[0, 1].quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='g')
    # ax[0, 1].set_aspect('equal')
    # ax[0, 1].title.set_text('velocity Prediction')
    # ax[0, 1].axis('off')
    #
    # #  draw features
    # voxel_feature_channel_numpy = voxel_feature[0, :, :, 0].cpu().numpy()
    # for i in range(1, 10):
    #     voxel_feature_channel_numpy += voxel_feature[0, :, :, 0 + 8 * i].cpu().numpy()
    # voxel_feature_pos = (voxel_feature_channel_numpy != 0).T
    # # X = idx_x[voxel_feature_pos]
    # # Y = idx_y[voxel_feature_pos]
    # # X = idx_x[voxel_feature_pos]
    # # Y = idx_y[voxel_feature_pos]
    # # ax[1, 1].plot(X, Y, '.')
    # ax[1, 1].imshow(voxel_feature_pos)
    # ax[1, 1].set_aspect('equal')
    # ax[1, 1].axis('off')
    # ax[1, 1].title.set_text('build voxel feature vis')
    #
    # plt.show()
    # # plt.savefig(os.path.join("img/", str(idx) + '.png'))
    # ax[0, 0].clear()
    # ax[0, 1].clear()
    # ax[0, 2].clear()
    # ax[1, 0].clear()
    # ax[1, 1].clear()
    # plt.close()
    # # vistool=vis(all_pc)
    # # vispy.app.run()
    # print("[INFO] code finished")