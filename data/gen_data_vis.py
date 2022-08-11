# Copyright (c) 2020 Mitsubishi Electric Research Laboratories (MERL). All rights reserved. The software, documentation and/or data in this file is provided on an "as is" basis, and MERL has no obligations to provide maintenance, support, updates, enhancements or modifications. MERL specifically disclaims any warranties, including, but not limited to, the implied warranties of merchantability and fitness for any particular purpose. In no event shall MERL be liable to any party for direct, indirect, special, incidental, or consequential damages, including lost profits, arising out of the use of this software and its documentation, even if MERL has been advised of the possibility of such damages. As more fully described in the license agreement that was required in order to download this software, documentation and/or data, permission to use, copy and modify this software without fee is granted, but only for educational, research and non-commercial purposes.


from nuscenes.nuscenes import NuScenes
import os
from nuscenes.utils.data_classes import LidarPointCloud
import numpy as np
import argparse
from data.data_utils import voxelize_occupy, gen_2d_grid_gt

####
import vispy
from vispy import app, visuals
from vispy.scene import visuals, SceneCanvas
from vispy.scene.visuals import Text
from vispy.visuals import TextVisual
import numpy as np
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

class Bbox():
    def __init__(self, x, y, z, w, l, h, theta, id):
        self.xyz = [x, y, z]
        self.hwl = [h, w, l]
        self.yaw = theta
        self.id = id

    def roty(self, t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, -s, 0],
                         [s, c, 0],
                         [0, 0, 1]])

    def center(self):
        return np.array([self.xyz[0], self.xyz[1], self.xyz[2]])

    def get_box_pts(self):
        """
            7 -------- 3
           /|         /|
          4 -------- 0 .
          | |        | |
          . 6 -------- 2
          |/         |/
          5 -------- 1
        Args:
            boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center

        Returns:
        """
        l = self.hwl[2]
        w = self.hwl[1]
        h = self.hwl[0]
        R = self.roty(float(self.yaw))  # 3*3
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        z_corners = [h / 2, -h / 2, -h / 2, h / 2, h / 2, -h / 2, -h / 2, h / 2]
        y_corners = [-w / 2, -w / 2, -w / 2, -w / 2, w / 2, w / 2, w / 2, w / 2]
        corner = np.vstack([x_corners, y_corners, z_corners])
        corner_3d = np.dot(R, corner)
        corner_3d[0, :] = corner_3d[0, :] + self.xyz[0]
        corner_3d[1, :] = corner_3d[1, :] + self.xyz[1]
        corner_3d[2, :] = corner_3d[2, :] + self.xyz[2]
        return corner_3d.astype(np.float32)

class vis():
    def __init__(self, scenes_idx=0, offset=0):
        self.curr_scenes = nusc.scene[scenes_idx]
        self.first_sample_token = self.curr_scenes['first_sample_token']
        self.curr_sample = nusc.get('sample', self.first_sample_token)
        self.curr_sample_data = nusc.get('sample_data', self.curr_sample['data']['LIDAR_TOP'])
        print("Processing scene {} ...".format(scenes_idx))
        self.offset = offset
        self.reset()
        self.to_next_sample()
        # self.update_canvas()

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
        self.line_vis = visuals.Line(color='r', method='gl', connect="segments", name="boxes line")
        self.lidar_view.add(self.line_vis)
        self.flip = 1.0

    def to_next_sample(self):
        """
        to next sample
        """
        print("[to_netxt_sample] to_next_sample")
        curr_sample_data = self.curr_sample_data
        if curr_sample_data['next'] == '':
            print("[INFO] curr data has reach the end")
            return

        curr_sample = self.curr_sample
        all_pc, all_times, trans_matrices = \
            LidarPointCloud.from_file_multisweep_bf_sample_data(nusc, curr_sample_data,
                                                                return_trans_matrix=True,
                                                                nsweeps_back=nsweeps_back,
                                                                nsweeps_forward=nsweeps_forward)

        # Store point cloud of each sweep
        pc = all_pc.points
        _, sort_idx = np.unique(all_times, return_index=True)
        unique_times = all_times[np.sort(sort_idx)]  # Preserve the item order in unique_times
        num_sweeps = len(unique_times)

        ## show lidar
        self.lidar_vis.set_gl_state('translucent', depth_test=False)
        self.lidar_vis.set_data(pc.transpose()[:, :3], edge_color=None, face_color=(1.0, 1.0, 1.0, 1.0), size=2)
        self.lidar_view.add(self.lidar_vis)
        self.lidar_view.camera = 'turntable'
        ## end show lidar

        # Get the syncchronized bounding boxes
        # First, we need to iterate all the instances woith this example
        num_instances = 0  # The number of instances within this sample
        corresponding_sample_token = curr_sample_data['sample_token']
        corresponding_sample_rec = nusc.get('sample', corresponding_sample_token)

        bboxes_all_instances = []
        for ana_token in corresponding_sample_rec['anns']:
            ann_rec = nusc.get('sample_annotation', ana_token)
            category_name = ann_rec['category_name']
            instance_token = ann_rec['instance_token']

            instance_boxes, instance_all_times, _, _ = LidarPointCloud. \
                get_instance_boxes_multisweep_sample_data(nusc, curr_sample_data,
                                                          instance_token,
                                                          nsweeps_back=nsweeps_back,
                                                          nsweeps_forward=nsweeps_forward)
            assert np.array_equal(unique_times, instance_all_times), "The sweep and instance times are inconsistent!"
            assert num_sweeps == len(instance_boxes), "The number of instance boxes does not match that of sweeps!"

            #Each row corresponds to a box annotation; the column consists of box center, box size, and quaternion
            box_data = np.zeros((len(instance_boxes), 3 + 3 + 1), dtype=np.float32)  # center, size, yaw
            box_data.fill(np.nan)
            for r, box in enumerate(instance_boxes):
                if box is not None:
                    row = np.array([*box.center, *box.wlh, box.orientation.yaw_pitch_roll[0]], dtype=np.float32)
                    box_data[r] = row[:]
            num_instances += 1
            bboxes_all_instances.append(box_data)
        # trans box_data to bboxes
        boxes = []
        box_id = 0
        for box_data in bboxes_all_instances:
            for i in range(box_data.shape[0]):
                data = box_data[i, :]
                boxes.append(Bbox(*data, box_id))
            box_id += 1
        self.draw_boxes(boxes)
        # end show bboxes

        ## for show next data prep
        if curr_sample['next'] != '':
            curr_sample = nusc.get('sample', curr_sample['next'])
            curr_sample_data = nusc.get('sample_data', curr_sample['data']['LIDAR_TOP'])
            self.curr_sample = curr_sample
            self.curr_sample_data = curr_sample_data

    def update_canvas(self):
        lidar_path = r'/home/user/SA-SSD/data/training/velodyne/%06d.bin' % self.offset  ## Path ## need to be changed
        print("path: " + str(lidar_path))
        title = "lidar" + str(self.offset)
        self.canvas.title = title
        lidar_points = np.fromfile(lidar_path, np.float32).reshape(-1, 4)
        self.lidar_vis.set_gl_state('translucent', depth_test=False)
        ### set lidar show
        self.lidar_vis.set_data(lidar_points[:, :3], edge_color = None, face_color = (1.0, 1.0, 1.0, 1.0), size = 2)
        self.lidar_view.add(self.lidar_vis)
        self.lidar_view.camera = 'turntable'

        ### set box show
        boxes = []
        for idx in range(300):
            boxes.append(Bbox(60.0 - idx * 1.0 * self.flip, -60 + idx * 1.0 * self.flip, -0.5, 5.0, 2.2, 1.9, 0.0))
        self.draw_boxes(boxes)
        self.flip = self.flip * -1.0

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

########################################################################################################################
def check_folder(folder_name):
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    return folder_name


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--root', default='/media/pwu/Data/3D_data/nuscene/all_nuscene', type=str, help='Root path to nuScenes dataset')
parser.add_argument('-s', '--split', default='train', type=str, help='The data split [train/val/test]')
parser.add_argument('-p', '--savepath', default='/media/pwu/62316788-a8e6-423c-9ed3-303ebb3ab2de/pwu/temporal_data/train', type=str, help='Directory for saving the generated data')
args = parser.parse_args()

scenes = np.load('data/split.npy', allow_pickle=True).item().get(args.split)

nusc = NuScenes(version='v1.0-trainval', dataroot=args.root, verbose=True)
print("Total number of scenes:", len(nusc.scene))

class_map = {'vehicle.car': 1, 'vehicle.bus.rigid': 1, 'vehicle.bus.bendy': 1, 'human.pedestrian': 2,
             'vehicle.bicycle': 3}  # background: 0, other: 4


if args.split == 'train':
    num_keyframe_skipped = 0  # The number of keyframes we will skip when dumping the data
    nsweeps_back = 30  # Number of frames back to the history (including the current timestamp)
    nsweeps_forward = 20  # Number of frames into the future (does not include the current timestamp)
    skip_frame = 0  # The number of frames skipped for the adjacent sequence
    num_adj_seqs = 2  # number of adjacent sequences, among which the time gap is \delta t
else:
    num_keyframe_skipped = 1
    nsweeps_back = 25  # Setting this to 30 (for training) or 25 (for testing) allows conducting ablation studies on frame numbers
    nsweeps_forward = 20
    skip_frame = 0
    num_adj_seqs = 1

# The specifications for BEV maps
voxel_size = (0.25, 0.25, 0.4)
area_extents = np.array([[-32., 32.], [-32., 32.], [-3., 2.]])
past_frame_skip = 3  # when generating the BEV maps, how many history frames need to be skipped
future_frame_skip = 0  # when generating the BEV maps, how many future frames need to be skipped
num_past_frames_for_bev_seq = 5  # the number of past frames for BEV map sequence


scenes = np.load('data/split.npy', allow_pickle=True).item().get(args.split)
print("Split: {}, which contains {} scenes.".format(args.split, len(scenes)))

############################### vis nuscenes dataset for doing some practice ###########################################
if __name__ == "__main__":
    scenes_idx = 599
    vistool=vis(scenes_idx=scenes_idx, offset=0)
    vispy.app.run()
