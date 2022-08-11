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
    def __init__(self, x, y, z, l, w, h, theta):
        self.xyz = [x, y, z]
        self.hwl = [h, w, l]
        self.yaw = theta

    def roty(self, t):
        ''' Rotation about the y-axis. '''
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c, 0, s],
                         [0, 1, 0],
                         [-s, 0, c]])

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
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        corner = np.vstack([x_corners, y_corners, z_corners])
        corner_3d = np.dot(R, corner)
        corner_3d[0, :] = corner_3d[0, :] + self.xyz[0]
        corner_3d[1, :] = corner_3d[1, :] + self.xyz[1]
        corner_3d[2, :] = corner_3d[2, :] + self.xyz[2]
        return corner_3d.astype(np.float32)

class vis():
    def __init__(self, offset):
        self.offset = offset
        self.reset()
        self.update_canvas()

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
                RANDOR_COLORS[box_idx % RANDOR_COLORS.shape[0]] / 255.0
            box_idx += 1
        self.line_vis.set_data(all_bboxes_pts, color=color_lines)

    def key_press_event(self, event):
        if event.key == 'N':
            self.offset += 1
            print("[EVENT] N")
            self.update_canvas()
        elif event.key == "B":
            self.offset -= 1
            print("[EVENT] B")
            self.update_canvas()

    def on_draw(selfself, event):
        print("[KEY INFO] draw")


if __name__ == '__main__':
    vis = vis(offset=0)
    vispy.app.run()