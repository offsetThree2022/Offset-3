#include <torch/serialize/tensor.h>
#include <torch/extension.h>
#include <assert.h>

void points_in_boxes_launcher(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points);

int points_in_boxes_gpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor box_idx_of_points_tensor) {
    // params boxes: (B, N, 9) [x, y, z, dx, dy, dz, heading, vx, vy] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z]
    // params boxes_idx_of_points: (B, npoints), default -1

    //    CHECK_INPUT(boxes_tensor);
    //    CHECK_INPUT(pts_tensor);
    //    CHECK_INPUT(box_idx_of_points_tensor);

    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int pts_num = pts_tensor.size(1);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *box_idx_of_points = box_idx_of_points_tensor.data<int>();

    points_in_boxes_launcher(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

    return 1;
}

void build_pixel_offset_velocity_target_launcher(
                                           int batch_size,
                                           int boxes_num,
                                           int rows,
                                           int cols,
                                           const float *boxes,
                                           const float *extents,
                                           float* pixel_target_map,
                                           float* offset_target_map,
                                           float* offset_weight_map,
                                           float* velocity_target_map,
                                           float* size_target_map,
                                           float* yaw_target_map,
                                           float* height_target_map,
                                           float* category_target_map,
                                           int* count_pixels_in_bboxes,
                                           int* record_box_map_idx_map);

int build_pixel_offset_velocity_target(int rows,
                                 int cols,
                                 at::Tensor extents_tensor,
                                 const at::Tensor boxes_tensor,
                                 at::Tensor pixel_target_map_tensor,
                                 at::Tensor offset_target_map_tensor,
                                 at::Tensor offset_weight_map_tensor,
                                 at::Tensor velocity_target_map_tensor,
                                 at::Tensor size_target_map_tensor,
                                 at::Tensor yaw_target_map_tensor,
                                 at::Tensor height_target_map_tensor,
                                 at::Tensor category_target_map_tensor,
                                 at::Tensor count_pixels_in_bboxes_tensor,
                                 at::Tensor record_box_map_idx_map_tensor)
{
    // params boxes: (B, N, 9) [x, y, z, dx, dy, dz, heading, vx, vy] (x, y, z) is the box center
    // CHECK_INPUT(boxes_tensor);
    int batch_size = boxes_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    const float* boxes = boxes_tensor.data<float>();
    const float* extents = extents_tensor.data<float>();
    float* pixel_target_map = pixel_target_map_tensor.data<float>();
    float* offset_target_map = offset_target_map_tensor.data<float>();
    float* offset_weight_map_data = offset_weight_map_tensor.data<float>();
    float* velocity_target_map = velocity_target_map_tensor.data<float>();
    float* size_target_map = size_target_map_tensor.data<float>();
    float* yaw_target_map = yaw_target_map_tensor.data<float>();
    float* height_target_map = height_target_map_tensor.data<float>();
    float* category_target_map = category_target_map_tensor.data<float>();
    int* count_pixels_in_bboxes_data = count_pixels_in_bboxes_tensor.data<int>();
    int* record_box_map_idx_map_data = record_box_map_idx_map_tensor.data<int>();
    build_pixel_offset_velocity_target_launcher(batch_size,
                                                   boxes_num,
                                                   rows,
                                                   cols,
                                                   boxes,
                                                   extents,
                                                   pixel_target_map,
                                                   offset_target_map,
                                                   offset_weight_map_data,
                                                   velocity_target_map,
                                                   size_target_map,
                                                   yaw_target_map,
                                                   height_target_map,
                                                   category_target_map,
                                                   count_pixels_in_bboxes_data,
                                                   record_box_map_idx_map_data);
    return 1;
}

void build_voxel_feature_launcher(int batch_size,
                                  const float* pts,
                                  const float* extents,
                                  const int rows,
                                  const int cols,
                                  const int pts_num,
                                  const int voxel_num,
                                  const int feature_num,
                                  int* pts_in_voxel_position,
                                  float* voxel_fea);

int build_voxel_feature(at::Tensor points_tensor,
                         const int rows,
                         const int cols,
                         at::Tensor extents_tensor,
                         const int voxel_num,
                         const int feature_num,
                         at::Tensor pts_in_voxel_position,
                         at::Tensor voxel_feature)
{
    /*
        input point cloud get voxel feature tensor
        voxel_feature shape: (B, rows, cols, voxel_num * feature_num)
        features:
            means: x, y, z, i, delta_t
            max  : z
            min  : z
            voxel_num: number of points pre voxel
    */
    int batch_size = points_tensor.size(0);
    int pts_num = points_tensor.size(1);
    const float* pts = points_tensor.data<float>();
    const float* extents = extents_tensor.data<float>();
    float* voxel_fea = voxel_feature.data<float>();
    int* pts_in_voxel_position_data = pts_in_voxel_position.data<int>();
    build_voxel_feature_launcher(batch_size, pts, extents, rows, cols, pts_num, voxel_num,
        feature_num, pts_in_voxel_position_data, voxel_fea);
    return 1;
}

int build_view_index_launcher(int batch_size,
                              const float* pts,
                              const float *boxes,
                              const float* extents,
                              const int rows,
                              const int cols,
                              const int pts_num,
                              const int num_channel,
                              int* pts_in_grid_position_data,
                              float* view_index,
                              int box_num,
                              int* points_in_which_bbox,
                              int* count_points_in_bboxes);

int build_view_index(at::Tensor points_tensor,
                     const int rows,
                     const int cols,
                     const int num_channel,
                     at::Tensor boxes_tensor,
                     at::Tensor extents_tensor,
                     at::Tensor pts_in_grid_position,
                     at::Tensor view_index_tensor,
                     at::Tensor points_in_which_bbox_tensor,
                     at::Tensor count_points_in_bboxes_tensor)
{
    int batch_size = points_tensor.size(0);
    int boxes_num = boxes_tensor.size(1);
    int pts_num = points_tensor.size(1);
    if (0) {
        printf("points_in_which_bbox_data shape: (%d, %d), pts_num(%d), boxes_num:(%d)\n",
            points_in_which_bbox_tensor.size(0),
            points_in_which_bbox_tensor.size(1),
            pts_num,
            boxes_num);
    }
    const float* pts = points_tensor.data<float>();
    const float* boxes = boxes_tensor.data<float>();
    const float* extents = extents_tensor.data<float>();
    float* view_index = view_index_tensor.data<float>();
    int* pts_in_grid_position_data = pts_in_grid_position.data<int>();
    int* points_in_which_bbox_data = points_in_which_bbox_tensor.data<int>();
    int* count_points_in_bboxes_data = count_points_in_bboxes_tensor.data<int>();
    build_view_index_launcher(batch_size, pts, boxes, extents, rows, cols, pts_num,
        num_channel, pts_in_grid_position_data, view_index, boxes_num,
        points_in_which_bbox_data, count_points_in_bboxes_data);
    return 1;
}


// cpu
inline void lidar_to_local_coords_cpu(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y) {
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}

inline int check_pt_in_box3d_cpu(const float *pt, const float *box3d, float &local_x, float &local_y) {
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    const float MARGIN = 1e-2;
    float x = pt[0], y = pt[1], z = pt[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    if (fabsf(z - cz) > dz / 2.0) return 0;
    lidar_to_local_coords_cpu(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}

int points_in_boxes_cpu(at::Tensor boxes_tensor, at::Tensor pts_tensor, at::Tensor pts_indices_tensor) {
    // params boxes: (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center, each box DO NOT overlaps
    // params pts: (num_points, 3) [x, y, z]
    // params pts_indices: (N, num_points)

    //    CHECK_CONTIGUOUS(boxes_tensor);
    //    CHECK_CONTIGUOUS(pts_tensor);
    //    CHECK_CONTIGUOUS(pts_indices_tensor);

    int boxes_num = boxes_tensor.size(0);
    int pts_num = pts_tensor.size(0);

    const float *boxes = boxes_tensor.data<float>();
    const float *pts = pts_tensor.data<float>();
    int *pts_indices = pts_indices_tensor.data<int>();

    float local_x = 0, local_y = 0;
    for (int i = 0; i < boxes_num; i++) {
        for (int j = 0; j < pts_num; j++) {
            int cur_in_flag = check_pt_in_box3d_cpu(pts + j * 3, boxes + i * 7, local_x, local_y);
            pts_indices[i * pts_num + j] = cur_in_flag;
        }
    }
    return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("points_in_boxes_gpu", &points_in_boxes_gpu, "points_in_boxes_gpu forward (CUDA)");
    m.def("points_in_boxes_cpu", &points_in_boxes_cpu, "points_in_boxes_cpu forward (CUDA)");
    m.def("build_voxel_feature", &build_voxel_feature, "build voxel features forward (CUDA)");
    m.def("build_view_index", &build_view_index, "build view index forward (CUDA)");
    m.def("build_pixel_offset_velocity_target", &build_pixel_offset_velocity_target, "build_pixel_offset_velocity_target forward (CUDA)");
}
