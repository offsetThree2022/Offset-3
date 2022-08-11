#include <math.h>
#include <stdio.h>

#define THREADS_PER_BLOCK 256
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
// #define DEBUG true


__device__ inline void lidar_to_local_coords(float shift_x, float shift_y, float rot_angle, float &local_x, float &local_y){
    float cosa = cos(-rot_angle), sina = sin(-rot_angle);
    local_x = shift_x * cosa + shift_y * (-sina);
    local_y = shift_x * sina + shift_y * cosa;
}

__device__ inline int check_pt_in_box3d(const float *pt, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center

    const float MARGIN = 1e-5;
    float x = pt[0], y = pt[1], z = pt[2];
    float cx = box3d[0], cy = box3d[1], cz = box3d[2];
    float dx = box3d[3], dy = box3d[4], dz = box3d[5], rz = box3d[6];

    if (fabsf(z - cz) > dz / 2.0) return 0;
    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}

__device__ inline int check_pt_in_box2d(const float x, const float y, const float *box3d, float &local_x, float &local_y){
    // param pt: (x, y, z)
    // param box3d: [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center

    const float MARGIN = 1e-5;
    float cx = box3d[0], cy = box3d[1];
    float dx = box3d[3], dy = box3d[4], rz = box3d[6];

    lidar_to_local_coords(x - cx, y - cy, rz, local_x, local_y);
    float in_flag = (fabs(local_x) < dx / 2.0 + MARGIN) & (fabs(local_y) < dy / 2.0 + MARGIN);
    return in_flag;
}

__global__ void points_in_boxes_kernel(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z] in LiDAR coordinate
    // params boxes_idx_of_points: (B, npoints), default -1

    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || pt_idx >= pts_num) return;

    boxes += bs_idx * boxes_num * 7;
    pts += bs_idx * pts_num * 3 + pt_idx * 3;
    box_idx_of_points += bs_idx * pts_num + pt_idx;

    float local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    for (int k = 0; k < boxes_num; k++){
        cur_in_flag = check_pt_in_box3d(pts, boxes + k * 7, local_x, local_y);
        if (cur_in_flag){
            box_idx_of_points[0] = k;
            break;
        }
    }
}

void points_in_boxes_launcher(int batch_size, int boxes_num, int pts_num, const float *boxes,
    const float *pts, int *box_idx_of_points){
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    // params pts: (B, npoints, 3) [x, y, z]
    // params boxes_idx_of_points: (B, npoints), default -1
    cudaError_t err;

    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    points_in_boxes_kernel<<<blocks, threads>>>(batch_size, boxes_num, pts_num, boxes, pts, box_idx_of_points);

    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

__global__ void build_pixel_offset_velocity_target_kernel(
                                                    int rows,
                                                    int cols,
                                                    int batch_size,
                                                    int blocksPerBatch,
                                                    int boxes_num,
                                                    const float yaw_norm,
                                                    const float* extents,
                                                    const float* boxes,
                                                    float* pixel_target_map,
                                                    float* offset_target_map,
                                                    float* velocity_target_map,
                                                    float* size_target_map,
                                                    float* yaw_target_map,
                                                    float* height_target_map,
                                                    float* category_target_map,
                                                    int* count_pixels_in_bboxes,
                                                    int* record_box_map_idx_map)
{
    // record center_map that, record idx -> pixel_center, idy -> pixel_center
    // params boxes: (B, N, 10) [x, y, z, dx, dy, dz, heading, vx, vy, category] (x, y, z) is the box center
    int num_box_elem = 10;
    const float grid_size_row = (extents[1] - extents[0]) / rows;
    const float grid_size_col = (extents[3] - extents[2]) / cols;
    int bs_idx = blockIdx.y / blocksPerBatch;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    if (bs_idx >= batch_size || idx >= rows || idy >= cols) {
        // printf("return: %d, %d, %d\n", bs_idx, idx, idy);
        return;
    }
    boxes += bs_idx * boxes_num * num_box_elem;
    count_pixels_in_bboxes += bs_idx * boxes_num;
    const float x = idx * grid_size_row + grid_size_row / 2 + extents[0];
    const float y = idy * grid_size_col + grid_size_col / 2 + extents[2];
    // printf("idx, idy (%d, %d) ==> (%f, %f) (%f, %f)\n", idx, idy, grid_size_row, grid_size_col, x, y);
    pixel_target_map = pixel_target_map + bs_idx * rows * cols;
    offset_target_map = offset_target_map + bs_idx * 2 * rows * cols; // 2 means two channels
    velocity_target_map = velocity_target_map + bs_idx * 2 * rows * cols; // 2 means two channels
    size_target_map = size_target_map + bs_idx * 2 * rows * cols;
    yaw_target_map = yaw_target_map + bs_idx * 2 * rows * cols;
    height_target_map = height_target_map + bs_idx * 2 * rows * cols;
    category_target_map = category_target_map + bs_idx * rows * cols;
    record_box_map_idx_map = record_box_map_idx_map + bs_idx * rows * cols;
    int grids = rows * cols;
    float local_x = 0, local_y = 0;
    int cur_in_flag = 0;
    // int box_id = -1;
    // float center_x = 0, center_y = 0;
    for (int k = 0; k < boxes_num; ++k) {
        if (boxes[3] < 0.005) { // boxes length is zeros boxes
            continue;
        }
        cur_in_flag = check_pt_in_box2d(x, y, boxes + k * num_box_elem, local_x, local_y);
        if (cur_in_flag) {
            const float* box_addr = boxes + k * num_box_elem;
            float size_x = box_addr[3];
            float size_y = box_addr[4];
            float yaw = box_addr[6];
            if (size_x < size_y) {
                float tmp_size = size_x;
                size_x = size_y;
                size_y = tmp_size;
                yaw += M_PI * 0.5;
            }
            yaw *= yaw_norm;
            record_box_map_idx_map[idx * cols + idy] = k;
            size_target_map[idx * cols + idy] = size_x;
            size_target_map[idx * cols + idy + grids] = size_y;
            yaw_target_map[idx * cols + idy] = sin(yaw);
            yaw_target_map[idx * cols + idy + grids] = cos(yaw);
            height_target_map[idx * cols + idy] = box_addr[2] - box_addr[5] * 0.5;
            height_target_map[idx * cols + idy + grids] = box_addr[2] + box_addr[5] * 0.5;
            // pixel target map
            pixel_target_map[idx * cols + idy] = 1.0;
            // box_id = k;
            // center_x = box_addr[0];
            // center_y = box_addr[1];
            // center_x = floor((center_x - extents[0] - grid_size_row / 2) / grid_size_row);
            // center_y = floor((center_y - extents[2] - grid_size_col / 2) / grid_size_col);
            // offset_target_map[idx * cols + idy] = idx - center_x;
            // offset_target_map[idx * cols + idy + grids] = idy - center_y;
            offset_target_map[idx * cols + idy] = x - box_addr[0];
            offset_target_map[idx * cols + idy + grids] = y - box_addr[1];
            velocity_target_map[idx * cols + idy] = box_addr[7];
            velocity_target_map[idx * cols + idy + grids] = box_addr[8];
            category_target_map[idx * cols + idy] = box_addr[9];
            atomicAdd(count_pixels_in_bboxes + k, 1);
            #ifdef DEBUG
            printf("(%f, %f): (%d, %d) ==> vel (%f, %f): cat (%f)\n",
                box_addr[0], box_addr[1], idx, idy, box_addr[7], box_addr[8], box_addr[9]);
            #endif
            // break;
            return;
        }
    }
}

__global__ void build_offset_weight_kernel(int rows,
                                    int cols,
                                    int batch_size,
                                    int blocksPerBatch,
                                    int boxes_num,
                                    const float* extents,
                                    int* record_box_map_idx_map,
                                    int* count_pixels_in_bboxes,
                                    float* offset_weight_map)
{
    const float grid_size_row = (extents[1] - extents[0]) / rows;
    int bs_idx = blockIdx.y / blocksPerBatch;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int blockIdx_y = (blockIdx.y % blocksPerBatch);
    int idy = blockIdx_y * blockDim.y + threadIdx.y;
    if (bs_idx >= batch_size || idx >= rows || idy >= cols) {
        // printf("return: %d, %d, %d\n", bs_idx, idx, idy);
        return;
    }
    count_pixels_in_bboxes += bs_idx * boxes_num;
    offset_weight_map = offset_weight_map + bs_idx * 2 * rows * cols;
    record_box_map_idx_map = record_box_map_idx_map + bs_idx * rows * cols;
    int grid = idx * cols + idy;
    int grids = rows * cols;
    int box_idx = record_box_map_idx_map[grid];
    int pixel_size = count_pixels_in_bboxes[box_idx];
    if ((box_idx >= 0) && (pixel_size > 0)) {
        float val = grid_size_row / pixel_size;
        offset_weight_map[grid] = float(val);
        offset_weight_map[grid + grids] = float(val);
        // printf("[INFO build offset weight] %d, %d -> res: %f, %f\n",
        //    box_idx, pixel_size, grid_size_row, val);
    }
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
                                           int* record_box_map_idx_map)
{
    cudaError_t err;
    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid(((rows + threadsPerBlock.x - 1) / threadsPerBlock.x),
                       ((cols + threadsPerBlock.y - 1) / threadsPerBlock.y) * batch_size);
    int blocksPerBatch = (cols + threadsPerBlock.y - 1) / threadsPerBlock.y;

    if (0) {
        fprintf(stderr, "before kernel function\n");
        fprintf(stderr, "batch_size:     %d\n", batch_size);
        fprintf(stderr, "boxes_num :     %d\n", boxes_num);
        fprintf(stderr, "rows, cols:     (%d, %d)\n", rows, cols);
        fprintf(stderr, "blocksPerBatch: (%d)\n", blocksPerBatch);
    }
    const float yaw_norm = 2.0; // sin(2 * x) means 180 deg normal
    #ifdef DEBUG
        cudaDeviceSynchronize();
        fprintf(stderr, "start build_pixel_offset_velocity_target_kernel\n");
    #endif
    build_pixel_offset_velocity_target_kernel<<<blocksPerGrid, threadsPerBlock>>>(rows,
                                                                            cols,
                                                                            batch_size,
                                                                            blocksPerBatch,
                                                                            boxes_num,
                                                                            yaw_norm,
                                                                            extents,
                                                                            boxes,
                                                                            pixel_target_map,
                                                                            offset_target_map,
                                                                            velocity_target_map,
                                                                            size_target_map,
                                                                            yaw_target_map,
                                                                            height_target_map,
                                                                            category_target_map,
                                                                            count_pixels_in_bboxes,
                                                                            record_box_map_idx_map);

    build_offset_weight_kernel<<<blocksPerGrid, threadsPerBlock>>>(rows,
                                                            cols,
                                                            batch_size,
                                                            blocksPerBatch,
                                                            boxes_num,
                                                            extents,
                                                            record_box_map_idx_map,
                                                            count_pixels_in_bboxes,
                                                            offset_weight_map);
    #ifdef DEBUG
        cudaDeviceSynchronize();
        fprintf(stderr, "end build_pixel_offset_velocity_target_kernel\n");
    #endif
    #ifdef DEBUG
        cudaDeviceSynchronize();
        fprintf(stderr, "after kernel function\n");
        fprintf(stderr, "batch_size:     %d\n", batch_size);
        fprintf(stderr, "boxes_num :     %d\n", boxes_num);
        fprintf(stderr, "rows, cols:     (%d, %d)\n", rows, cols);
        fprintf(stderr, "blocksPerBatch: (%d)\n", blocksPerBatch);
    #endif
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
#ifdef DEBUG
    cudaDeviceSynchronize();  // for using printf in kernel function
#endif
}

__device__ inline void atomicMin(float *address, const float val)
{
    int ret = __float_as_int(*address);
    while(val < __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        {
            break;
        }
    }
    *address = __int_as_float(ret);
}

__device__ inline void atomicMax(float *address, const float val)
{
    int ret = __float_as_int(*address);
    while(val > __int_as_float(ret))
    {
        int old = ret;
        if ((ret = atomicCAS((int *)address, old, __float_as_int(val))) == old)
        {
            break;
        }
    }
    *address = __int_as_float(ret);
}

__global__ void build_voxel_feature_part1_kernel(int batch_size,
                                                 const float* pts,
                                                 const float* extents,
                                                 const int rows,
                                                 const int cols,
                                                 const int pts_num,
                                                 const int voxel_num,
                                                 const int feature_num,
                                                 const int batch_size_pre_voxel,
                                                 int* pts_in_voxel_position,
                                                 float* voxel_fea)
{
    // for sum all voxel
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || pt_idx >= pts_num) return;
    const float range_z_max = extents[5];
    const float range_z_min = extents[4];
    const float range_y_max = extents[3];
    const float range_y_min = extents[2];
    const float range_x_max = extents[1];
    const float range_x_min = extents[0];
    pts += bs_idx * pts_num * 5 + pt_idx * 5; // 5 means x, y, z, i, delta_t
    if ((pts[0] >= range_x_max) ||
        (pts[0] <= range_x_min) ||
        (pts[1] >= range_y_max) ||
        (pts[1] <= range_y_min) ||
        (pts[2] >= range_z_max) ||
        (pts[2] <= range_z_min)) {
        return;
    }
    const float voxel_size_row = (range_x_max - range_x_min) / rows; // add 1e-6 ?????
    const float voxel_size_col = (range_y_max - range_y_min) / cols;
    const float voxel_size_z = (range_z_max - range_z_min) / voxel_num;
    // to be continue
    const int i = floor((pts[0] - range_x_min - voxel_size_row / 2) / voxel_size_row);
    const int j = floor((pts[1] - range_y_min - voxel_size_col / 2) / voxel_size_col);
    const int k = floor((pts[2] - range_z_min - voxel_size_z / 2) / voxel_size_z);
    if (i >= rows || i < 0 || j >= cols || j < 0 || k >= voxel_num || k < 0) {
        return;
    }
    int pos_idx = bs_idx * batch_size_pre_voxel + (i * cols + j) * voxel_num * feature_num + k * feature_num;
    float* feature = voxel_fea + pos_idx;
    // voxel_num
    atomicAdd(feature + 7, 1); // pts_per_voxel
    pts_in_voxel_position += bs_idx * pts_num + pt_idx;
    pts_in_voxel_position[0] = pos_idx;
}

__global__ void build_voxel_feature_part2_kernel(int batch_size,
                                                 const int pts_num,
                                                 const float* pts,
                                                 int* pts_in_voxel_position,
                                                 float* voxel_fea) {
    // params boxes: (B, N, 7) [x, y, z, dx, dy, dz, heading] (x, y, z) is the box center
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || pt_idx >= pts_num) return;
    pts += bs_idx * pts_num * 5 + pt_idx * 5; // 5 means x, y, z, i, delta_t
    const int curr_pt_voxel_pos = *(pts_in_voxel_position + bs_idx * pts_num + pt_idx);
    if (curr_pt_voxel_pos == -1) {
        return;
    }
    float* feature = voxel_fea + curr_pt_voxel_pos;
    const int curr_voxel_num = *(feature + 7);
    // sum some value, sum_x, sum_y, sum_z, sum_i, sum_delta_t
    // max_z. min_z
    // voxel_num
    // printf("before feature z: %f, %f\n", *(feature + 5), pts[2]);
    // printf("after max      z: %f, %f\n", *(feature + 5), pts[2]);
    atomicAdd(feature,     pts[0] / curr_voxel_num);     // mean_x
    atomicAdd(feature + 1, pts[1] / curr_voxel_num); // mean_y
    atomicAdd(feature + 2, pts[2] / curr_voxel_num); // mean_z
    atomicAdd(feature + 3, pts[3] / curr_voxel_num); // mean_i
    atomicAdd(feature + 4, pts[4] / curr_voxel_num); // mean_t
    atomicMax(feature + 5, pts[2]); // max_z
    atomicMin(feature + 6, pts[2]); // min_z
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
                                  float* voxel_fea)
{
    if (0) {
        fprintf(stderr, "build_voxel_feature_launcher\n");
        fprintf(stderr, "batch_size: %d\n", batch_size);
        fprintf(stderr, "rows, cols => (%d, %d)\n", rows, cols);
    }
    cudaError_t err;
    // for sum all voxel
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    const int batch_size_pre_voxel = rows * cols * voxel_num * feature_num;
    build_voxel_feature_part1_kernel<<<blocks, threads>>>(batch_size,
                                                          pts,
                                                          extents,
                                                          rows,
                                                          cols,
                                                          pts_num,
                                                          voxel_num,
                                                          feature_num,
                                                          batch_size_pre_voxel,
                                                          pts_in_voxel_position,
                                                          voxel_fea);


    // for divide voxel point num
    build_voxel_feature_part2_kernel<<<blocks, threads>>>(batch_size,
                                                          pts_num,
                                                          pts,
                                                          pts_in_voxel_position,
                                                          voxel_fea);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}

__global__ void build_view_index_part1_kernel(int batch_size,
                                              const float* pts,
                                              const float* extents,
                                              const int rows,
                                              const int cols,
                                              const int pts_num,
                                              const int num_channel,
                                              const int batch_size_pre_grids_channel,
                                              int* pts_in_voxel_position,
                                              float* view_index)
{
    // for sum all grids
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || pt_idx >= pts_num) return;
    const float range_z_max = extents[5];
    const float range_z_min = extents[4];
    const float range_y_max = extents[3];
    const float range_y_min = extents[2];
    const float range_x_max = extents[1];
    const float range_x_min = extents[0];
    pts += bs_idx * pts_num * 5 + pt_idx * 5; // 5 means x, y, z, i, delta_t
    if ((pts[0] >= range_x_max) ||
        (pts[0] <= range_x_min) ||
        (pts[1] >= range_y_max) ||
        (pts[1] <= range_y_min) ||
        (pts[2] >= range_z_max) ||
        (pts[2] <= range_z_min)) {
        return;
    }
    const float voxel_size_row = (range_x_max - range_x_min) / rows; // add 1e-6 ?????
    const float voxel_size_col = (range_y_max - range_y_min) / cols;
    const int i = floor((pts[0] - range_x_min - voxel_size_row / 2) / voxel_size_row);
    const int j = floor((pts[1] - range_y_min - voxel_size_col / 2) / voxel_size_col);
    if (i >= rows || i < 0 || j >= cols || j < 0) {
        return;
    }
    int pos_idx = bs_idx * batch_size_pre_grids_channel + (i * cols + j) * num_channel;
    view_index = view_index + pos_idx;
    // voxel_num
    atomicAdd(view_index, 1); // pts_per_grids
    pts_in_voxel_position += bs_idx * pts_num + pt_idx;
    pts_in_voxel_position[0] = pos_idx;
}

__global__ void build_view_index_part2_kernel(int batch_size,
                                  const int pts_num,
                                  const int boxes_num,
                                  const float* pts,
                                  int* pts_in_grid_position,
                                  int* points_in_which_bbox,
                                  int* count_points_in_bboxes,
                                  float* view_index)
{
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || pt_idx >= pts_num) return;
    pts += bs_idx * pts_num * 5 + pt_idx * 5; // 5 means x, y, z, i, delta_t
    points_in_which_bbox += bs_idx * pts_num + pt_idx;
    count_points_in_bboxes += bs_idx * boxes_num;
    const int curr_pt_grid_pos = *(pts_in_grid_position + bs_idx * pts_num + pt_idx);
    if (curr_pt_grid_pos == -1) {
        return;
    }
    // record curr pt in which bbox, and how many points dose this bbox have
    const int pt_2_box_idx = points_in_which_bbox[0];
    view_index = view_index + curr_pt_grid_pos;
    const int curr_grid_num = (*view_index);
    // atomicAdd(feature, pts[0] / curr_voxel_num);     // grid points count
    atomicAdd(view_index + 1, pts[0]); // sum_x
    atomicAdd(view_index + 2, pts[1]); // sum_y
    atomicAdd(view_index + 3, pts[0] / curr_grid_num); // center_x
    atomicAdd(view_index + 4, pts[1] / curr_grid_num); // center_y
    if (pt_2_box_idx >= 0) {
        const int curr_pts_in_box_total_pts = count_points_in_bboxes[pt_2_box_idx];
        if ((curr_pts_in_box_total_pts > 0)) { // means curr box which content this point has more than one point
            atomicAdd(view_index + 5, 1.0 / float(curr_pts_in_box_total_pts));
        }
    }
}

__global__ void build_box_count_and_point_index_kernel(int batch_size,
                                                      const float* pts,
                                                      const float* boxes,
                                                      const float* extents,
                                                      const int rows,
                                                      const int cols,
                                                      const int pts_num,
                                                      const int boxes_num,
                                                      const int num_channel,
                                                      const int batch_size_pre_grids_channel,
                                                      int* points_in_which_bbox,
                                                      int* count_points_in_bboxes)
{
    int bs_idx = blockIdx.y;
    int pt_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (bs_idx >= batch_size || pt_idx >= pts_num) return;
    const float range_z_max = extents[5];
    const float range_z_min = extents[4];
    const float range_y_max = extents[3];
    const float range_y_min = extents[2];
    const float range_x_max = extents[1];
    const float range_x_min = extents[0];
    pts += bs_idx * pts_num * 5 + pt_idx * 5; // 5 means x, y, z, i, delta_t
    if ((pts[0] >= range_x_max) ||
        (pts[0] <= range_x_min) ||
        (pts[1] >= range_y_max) ||
        (pts[1] <= range_y_min) ||
        (pts[2] >= range_z_max) ||
        (pts[2] <= range_z_min)) {
        return;
    }
    float local_x = 0, local_y = 0;
    int num_box_elem = 10;
    boxes += bs_idx * boxes_num * num_box_elem;
    points_in_which_bbox += bs_idx * pts_num + pt_idx;
    count_points_in_bboxes += bs_idx * boxes_num;
    int cur_in_flag = 0;
    for (int k = 0; k < boxes_num; ++k) {
        const float* box_address = boxes + k * num_box_elem;
        if (box_address[3] < 0.005) { // boxes length is zeros boxes
            continue;
        }
        cur_in_flag = check_pt_in_box2d(pts[0], pts[1], box_address, local_x, local_y);
        if (cur_in_flag) {
            points_in_which_bbox[0] = k;
            // const float* box_address = boxes + k * num_box_elem;
            // printf("[INFO] boxes: (%f, %f, %f, %f, %f, %f, %f) -> pts: (%f, %f, %f)\n",
            //    box_address[0], box_address[1], box_address[2],
            //    box_address[3], box_address[4], box_address[5], box_address[6],
            //    pts[0], pts[1], pts[2]);
            atomicAdd(count_points_in_bboxes + k, 1);
        }
    }
}

int build_view_index_launcher(int batch_size,
                              const float* pts,
                              const float* boxes,
                              const float* extents,
                              const int rows,
                              const int cols,
                              const int pts_num,
                              const int num_channel,
                              int* pts_in_grid_position_data,
                              float* view_index,
                              int box_num,
                              int* points_in_which_bbox,
                              int* count_points_in_bboxes)

{
    if (0) {
        fprintf(stderr, "build_view_index_launcher\n");
        fprintf(stderr, "batch_size: %d\n", batch_size);
        fprintf(stderr, "rows, cols => (%d, %d)\n", rows, cols);
        fprintf(stderr, "box_num => (%d)\n", box_num);
    }
    cudaError_t err;
    // for sum all voxel
    dim3 blocks(DIVUP(pts_num, THREADS_PER_BLOCK), batch_size);
    dim3 threads(THREADS_PER_BLOCK);
    const int batch_size_pre_grids_channel = rows * cols * num_channel;

    build_box_count_and_point_index_kernel<<<blocks, threads>>>(
                                                       batch_size,
                                                       pts,
                                                       boxes,
                                                       extents,
                                                       rows,
                                                       cols,
                                                       pts_num,
                                                       box_num,
                                                       num_channel,
                                                       batch_size_pre_grids_channel,
                                                       points_in_which_bbox,
                                                       count_points_in_bboxes);

    build_view_index_part1_kernel<<<blocks, threads>>>(batch_size,
                                                       pts,
                                                       extents,
                                                       rows,
                                                       cols,
                                                       pts_num,
                                                       num_channel,
                                                       batch_size_pre_grids_channel,
                                                       pts_in_grid_position_data,
                                                       view_index);

    build_view_index_part2_kernel<<<blocks, threads>>>(batch_size,
                                                       pts_num,
                                                       box_num,
                                                       pts,
                                                       pts_in_grid_position_data,
                                                       points_in_which_bbox,
                                                       count_points_in_bboxes,
                                                       view_index);
    err = cudaGetLastError();
    if (cudaSuccess != err) {
        fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
        exit(-1);
    }
}



