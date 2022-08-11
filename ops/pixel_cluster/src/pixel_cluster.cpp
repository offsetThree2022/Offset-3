#include "pixel_clsuter.h"

void testPixelCluster()
{
    PixelCluster pixelClu;
    pixelClu.sayHello();
}

void processBatchData(bool is_training,
                      const float* non_zeros_map_data,
                      const float* pixel_pred_data,
                      const float* offset_pred_data,
                      const float* pixel_gt_data,
                      float* pixel_weight_data,
                      float* offset_weight_data);

void PixelCluster::init(bool is_training,
                        const float *non_zero_map_data,
                        const float *pixel_pred_data,
                        const float *offset_pred_data,
                        const float *pixel_gt_data,
                        float *pixel_weight_data,
                        float *offset_weight_data)
{
    if (DEBUG) {
        printf("[PixelCluster::init] start init\n");
    }
    non_zeros_map_tensor_ =  non_zero_map_data;
    pixel_pred_tensor_ = pixel_pred_data;
    offset_pred_tensor_ = offset_pred_data;
    pixel_gt_tensor_ = pixel_gt_data;
    pixel_weight_tensor_ = pixel_weight_data;
    offset_weight_tensor_ = offset_weight_data;
    param_.is_training = is_training;
    if (DEBUG) {
        printf("[PixelCluster::intit] is training %d\n", param_.is_training);
        printf("[PixelCluster::init] finisehd init\n");
    }
}

void PixelCluster::ClusterPixelMap()
{
    if (DEBUG) {
        printf("[PixelCluster::ClusterPixelMap] start innit\n");
    }
    std::vector<bool> grid_skips(grids_, true);
    nodes_.resize(grids_);
    Node* nodes = nodes_.data();
    for (int row = 0; row < rows_; ++row) {
        for (int col = 0; col < cols_; ++col) {
            int grid = row * rows_ + col;
            // if training, only non zero pixel will be skip
            grid_skips[grid] = (non_zeros_map_tensor_[grid] <= 0.0001) ||  //  than 0.1 means equal to zero
                    (!param_.is_training &&
                            (pixel_pred_tensor_[grid] < param_.pixel_threshold));
            int center_row = offset_pred_tensor_[grid];
            int center_col = offset_pred_tensor_[grid + grids_];
            center_row = row - center_row;
            center_col = col - center_col;
            center_row = std::min(std::max(center_row, 0), rows_ - 1);
            center_col = std::min(std::max(center_col, 0), cols_ - 1); // to its center
            nodes[grid] =  Node{.parent=static_cast<unsigned>(center_row * cols_ + center_col),
                                .rank = 0U, .traversed = 0U};
        }
    }
    if (DEBUG) {
        printf("[PixelCluster::ClusterPixelMap] traversed row pixel, if not skip\n");
    }
    // start to travered all path
    std::vector<int> traversed_path(grids_, false);
    root_grids_.reserve(grids_);
    root_grids_.clear();
    for (int grid = 0; grid < grids_; ++grid) {
        if (!grid_skips[grid] && nodes_[grid].traversed == 0) {
            traverse_node(nodes, grid, traversed_path.data());
        }
    }
    if (DEBUG) {
        printf("[PixelCluster::ClusterPixelMap] traverse_node finished\n");
    }
    for(int grid : root_grids_) {
        int row = grid / cols_;
        int col = grid % cols_;
        int min_row = std::max(row - 1, 0);
        int max_row = std::min(row + 1, rows_ - 1);
        int min_col = std::max(col - 1, 0);
        int max_col = std::min(col + 1, cols_ - 1);
        for (int row2 = min_row; row2 <= max_row; ++row2) {
            for (int col2 = min_col; col2 <= max_col; ++col2) {
                int grid2 = row2 * cols_ + col2;
                if ((grid2 != grid) && (nodes_[grid2].traversed == 3)) {
                    utils::disjoint_set_union(nodes, grid, grid2);
                }
            }
        }
    }
    if (DEBUG) {
        printf("[PixelCluster::ClusterPixelMap] disjoint_set_union finished\n");
    }
    grid_to_clusters_.resize(grids_);
    grid_next_grids_.resize(grids_);
    for (int grid : root_grids_) {
        grid_to_clusters_[grid] = -1;
    }
    clusters_.clear();
    for (int row = 0; row < rows_; ++row) {
        for (int col = 0; col < cols_; ++col) {
            int grid = row * cols_ + col;
            if (grid_skips[grid]) {
                continue;
            }
            int root = utils::disjoint_set_find(nodes, grid);
            if (grid_to_clusters_[root] < 0) {
                grid_to_clusters_[root] = (int)clusters_.size();
                clusters_.emplace_back();
            }
            Cluster* clu =  &clusters_[grid_to_clusters_[root]];
            grid_next_grids_[grid] = clu->first_grid;
            clu->first_grid = grid;
        }
    }
    if (DEBUG) {
        printf("[PixelCluster::ClusterPixelMap] finished init\n");
    }
}

void PixelCluster::traverse_node(Node* nodes, int x, int* path)
{
    int path_len = 0;
    while (nodes[x].traversed == 0) {
        path[path_len++] =  x;
        nodes[x].traversed = 1;
        x = nodes[x].parent;
    }
    int root;
    if (nodes[x].traversed == 1) {
        int i = path_len;
        int y;
        do {
            i--;
            y = path[i];
            nodes[y] = Node{.parent=(unsigned)x, .rank = 0U, .traversed = 3U};
            root_grids_.push_back(y);
        } while (i >= 1 && y != x);
        root = x;
        path_len = i;
    } else {
        root = nodes[x].parent;
    }
    for (int i = 0; i < path_len; i++) {
        nodes[path[i]] =
            Node{.parent = (unsigned)root, .rank = 0U, .traversed = 2U};
    }
}

void testPixelClusterInputAndOutPut(bool is_training,
                                    at::Tensor non_zero_map,
                                    at::Tensor pixel_pred,
                                    at::Tensor offset_pred,
                                    at::Tensor pixel_gt,
                                    at::Tensor pixel_weight,
                                    at::Tensor offset_weight)
{
    CHECK_CONTIGUOUS(non_zero_map);
    CHECK_CONTIGUOUS(pixel_pred);
    CHECK_CONTIGUOUS(offset_pred);
    CHECK_CONTIGUOUS(pixel_weight);
    CHECK_CONTIGUOUS(offset_weight);
    CHECK_CONTIGUOUS(pixel_gt);
    if (DEBUG) {
        printf("[INFO] testPixelClusterInputAndOutput is_training: %d \n", is_training);
    }
    int batch_size = non_zero_map.size(0);
    int row = non_zero_map.size(1);
    int col = non_zero_map.size(2);

    const float* non_zeros_map_data = non_zero_map.data<float>();
    const float* pixel_pred_data = pixel_pred.data<float>();
    const float* offset_pred_data = offset_pred.data<float>();
    const float* pixel_gt_data = pixel_gt.data<float>();
    float* pixel_weight_data = pixel_weight.data<float>();
    float* offset_weight_data = offset_weight.data<float>();

    int oneMapLen = row * col;
    if (DEBUG) {
        printf("[INFO] testPixelClusterInputAndOutput oneMapLen: %d \n", oneMapLen);
    }
    std::vector<std::thread> threads(batch_size);
    for (int idx = 0; idx < batch_size; ++idx) {
        const float* non_zeros_map_data_pos = non_zeros_map_data + idx * oneMapLen;
        const float* offset_pred_data_pos = offset_pred_data + idx * oneMapLen * 2;
        const float* pixel_pred_data_pos =  pixel_pred_data + idx * oneMapLen;
        const float* pixel_gt_data_pos =  pixel_gt_data + idx * oneMapLen;
        float* pixel_weight_data_pos = pixel_weight_data + idx * oneMapLen;
        float* offset_weight_data_pos =  offset_weight_data + idx * oneMapLen * 2;
        threads[idx] =  std::thread(&processBatchData,
                                    is_training,
                                    non_zeros_map_data_pos,
                                    pixel_pred_data_pos,
                                    offset_pred_data_pos,
                                    pixel_gt_data_pos,
                                    pixel_weight_data_pos,
                                    offset_weight_data_pos);
    }
    for (auto it = threads.begin(); it != threads.end(); ++it) {
        it->join();
    }
    if (DEBUG) {
        printf("[CPP][testPixelClusterInputAndOutput]: non_zeros_maps size: [%d, %d, %d]\n",
            non_zero_map.size(0), non_zero_map.size(1), non_zero_map.size(2));
    }
}

void processBatchData(bool is_training,
                      const float* non_zero_map_data,
                      const float* pixel_pred_data,
                      const float* offset_pred_data,
                      const float* pixel_gt_data,
                      float* pixel_weight_data,
                      float* offset_weight_data)
{
    PixelCluster pixelClu;
    pixelClu.init(is_training,
                  non_zero_map_data,
                  pixel_pred_data,
                  offset_pred_data,
                  pixel_gt_data,
                  pixel_weight_data,
                  offset_weight_data);
    pixelClu.ClusterPixelMap();
    pixelClu.BuildCluster();
    pixelClu.Build_pixel_weight();
}

void PixelCluster::BuildCluster()
{
    for(int idx = 0; idx < static_cast<size_t>(clusters_.size()); ++idx) {
        Cluster* clu =  &clusters_[idx];
        std::vector<int> candidate_grids;
        candidate_grids.clear();
        for (int grid = clu->first_grid; grid > 0; grid = grid_next_grids_[grid]) {
            candidate_grids.emplace_back(grid);
        }
        clu->candidate_grids = candidate_grids;
        // printf("[INFO] PixelCluster::BuildCluster clu->canditata_grid has %s grid\n", clu->canditate_grid.size());
        if (!param_.is_training) {
            clu->grids = std::move(clu->candidate_grids);
        } else {
            std::vector<int> grids;
            grids.clear();
            for (int grid : clu->candidate_grids) {
                if (pixel_pred_tensor_[grid] >= param_.pixel_threshold) {
                    grids.push_back(grid);
                }
            }
            clu->grids = grids;
        }
        Eigen::Vector2f center = {0.0F, 0.0F};
        int root = utils::disjoint_set_find(nodes_.data(), clu->first_grid);
        project(root, clu->center.x(), clu->center.y());
    }
    if (DEBUG) {
        printf("[INFO] current frame has %d cluster \n", clusters_.size());
    }
}

void PixelCluster::project(int grid, float &x, float &y)
{
    int row = grid / cols_;
    int col =  grid % cols_;
    x = (static_cast<float>(row) + 0.5) * resolution_ + range_x_min_;
    y = (static_cast<float>(col) + 0.5) * resolution_ + range_y_min_;
}

void PixelCluster::Build_pixel_weight()
{
    for (int i = 0; i < static_cast<int>(clusters_.size()); ++i) {
        const auto& grids = clusters_[i].candidate_grids;
        bool has_objectness = std::any_of(grids.begin(), grids.end(), [&](int i){
            // printf("[INFO] pixel_gt_tensor[%d] ==> %f\n", i, pixel_gt_tensor_[i]);
            return pixel_gt_tensor_[i] > 0.1F;
        });
        if (has_objectness) {
            for (int i : grids) {
                pixel_weight_tensor_[i] = 1.0F;
                // printf("[INFO] PixelCluster::Build_pixel_weight grid [%d, %d] has weight\n, i / cols_, i % cols_");
            }
        }
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forwardSayHello", &testPixelCluster, "test pixel cluster class");
    m.def("forward", &testPixelClusterInputAndOutPut, "test pixel cluster class input and output");
}
