#include <pybind11/pybind11.h>
#include <algorithm>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <thread>

#include "utils.h"

#define DEBUG false

#define CHECK_CONTIGUOUS(x) do { \
if (!x.is_contiguous()) {        \
    fprintf(stderr, "%s must be contiguous tensor at %s:%d\n",#x, __FILE__, __LINE__); \
    exit(-1);                    \
    }                            \
} while (0)

namespace py = pybind11;
using namespace pybind11::literals;

#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x, " must be contiguous ")

struct param {
    int rows =  256;
    int cols =  256;
    float pixel_threshold = 0.5;
    int max_num_proposals =  128;
    bool is_training = false;
};

struct Cluster {
    std::vector<int> grids;
    std::vector<int> candidate_grids;
    std::vector<int> voxels;
    float confidence_prob = 0.0f;
    Eigen::Vector2f center;
    Eigen::Vector2f velocity;
    Eigen::Vector2f proposals_crop_center_;

    int first_grid = -1;
};

class PixelCluster
{
public:
    PixelCluster()
    {

    }

    void sayHello()
    {
        std::cout << "Say Hello ======================> from pybind11\n";
    }

    void init(bool is_training,
              const float* non_zero_map_data,
              const float* pixel_pred_data,
              const float* offset_pred_data,
              const float* pixel_gt_data,
              float* pixel_weight_data,
              float* offset_weight_data);

    void ClusterPixelMap();
    void BuildCluster();
    void Build_pixel_weight();

private:
    struct Node {
        unsigned int parent : 24;
        unsigned int rank : 6;
        unsigned int traversed : 2;
    };

private:
    param param_;
    int rows_ = param_.rows;
    int cols_ = param_.cols;
    int grids_ = rows_ * cols_;

    std::vector<Node> nodes_;
    std::vector<int> root_grids_;
    std::vector<int> grid_to_clusters_;
    std::vector<int> grid_next_grids_;

    std::vector<Cluster> clusters_;
    std::vector<Cluster*> proposals_;

private:
    const float* non_zeros_map_tensor_;
    const float* pixel_pred_tensor_;
    const float* offset_pred_tensor_;
    const float* pixel_gt_tensor_;
    float* pixel_weight_tensor_;
    float* offset_weight_tensor_;

private:
    const float range_x_min_ = -32.0F;
    const float range_x_max_ = 32.0F;
    const float range_y_min_ = -32.0F;
    const float range_y_max_ = 32.0F;
    float resolution_ = (range_x_max_ - range_x_min_) / rows_;

private:
    void project(int grid, float& x, float& y);
    void traverse_node(Node* nodes, int x, int* path);
};
