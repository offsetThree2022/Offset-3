import logging
import os
import pickle
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

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


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers


def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    # torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank


def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    # torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        init_method='tcp://127.0.0.1:%d' % tcp_port,
        rank=local_rank,
        world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret


def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor


