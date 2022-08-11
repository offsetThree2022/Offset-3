import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
import sys
import os
from shutil import copytree, copy

import tqdm

from model import OffsetThree
import copy
from data.nuscenes_dataloader import TrainDatasetMultiSeq
import matplotlib.pyplot as plt

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Box

import numpy as np
import configs

import pickle
from tqdm import tqdm


from data.datasets.nuscenes import NuScenesDataset

if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")

resume_det = "/media/user/C14D581BDA18EBFA/code/OffsetThree/logs/train_multi_seq/box_512_baseline/epoch_12.pth"

devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_num = torch.cuda.device_count()
# print("device number", device_num)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a detector")
    # parser.add_argument("config", help="train config file path")
    parser.add_argument("--work_dir", required=True, help="the dir to save logs and models")
    parser.add_argument("--gen_root_dir", required=True, help="gen data dir")
    parser.add_argument(
        "--checkpoint", help="the dir to checkpoint which the model read from"
    )
    parser.add_argument(
        "--txt_result",
        type=bool,
        default=False,
        help="whether to save results to standard KITTI format of txt type",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=1,
        help="number of gpus to use " "(only applicable to non-distributed training)",
    )
    parser.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm", "mpi"],
        default="none",
        help="job launcher",
    )
    parser.add_argument("--speed_test", action="store_true")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--testset", action="store_true")

    args = parser.parse_args()
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = str(args.local_rank)

    return args

def save_pred(pred, root):
    with open(os.path.join(root, "prediction.pkl"), "wb") as f:
        pickle.dump(pred, f)

def main():
    print("Start dist val and test")
    args = parse_args()

    distributed = False
    if "WORLD_SIZE" in os.environ:
        distributed = int(os.environ["WORLD_SIZE"]) > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
    batch_size = 4
    if args.testset:
        print("Use Test Set")
        dataset = TrainDatasetMultiSeq(batch_size=batch_size, devices=devices, tracking=False, split='test')
    else:
        print("Use Val Set")
        dataset = TrainDatasetMultiSeq(batch_size=batch_size, devices=devices, tracking=False, split='val')

    num_devices = torch.cuda.device_count()
    torch.multiprocessing.set_start_method("spawn")
    need_shuffle = False
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=need_shuffle,
                                              num_workers=batch_size, collate_fn=dataset.collate_batch)
    # for tracking set is training for false, when not training with end to end
    model = OffsetThree(num_feature_channel=configs.bird.num_feature_channel, batch_size=batch_size, device=devices,
                      is_training=False, dist_test=True)
    model = nn.DataParallel(model)
    model = model.to(devices)

    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    detections = {}
    cpu_device = torch.device("cpu")

    torch.cuda.synchronize()

    start = int(len(dataset) / 3)
    end = int(len(dataset) * 2 / 3)

    time_start = 0
    time_end = 0

    # for i, data_batch in enumerate(data_loader):
    i = 0
    for data_batch in tqdm(data_loader):
        if i == start:
            torch.cuda.synchronize()
            time_start = time.time()
        if i == end:
            torch.cuda.synchronize()
            time_end = time.time()

        with torch.no_grad():
            outputs_dict = model(data_batch)

        outputs = outputs_dict["output"]
        for output in outputs:
            token = output["token"]
            for k, v in output.items():
                if k not in [
                    "token",
                ]:
                    output[k] = v.to(cpu_device)
            detections.update(
                {token: output}
            )
        i += 1

    # all_predictions = all_gather(detections) when multi gpu test

    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    save_pred(detections, args.work_dir)

    result_dict, _ = dataset.evaluation(copy.deepcopy(detections), output_dir=args.work_dir,
                                        gen_root_dir=args.gen_root_dir, testset=args.testset)

    if result_dict is not None:
        for k, v in result_dict["results"].items():
            print(f"Evaluation {k}: {v}")
    print("\n Total time per frame: ", (time_end - time_start) / (end - start))
    torch.cuda.synchronize()


if __name__ == '__main__':
    main()
















