import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import configs
import time
import sys
import os
from shutil import copytree, copy
from model import OffsetThree
from data.nuscenes_dataloader import TrainDatasetMultiSeq
if "/opt/ros/kinetic/lib/python2.7/dist-packages" in sys.path:
    sys.path.remove("/opt/ros/kinetic/lib/python2.7/dist-packages")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)


def check_folder(folder_path):
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
    return folder_path


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data', default=None, type=str, help='The path to the preprocessed sparse BEV training data')
parser.add_argument('--resume', default='', type=str, help='The path to the saved model that is loaded to resume training')
parser.add_argument('--nepoch', default=45, type=int, help='Number of epochs')

parser.add_argument('--nn_sampling', action='store_true', help='Whether to use nearest neighbor sampling in bg_tc loss')
parser.add_argument('--log', action='store_true', help='Whether to log')
parser.add_argument('--logpath', default='', help='The path to the output log file')

args = parser.parse_args()
print(args)

need_log = configs.train.log
num_epochs = configs.train.num_epochs


def main():
    start_epoch = 1
    # Whether to log the training information
    if need_log:
        logger_root = args.logpath if args.logpath != '' else 'logs'
        time_stamp = time.strftime("%Y-%m-%d_%H-%M-%S")

        if args.resume == '':
            model_save_path = check_folder(logger_root)
            model_save_path = check_folder(os.path.join(model_save_path, 'train_multi_seq'))
            model_save_path = check_folder(os.path.join(model_save_path, time_stamp))

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "w")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[0:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

            # Copy the code files as logs
            copytree('nuscenes-devkit', os.path.join(model_save_path, 'nuscenes-devkit'))
            copytree('data', os.path.join(model_save_path, 'data'))
            python_files = [f for f in os.listdir('.') if f.endswith('.py')]
            for f in python_files:
                copy(f, model_save_path)
        else:
            model_save_path = args.resume  # eg, "logs/train_multi_seq/1234-56-78-11-22-33"

            log_file_name = os.path.join(model_save_path, 'log.txt')
            saver = open(log_file_name, "a")
            saver.write("GPU number: {}\n".format(torch.cuda.device_count()))
            saver.flush()

            # Logging the details for this experiment
            saver.write("command line: {}\n".format(" ".join(sys.argv[1:])))
            saver.write(args.__repr__() + "\n\n")
            saver.flush()

    # Specify gpu device
    devices = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_num = torch.cuda.device_count()
    print("device number", device_num)
    torch.multiprocessing.set_start_method("spawn")
    data_nuscenes = TrainDatasetMultiSeq(devices=devices)
    trainloader = torch.utils.data.DataLoader(data_nuscenes, batch_size=configs.data.batch_size, shuffle=True,
                                              num_workers=configs.data.num_worker,
                                              collate_fn=data_nuscenes.collate_batch)
    print("Training dataset size:", len(data_nuscenes))

    model = OffsetThree(num_feature_channel=configs.bird.num_feature_channel, batch_size=configs.data.batch_size,
                      device=devices, is_training=True)
    model = nn.DataParallel(model)
    model = model.to(devices)

    optimizer = optim.Adam(model.parameters(), lr=0.0016)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 15, 20, 25, 35], gamma=0.5)

    if configs.train.resume_det is not None:
        checkpoint = torch.load(configs.train.resume_det)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        print("Load model from {}, at epoch {}".format(configs.train.resume_det, start_epoch - 1))

    for epoch in range(start_epoch, num_epochs + 1):
        lr = optimizer.param_groups[0]['lr']
        print("Epoch {}, learning rate {}".format(epoch, lr))

        if need_log:
            saver.write("epoch: {}, lr: {}\t".format(epoch, lr))
            saver.flush()

        model.train()
        loss_pixel, loss_offset, loss_confidence, loss_velocity, loss_size, loss_yaw, loss_height, loss_category \
            = train(model, trainloader, optimizer, devices, epoch, scheduler)

        if need_log:
            saver.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t\n"
                        .format(loss_pixel, loss_offset, loss_confidence,
                                loss_velocity, loss_size, loss_yaw, loss_height, loss_category))
            saver.flush()

        # save model
        if need_log and (epoch % 1 == 0 or epoch == num_epochs or epoch == 1 or epoch > 20):
            save_dict = {'epoch': epoch,
                         'model_state_dict': model.state_dict(),
                         'optimizer_state_dict': optimizer.state_dict(),
                         'scheduler_state_dict': scheduler.state_dict(),
                         'loss': loss_velocity.avg}
            torch.save(save_dict, os.path.join(model_save_path, 'epoch_' + str(epoch) + '.pth'))

    if need_log:
        saver.close()


def train(model, trainloader, optimizer, device, epoch, scheduler):
    running_loss_total = AverageMeter('total', ':.6f')
    running_loss_pixel = AverageMeter('pixel', ':.6f')
    running_loss_offset = AverageMeter('offset', ':.6f')
    running_loss_confidence = AverageMeter('conf', ':.6f')
    running_loss_velocity = AverageMeter('vel', ':.6f')
    running_loss_size = AverageMeter('size', ':.6f')
    running_loss_yaw = AverageMeter('yaw', ':.6f')
    running_loss_height = AverageMeter('height', ':.6f')
    running_loss_cat = AverageMeter('cat', ':.6f')

    # torch.cuda.synchronize()
    time_s = time.time()
    for i, data_dict in enumerate(trainloader, 0):
        # torch.cuda.synchronize()
        time_0 = time.time()
        total_loss, detect_loss_dict = forward_and_bp_loss(model, optimizer, data_dict, scheduler)
        # torch.cuda.synchronize()
        # print('[TIME] forward_and_bp_loss: {}'.format((time.time() - time_0) * 1000))
        loss_pixel = detect_loss_dict['loss_pixel']
        loss_offset = detect_loss_dict['loss_offset']
        loss_confidence = detect_loss_dict['loss_confidence']
        loss_velocity = detect_loss_dict['loss_velocity']
        loss_size = detect_loss_dict['loss_size']
        loss_yaw = detect_loss_dict['loss_yaw']
        loss_height = detect_loss_dict['loss_height']
        loss_category = detect_loss_dict['loss_category']

        # time_loss = (time.time() - time_loss_s) * 1000
        if not all((loss_pixel, loss_offset, loss_confidence, loss_velocity)):
            print("{}, \t{}, \tat epoch {}, \titerations {} [empty occupy map]".
                  format(loss_pixel, loss_velocity, epoch, i))
            continue

        running_loss_pixel.update(loss_pixel)
        running_loss_offset.update(loss_offset)
        running_loss_confidence.update(loss_confidence)
        running_loss_velocity.update(loss_velocity)
        running_loss_size.update(loss_size)
        running_loss_yaw.update(loss_yaw)
        running_loss_height.update(loss_height)
        running_loss_cat.update(loss_category)
        running_loss_total.update(total_loss)

        print("[{}/{}]\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.2f} ms"
              .format(epoch, i, running_loss_total, running_loss_pixel, running_loss_offset, running_loss_confidence,
                      running_loss_velocity, running_loss_size, running_loss_yaw, running_loss_height,
                      running_loss_cat, (time.time() - time_s) * 1000 / (i + 1)))
        # torch.cuda.synchronize()
        # print('[TIME] finished one iter: {}'.format((time.time() - time_0) * 1000))
        # print('[TIME]                                                                           ')

    return running_loss_pixel, running_loss_offset, running_loss_confidence, \
           running_loss_velocity, running_loss_size, running_loss_yaw, running_loss_height, running_loss_cat


def compute_box_loss():
    """
        compute box relative loss
        assign predict box to target box by iou
        get predict box, cls ect
    """
    #  get predict box
    pass


#  Compute and back-propagate the loss
def forward_and_bp_loss(model, optimizer, data_dict, scheduler):
    optimizer.zero_grad()
    # torch.cuda.synchronize()
    time_0 = time.time()
    detect_loss, detect_loss_dict, _ = model(data_dict)
    # torch.cuda.synchronize()
    # print('[TIME] time info has cost {}'.format(1000.0 * (time.time() - time_0)))
    time_1 = time.time()
    detect_loss.backward()
    # torch.cuda.synchronize()
    # print('[TIME] back up has cost {}'.format(1000.0 * (time.time() - time_1)))
    optimizer.step()
    scheduler.step()
    return detect_loss.item(), detect_loss_dict


if __name__ == "__main__":
    # os.environ['CUDA_LAUNCH_pixel'] = '1' for debug
    main()
