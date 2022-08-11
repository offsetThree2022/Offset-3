import random
import math
import numpy as np
from random import randint

dataDir = "/media/user/C14D581BDA18EBFA/kitti/ImageSets"
train = []
val_orin = []
with open(dataDir + "/train.txt") as f:
    lines = f.readlines()
    for line in lines:
        train.append(line)
        if (len(line) > 7):
            print("XXXXXXXXXXXXXXXXXXXXXX" + str(line))

with open(dataDir + "/val.txt") as f:
    lines = f.readlines()
    for line in lines:
        val_orin.append(line)
        if (len(line) > 7):
            print("XXXXXXXXXXXXXXXXXXXXXX" + str(line))
# int_rand = [i for i in range(len(val_orin))]
# random.shuffle(int_rand)
# random.shuffle(val_orin)
# val_add = val_orin[:math.floor(0.6 * len(val_orin))]
# train_new = train + val_add
# val_new = val_orin[math.floor(0.6 * len(val_orin)):]
# with open(dataDir + "/train_new.txt", 'w') as f:
#     for idx in train_new:
#         f.write(idx)
#
# with open(dataDir + "/val_new.txt", 'w') as f:
#     for idx in val_new:
#         f.write(idx)
# print(dataDir)

