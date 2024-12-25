import torch
import torch.nn.functional as F

import numpy as np
import os, argparse
import imageio
import time
from model.SggNet_models import SggNet
from data import test_dataset
import cv2

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=288, help='testing size')
opt = parser.parse_args()

dataset_path = '/share/home/jphe/dataset/'
# dataset_path = '/media/pc/1CAC3A59AC3A2E20/hjp/Camouflaged Object Detection/'

model = SggNet(None)
ckpt='/share/home/jphe/project/SggNet/srun/results/res/SggNet_95_0.2468_0.1502_EORSSD.pth'
print(ckpt)
model.load_state_dict(torch.load(ckpt))
model.cuda()
model.eval()

test_datasets = ['EORSSD','ORSSD']

for dataset in test_datasets:
    save_path = './models/SggNet/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/test-images/'
    print(dataset)
    gt_root = dataset_path + dataset + '/test-labels/'
    test_loader = test_dataset(image_root, gt_root, opt.testsize)
    time_sum = 0
    for i in range(test_loader.size):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        time_start = time.time()
        res, s34, s5, s12_sig, s34_sig, s5_sig,edge= model(image)
        time_end = time.time()
        time_sum = time_sum+(time_end-time_start)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        res =(res * 255).astype(np.uint8)

        # o = cv2.resize(edge, (gt.shape[1], gt.shape[0])) # [H,W]
        # o = (o - o.min()) / (o.max() - o.min() + 1e-8)
        # res = (o * 255).astype(np.uint8)

        imageio.imsave(save_path+name, res)
        if i == test_loader.size-1:
            print('Running time {:.5f}'.format(time_sum/test_loader.size))
            print('FPS {:.5f}'.format(test_loader.size / time_sum))

