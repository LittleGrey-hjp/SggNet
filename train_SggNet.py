import torch
from torch.autograd import Variable
import os
import argparse
from datetime import datetime
from data import get_loader
from model.SggNet_models import SggNet
from utils import clip_gradient, adjust_lr
import pytorch_iou

torch.cuda.set_device(0)
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=100, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
parser.add_argument('--batchsize', type=int, default=8, help='training batch size')
parser.add_argument('--trainsize', type=int, default=288, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
parser.add_argument('--path', type=str, help='path to train dataset')
parser.add_argument('--pretrain', type=str, help='path to pretrain model')

opt = parser.parse_args()

# build models
model = SggNet(opt.pretrain)
model.cuda()
params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)
# encoder_param = []
# decoer_param = []
# for name, param in model.named_parameters():
#     if "backbone" in name:
#         encoder_param.append(param)
#     else:
#         decoer_param.append(param)
# optimizer = torch.optim.Adam(
#     [{"params": encoder_param, "lr": opt.lr * 0.1}, {"params": decoer_param, "lr": opt.lr}])

#
image_root = opt.path + 'images/'
gt_root = opt.path + 'labels/'
edge_root = opt.path + 'edges/'

train_loader = get_loader(image_root, gt_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)

CE = torch.nn.BCEWithLogitsLoss()
MSE = torch.nn.MSELoss()
IOU = pytorch_iou.IOU(size_average = True)


def train(train_loader, model, optimizer, epoch):
    model.train()
    loss_all, loss_1_all=0, 0
    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()
        images, gts, edges = pack
        images = Variable(images)
        gts = Variable(gts)
        edges = Variable(edges)
        images = images.cuda()
        gts = gts.cuda()
        edges = edges.cuda()

        s12, s34, s5, s12_sig, s34_sig, s5_sig, edge_pre = model(images)
        # s12, s34, s5, s12_sig, s34_sig, s5_sig = model(images)

        loss1 = CE(s12, gts) + IOU(s12_sig, gts)
        loss2 = CE(s34, gts) + IOU(s34_sig, gts)
        loss3 = CE(s5, gts) + IOU(s5_sig, gts)
        loss4 = CE(edge_pre, edges)

        loss = loss1 + loss2 / 4 + loss3 / 8 + loss4
        # loss = loss1 + loss2 / 4 + loss3 / 8

        loss.backward()

        clip_gradient(optimizer, opt.clip)
        optimizer.step()

        loss_all += loss
        loss_1_all += loss1

        if i % 20 == 0 or i == total_step:
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Learning Rate: {}, Loss: {:.4f}, Loss1: {:.4f}, Loss2: {:.4f}, Loss3: {:.4f}, Loss4: {:.4f}'.
                    format(datetime.now(), epoch, opt.epoch, i, total_step,
                           opt.lr * opt.decay_rate ** (epoch // opt.decay_epoch), loss.data, loss1.data,
                           loss2.data/4, loss3.data/8, loss4.data))


    save_path = './results/ckpt_save/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 2 == 0:
        torch.save(model.state_dict(), save_path + 'SggNet_{}_{:.4f}_{:.4f}.pth'.format(epoch, loss_all / total_step,
                                                                                        loss_1_all / total_step)
                   , _use_new_zipfile_serialization=False)


print("Training Bgein")
for epoch in range(1, opt.epoch+1):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model, optimizer, epoch)
