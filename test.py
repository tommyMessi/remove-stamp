import os
import math
import argparse
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from PIL import Image
import numpy as np
from torch.autograd import Variable
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from data.dataloader import ErasingData
from models.sa_gan import STRnet2


parser = argparse.ArgumentParser()
parser.add_argument('--numOfWorkers', type=int, default=0,
                    help='workers for dataloader')
parser.add_argument('--modelsSavePath', type=str, default='',
                    help='path for saving models')
parser.add_argument('--logPath', type=str,
                    default='')
parser.add_argument('--batchSize', type=int, default=16)
parser.add_argument('--loadSize', type=int, default=512,
                    help='image loading size')
parser.add_argument('--dataRoot', type=str,
                    default='')
parser.add_argument('--pretrained',type=str, default='', help='pretrained models for finetuning')
parser.add_argument('--savePath', type=str, default='./results/sn_tv/')
args = parser.parse_args()

cuda = torch.cuda.is_available()
if cuda:
    print('Cuda is available!')
    cudnn.benchmark = True


def visual(image):
    im =(image).transpose(1,2).transpose(2,3).detach().cpu().numpy()
    Image.fromarray(im[0].astype(np.uint8)).show()

batchSize = args.batchSize
loadSize = (args.loadSize, args.loadSize)
dataRoot = args.dataRoot
savePath = args.savePath
result_with_mask = savePath + 'WithMaskOutput/'
result_straight = savePath + 'StrOuput/'
#import pdb;pdb.set_trace()

if not os.path.exists(savePath):
    os.makedirs(savePath)
    os.makedirs(result_with_mask)
    os.makedirs(result_straight)


Erase_data = ErasingData(dataRoot, loadSize, training=False)
Erase_data = DataLoader(Erase_data, batch_size=batchSize, shuffle=True, num_workers=args.numOfWorkers, drop_last=False)


netG = STRnet2(3)

netG.load_state_dict(torch.load(args.pretrained))

#
if cuda:
    netG = netG.cuda()

for param in netG.parameters():
    param.requires_grad = False

print('OK!')

import time
start = time.time()
netG.eval()
for imgs, gt, masks, path in (Erase_data):
    if cuda:
        imgs = imgs.cuda()
        gt = gt.cuda()
        masks = masks.cuda()
    out1, out2, out3, g_images,mm = netG(imgs)
    g_image = g_images.data.cpu()
    gt = gt.data.cpu()
    mask = masks.data.cpu()
    g_image_with_mask = gt * (mask) + g_image * (1- mask)

    save_image(g_image_with_mask, result_with_mask+path[0])
    save_image(g_image, result_straight+path[0])




