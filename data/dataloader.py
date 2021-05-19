import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import cv2
from os import listdir, walk
from os.path import join
from random import randint
import random
from PIL import Image
from torchvision.transforms import Compose, RandomCrop, ToTensor, ToPILImage, Resize, RandomHorizontalFlip


def random_horizontal_flip(imgs):
    if random.random() < 0.3:
        for i in range(len(imgs)):
            imgs[i] = imgs[i].transpose(Image.FLIP_LEFT_RIGHT)
    return imgs

def random_rotate(imgs):
    if random.random() < 0.3:
        max_angle = 10
        angle = random.random() * 2 * max_angle - max_angle
        # print(angle)
        for i in range(len(imgs)):
            img = np.array(imgs[i])
            w, h = img.shape[:2]
            rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
            img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
            imgs[i] =Image.fromarray(img_rotation)
    return imgs

def CheckImageFile(filename):
    return any(filename.endswith(extention) for extention in ['.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG', '.bmp', '.BMP'])

def ImageTransform(loadSize):
    return Compose([
        Resize(size=loadSize, interpolation=Image.BICUBIC),
        ToTensor(),
    ])

class ErasingData(Dataset):
    def __init__(self, dataRoot, loadSize, training=True):
        super(ErasingData, self).__init__()
        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform(loadSize)
        self.training = training
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        mask = Image.open(self.imageFiles[index].replace('all_images','mask'))
        gt = Image.open(self.imageFiles[index].replace('all_images','all_labels'))
        # import pdb;pdb.set_trace()
        if self.training:
        # ### for data augmentation
            all_input = [img, mask, gt]
            all_input = random_horizontal_flip(all_input)   
            all_input = random_rotate(all_input)
            img = all_input[0]
            mask = all_input[1]
            gt = all_input[2]
        ### for data augmentation
        inputImage = self.ImgTrans(img.convert('RGB'))
        mask = self.ImgTrans(mask.convert('RGB'))
        groundTruth = self.ImgTrans(gt.convert('RGB'))
        path = self.imageFiles[index].split('/')[-1]
       # import pdb;pdb.set_trace()

        return inputImage, groundTruth, mask, path
    
    def __len__(self):
        return len(self.imageFiles)

class devdata(Dataset):
    def __init__(self, dataRoot, gtRoot, loadSize=512):
        super(devdata, self).__init__()
        self.imageFiles = [join (dataRootK, files) for dataRootK, dn, filenames in walk(dataRoot) \
            for files in filenames if CheckImageFile(files)]
        self.gtFiles = [join (gtRootK, files) for gtRootK, dn, filenames in walk(gtRoot) \
            for files in filenames if CheckImageFile(files)]
        self.loadSize = loadSize
        self.ImgTrans = ImageTransform(loadSize)
    
    def __getitem__(self, index):
        img = Image.open(self.imageFiles[index])
        gt = Image.open(self.gtFiles[index])
        #import pdb;pdb.set_trace()
        inputImage = self.ImgTrans(img.convert('RGB'))

        groundTruth = self.ImgTrans(gt.convert('RGB'))
        path = self.imageFiles[index].split('/')[-1]

        return inputImage, groundTruth,path
    
    def __len__(self):
        return len(self.imageFiles)
