import os
from random import random
import sys

import numpy as np
from tqdm import tqdm
from scipy import misc
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
import matplotlib.pyplot as plt
sys.path.append(os.getcwd())
from utils import imutils

def load_txt(txt_name):
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]
        return name_list


class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, txt_dir=None, split='train', stage='train', crop_size=None, scales=None, mean_bgr=None,):
        # super()
        # stage: train, val, test
        self.root_dir = root_dir
        self.txt_name = os.path.join(txt_dir, split) + '.txt'
        self.name_list = load_txt(self.txt_name)
        self.stage = stage
        self.crop_size = crop_size
        self.scales = scales
        self.mean_bgr = mean_bgr
        self.img_transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        image_path = os.path.join(self.root_dir, 'JPEGImages', self.name_list[idx]+'.jpg')
        mask_path = os.path.join(self.root_dir, 'SegmentationClassAug', self.name_list[idx]+'.png')

        image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)

        if self.stage is 'train':
            #image = cv2.imread(image_path, cv2.IMREAD_COLOR)#.astype(np.float32)
            mask = misc.imread(mask_path)

            image, mask = self.joint_transforms(image, mask)

            image = image - self.mean_bgr
            image = self.img_transforms(image).float()

            return self.name_list[idx], image, mask

        elif self.stage is 'val':
            #image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
            mask = misc.imread(mask_path).astype(np.float32)

            image = image - self.mean_bgr
            image = self.img_transforms(image).float()
            
            return self.name_list[idx], image, mask

        elif self.stage is 'test':
            
            image = image - self.mean_bgr
            image = self.img_transforms(image).float()
            mask = None
            return self.name_list[idx], image, mask
        
    def joint_transforms(self, image, mask):

        image, mask = imutils.random_scaling(image, mask, scales=self.scales)
        #image, mask = imutils.random_flipud(image, mask)
        image, mask = imutils.random_fliplr(image, mask)
        image, mask = imutils.random_crop(image, mask, crop_size=self.crop_size)
        #image, mask = imutils.random_rot(image, mask)
        
        return image, mask

if __name__ == "__main__":
    root_dir = '/data/users/rulixiang/VOCdevkit/VOC2012'
    txt_file = '/data/users/rulixiang/deeplab-pytorch/dataset/voc/train.txt'
    voc12dataset = VOCSegmentationDataset(root_dir, txt_dir = 'dataset/voc', stage='train', crop_size=321, scales=[0.5, 0.75, 1.0, 1.25, 1.5], mean_bgr=[122.675, 116.669, 104.008],)
    loader = DataLoader(voc12dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, batch in tqdm(enumerate(loader),total=len(loader)):
        print(i)
