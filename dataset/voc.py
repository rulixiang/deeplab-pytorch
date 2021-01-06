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
        #self.img_transforms = transforms.Compose([transforms.ToTensor(),])

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, idx):

        image_path = os.path.join(self.root_dir, 'JPEGImages', self.name_list[idx]+'.jpg')
        mask_path = os.path.join(self.root_dir, 'SegmentationClassAug', self.name_list[idx]+'.png')

        image = misc.imread(image_path).astype(np.float32)
        # convert to bgr
        image = image[:,:,[2,1,0]]

        if self.stage is 'train':

            mask = misc.imread(mask_path)
            image, mask = self.joint_transforms(image, mask)

        elif self.stage is 'val':
            #image = cv2.imread(image_path, cv2.IMREAD_COLOR).astype(np.float32)
            mask = misc.imread(mask_path).astype(np.float32)

        elif self.stage is 'test':

            mask = None

        image[:,:,0] = image[:,:,0] - self.mean_bgr[0]
        image[:,:,1] = image[:,:,1] - self.mean_bgr[1]
        image[:,:,2] = image[:,:,2] - self.mean_bgr[2]
        image = image.transpose([2,0,1])

        return self.name_list[idx], image, mask
        
    def joint_transforms(self, image, mask):

        image, mask = imutils.random_scaling(image, mask, scales=self.scales)
        #image, mask = imutils.random_flipud(image, mask)
        image, mask = imutils.random_fliplr(image, mask)
        image, mask = imutils.random_crop(image, mask, crop_size=self.crop_size, mean_bgr=self.mean_bgr)
        #image, mask = imutils.random_rot(image, mask)
        
        return image, mask

if __name__ == "__main__":
    root_dir = '/home/rlx/VOCdevkit/VOC2012'
    txt_file = '/home/rlx/deeplab-pytorch/dataset/voc/train.txt'
    voc12dataset = VOCSegmentationDataset(root_dir, txt_dir = 'dataset/voc', stage='train', crop_size=321, scales=[0.5, 0.75, 1.0, 1.25, 1.5], mean_bgr=[104.008, 116.669, 122.675],)
    loader = DataLoader(voc12dataset, batch_size=4, shuffle=True, num_workers=4)
    for i, batch in tqdm(enumerate(loader),total=len(loader)):
        print(i)
