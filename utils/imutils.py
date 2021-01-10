
import random
import numpy as np
from PIL import Image
#from scipy import misc
import torch
import torchvision
import cv2

def random_scaling(image, mask, scales=None):
    scale = random.choice(scales)
    h, w, = mask.shape
    new_scale = [int(scale * w), int(scale * h)]
    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_mask = Image.fromarray(mask).resize(new_scale, resample=Image.NEAREST)
    new_image = np.asarray(new_image).astype(np.float32)
    new_mask = np.asarray(new_mask)
    return new_image, new_mask

def random_fliplr(image, mask):
    if random.random() > 0.5:
        mask = np.fliplr(mask)
        image = np.fliplr(image)

    return image, mask

def random_flipud(image, mask):

    if random.random() > 0.5:
        mask = np.flipud(mask)
        image = np.flipud(image)

    return image, mask

def random_rot(image, mask):

    k = random.randrange(3) + 1

    image = np.rot90(image, k).copy()
    mask = np.rot90(mask, k).copy()

    return image, mask

def random_crop(image, mask, crop_size, mean_bgr):

    h, w = mask.shape
    H = max(crop_size, h)
    W = max(crop_size, w)
    pad_image = np.zeros((H,W,3), dtype=np.float32)
    pad_mask = np.ones((H,W), dtype=np.float32)*255

    pad_image[:,:,0] = mean_bgr[0]
    pad_image[:,:,1] = mean_bgr[1]
    pad_image[:,:,2] = mean_bgr[2]
    
    H_pad = int(np.floor(H-h))
    W_pad = int(np.floor(W-w))
    pad_image[H_pad:(H_pad+h), W_pad:(W_pad+w), :] = image
    pad_mask[H_pad:(H_pad+h), W_pad:(W_pad+w)] = mask

    H_start = random.randrange(H - crop_size + 1)
    W_start = random.randrange(W - crop_size + 1)

    #print(W_start)

    image = pad_image[H_start:(H_start+crop_size), W_start:(W_start+crop_size),:]
    mask = pad_mask[H_start:(H_start+crop_size), W_start:(W_start+crop_size)]

    #cmap = colormap()
    #misc.imsave('cropimg.png',image/255)
    #misc.imsave('cropmask.png',encode_cmap(mask))
    return image, mask

def encode_cmap(label):
    cmap = colormap()
    return cmap[label.astype(np.int16),:]

def tensorboard_image(inputs=None, outputs=None, labels=None, bgr=None):
    ## images
    inputs[:,0,:,:] = inputs[:,0,:,:] + bgr[0]
    inputs[:,1,:,:] = inputs[:,1,:,:] + bgr[1]
    inputs[:,2,:,:] = inputs[:,2,:,:] + bgr[2]
    inputs = inputs[:,[2,1,0],:,:].type(torch.uint8)
    grid_inputs = torchvision.utils.make_grid(tensor=inputs, nrow=2)

    ## preds
    preds = torch.argmax(outputs, dim=1).cpu().numpy()
    preds_cmap = encode_cmap(preds)
    preds_cmap = torch.from_numpy(preds_cmap).permute([0, 3, 1, 2])
    grid_outputs = torchvision.utils.make_grid(tensor=preds_cmap, nrow=2)

    ## labels
    labels_cmap = encode_cmap(labels.cpu().numpy())
    labels_cmap = torch.from_numpy(labels_cmap).permute([0, 3, 1, 2])
    grid_labels = torchvision.utils.make_grid(tensor=labels_cmap, nrow=2)

    return grid_inputs, grid_outputs, grid_labels

def colormap(N=256, normalized=False):
    def bitget(byteval, idx):
        return ((byteval & (1 << idx)) != 0)

    dtype = 'float32' if normalized else 'uint8'
    cmap = np.zeros((N, 3), dtype=dtype)
    for i in range(N):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7-j)
            g = g | (bitget(c, 1) << 7-j)
            b = b | (bitget(c, 2) << 7-j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    cmap = cmap/255 if normalized else cmap
    return cmap