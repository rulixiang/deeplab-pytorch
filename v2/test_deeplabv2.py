
import argparse
import json
import os
import sys
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='1', type=str, help="gpu")
parser.add_argument("--config", default='./config/deeplabv2_voc12.yaml', type=str, help="config")
parser.add_argument("--crf", default=True, type=bool, help="use crf post processing")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, 'models'))
sys.path.append(os.path.join(current_path, 'dataset'))
sys.path.append(os.path.join(current_path, 'utils'))

import DeepLabV2_ResNet101
import numpy as np
import torch
from torch import mean, multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import voc
from omegaconf import OmegaConf
from scipy import misc
from tqdm import tqdm
from utils import imutils, pyutils, crf

def makedirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
    return True

def test(config=None):

    makedirs(os.path.join(config.exp.path, config.exp.results))
    makedirs(os.path.join(config.exp.path, config.exp.preds))
    makedirs(os.path.join(config.exp.path, config.exp.logits))

    print('Inferring...')

    val_dataset = voc.VOCSegmentationDataset(root_dir=config.dataset.root_dir, txt_dir=config.dataset.txt_dir, split=config.val.split, stage='val', mean_bgr=config.dataset.mean_bgr,)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=config.train.num_workers, pin_memory=True, drop_last=False)

    # device
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available() is True:
        device = torch.device('cuda')
        print('%d GPUs are available:'%(torch.cuda.device_count()))
        for k in range(torch.cuda.device_count()):
            print('    %s: %s'%(args.gpu.split(',')[k], torch.cuda.get_device_name(k)))
    else:
        print('Using CPU:')
        device = torch.device('cpu')

    model = DeepLabV2_ResNet101.DeepLabV2_ResNet101_MSC(n_classes=config.dataset.n_classes, n_blocks=config.model.blocks, atrous_rates=config.model.atrous_rates, scales=config.model.scales)
    model = nn.DataParallel(model)

    checkpoint_path = os.path.join(config.exp.path, config.exp.checkpoint_dir, config.exp.final_weights)
    state_dict = torch.load(checkpoint_path)
    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    preds, gts = [], []

    dst_path_preds = os.path.join(config.exp.path, config.exp.preds)
    dst_path_logits = os.path.join(config.exp.path, config.exp.logits)

    with torch.no_grad():
        for _, data in tqdm(enumerate(val_loader), total=len(val_loader), ascii=' 123456789>'):
            name, inputs, labels = data

            outputs = model(inputs)
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(outputs, size=inputs.shape[2:], mode='bilinear', align_corners=True)

            pred = torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.uint8)
            label = labels.cpu().numpy().astype(np.uint8)

            pred_cmap = imutils.encode_cmap(pred)[0,:]

            misc.imsave(dst_path_preds + '/' + name[0] + '.png', pred_cmap)
            np.save(dst_path_logits+ '/' + name[0] + '.npy', outputs.cpu().numpy())
            
            preds += list(pred)
            gts += list(label)

    score = pyutils.scores(gts, preds)

    torch.cuda.empty_cache()

    json_path = os.path.join(config.exp.path, config.exp.results, config.val.split) + '.json'
    with open(json_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)

    print('Prediction results saved to %s'%(dst_path_preds))
    print('Evaluation results saved to %s, pixel acc is %f, mean IoU is %f'%(json_path, score['Pixel Accuracy'], score['Mean IoU']))
    
    return score

def crf_proc(config):
    print("crf post-processing...")

    txt_name = os.path.join(config.dataset.txt_dir, config.val.split) + '.txt'
    with open(txt_name) as f:
        name_list = [x for x in f.read().split('\n') if x]

    images_path = os.path.join(config.dataset.root_dir, 'JPEGImages',)
    labels_path = os.path.join(config.dataset.root_dir, 'SegmentationClassAug')
    logits_path = os.path.join(config.exp.path, config.exp.logits)
    crf_path = os.path.join(config.exp.path, config.exp.crf)
    mean_bgr = config.dataset.mean_bgr
    makedirs(crf_path)

    post_processor = crf.DenseCRF(
        iter_max=10,    # 10
        pos_xy_std=3,   # 3
        pos_w=3,        # 3
        bi_xy_std=140,  # 121, 140
        bi_rgb_std=5,   # 5, 5
        bi_w=5,         # 4, 5
    )

    def _job(i):

        name = name_list[i]

        logit_name = os.path.join(logits_path, name + ".npy")
        logit = np.load(logit_name)
        image_name = os.path.join(images_path, name + ".jpg")
        image = misc.imread(image_name).astype(np.float32)
        label_name = os.path.join(labels_path, name + ".png")
        label = misc.imread(label_name)

        #image[:,:,0] = image[:,:,0] - mean_bgr[2]
        #image[:,:,1] = image[:,:,1] - mean_bgr[1]
        #image[:,:,2] = image[:,:,2] - mean_bgr[0]
        #image = image[:,:,[2,1,0]]

        H, W, _ = image.shape
        logit = torch.FloatTensor(logit)#[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logit, dim=1)[0].numpy()

        image = image.astype(np.uint8)
        prob = post_processor(image, prob)
        pred = np.argmax(prob, axis=0)

        _pred = np.squeeze(pred).astype(np.uint8)
        _pred_cmap = imutils.encode_cmap(_pred)

        misc.imsave(crf_path+'/'+name+'.png', _pred_cmap)

        return pred, label

    n_jobs = int(multiprocessing.cpu_count() * 0.8)
    results = joblib.Parallel(n_jobs=n_jobs, verbose=10, pre_dispatch="all")([joblib.delayed(_job)(i) for i in range(len(name_list))])

    preds, gts = zip(*results)

    score = pyutils.scores(gts, preds)
    json_path = os.path.join(config.exp.path, config.exp.results, config.val.split) + '_crf.json'
    with open(json_path, "w") as f:
        json.dump(score, f, indent=4, sort_keys=True)
        
    print('Prediction results saved to %s.'%(crf_path))
    print('Evaluation results saved to %s, pixel acc is %f, mean IoU is %f.'%(json_path, score['Pixel Accuracy'], score['Mean IoU']))
    
    return True

if __name__=="__main__":

    config = OmegaConf.load(args.config)
    score = test(config)
    if args.crf is True:
        pass
        crf_score = crf_proc(config)
