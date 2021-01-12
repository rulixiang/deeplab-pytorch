
import argparse
import os
import sys
from datetime import datetime

TIMESTAMP = "{0:%Y-%m-%d-%H-%M-%S/}".format(datetime.now())
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", default='2', type=str, help="gpu")
parser.add_argument("--config", default='./config/deeplabv2_voc12.yaml', type=str, help="config")
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
current_path = os.getcwd()
sys.path.append(os.path.join(current_path, 'models'))
sys.path.append(os.path.join(current_path, 'dataset'))
sys.path.append(os.path.join(current_path, 'utils'))

from collections import OrderedDict

import DeepLabV2_ResNet101
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import voc
from omegaconf import OmegaConf
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import imutils, pyutils


def makedirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)
    return True

def imresize(labels, size):

    new_labels = []
    for label in labels:
        label = label.cpu().float().numpy()
        label = Image.fromarray(label).resize(size, resample=Image.NEAREST)
        new_labels.append(np.asarray(label))
    new_labels = torch.LongTensor(new_labels)

    return new_labels

def get_params(model, key):

    if key == '1x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0]!='conv8':
                    yield m[1].weight
    if key == '2x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0]!='conv8':
                    yield m[1].bias
    if key == '10x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0]=='conv8':
                    yield m[1].weight
    if key == '20x':
        for m in model.named_modules():
            if isinstance(m[1], nn.Conv2d):
                if m[0]=='conv8':
                    yield m[1].bias

def validate(model=None, criterion=None, data_loader=None, writer=None):

    print('Validating...')

    n_gpus = torch.cuda.device_count()

    val_loss = 0.0
    preds, gts = [], []
    model.eval()

    with torch.no_grad():
        for _, data in tqdm(enumerate(data_loader), total=len(data_loader), ascii=' 123456789>'):
            _, inputs, labels = data

            #inputs = inputs.to()
            #labels = labels.to(inputs.device)

            outputs = model(inputs)
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(outputs, size=inputs.shape[2:], mode='bilinear', align_corners=True)

            loss = criterion(resized_outputs, labels)
            val_loss += loss

            preds += list(torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    score = pyutils.scores(gts, preds)

    return val_loss.cpu().numpy() / float(len(data_loader)), score

def train(config=None):
    # loop over the dataset multiple times

    train_dataset = voc.VOCSegmentationDataset(root_dir=config.dataset.root_dir, txt_dir=config.dataset.txt_dir, split=config.train.split, stage='train', crop_size=config.dataset.crop_size, scales=config.train.scales, mean_bgr=config.dataset.mean_bgr,)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=config.train.num_workers, pin_memory=True, drop_last=True)

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

    # build and initialize model
    model = DeepLabV2_ResNet101.DeepLabV2_ResNet101_MSC(n_classes=config.dataset.n_classes, n_blocks=config.model.blocks, atrous_rates=config.model.atrous_rates, scales=config.model.scales)

    # save model to tensorboard 
    writer_path = os.path.join(config.exp.path, config.exp.tensorboard_dir, TIMESTAMP)
    writer = SummaryWriter(writer_path)
    dummy_input = torch.rand(2, 3, 321, 321)
    writer.add_graph(model, dummy_input)

    # load initial weights
    pretrained_dict = torch.load(config.exp.init_weights)
    #new_keys = sorted(list(model.state_dict().keys()))[6:]
    new_state_dict = OrderedDict()

    for item in model.state_dict().keys():
        if 'aspp' not in item:
            prev_key = item.replace('batch_norm', 'bn')
            prev_key = prev_key.replace('conv1', 'reduce')
            prev_key = prev_key.replace('conv3', 'increase')
            prev_key = prev_key.replace('conv2', 'conv3x3')
            prev_key = prev_key.replace('conv7x7', 'conv1')
            prev_key = prev_key.replace('base.', '')
            #print(" %s  <=======  %s"%(item.ljust(56,' '), prev_key.ljust(24, ' ')))
            new_state_dict[item] = pretrained_dict[prev_key]

    model.load_state_dict(new_state_dict,strict=False)

    model = nn.DataParallel(model)
    model.to(device)
    
    optimizer = torch.optim.SGD(
        # 
        params=[
            {
                "params": get_params(model, key="1x"),
                "lr": config.train.opt.learning_rate,
                "weight_decay": config.train.opt.weight_decay,
            },
            {
                "params": get_params(model, key="10x"),
                "lr": 10 * config.train.opt.learning_rate,
                "weight_decay": config.train.opt.weight_decay,
            },
            {
                "params": get_params(model, key="20x"),
                "lr": 20 * config.train.opt.learning_rate,
                "weight_decay": 0.0,
            },
        ],
        momentum=config.train.opt.momentum,
    )

    # criterion
    criterion = nn.CrossEntropyLoss(ignore_index=config.dataset.ignore_label)
    criterion = criterion.to(device)

    for group in optimizer.param_groups:
        group.setdefault('initial_lr', group['lr'])
    
    makedirs(os.path.join(config.exp.path, config.exp.checkpoint_dir))
    makedirs(os.path.join(config.exp.path, config.exp.tensorboard_dir))
    
    iteration = 0
    max_epoch = config.train.max_iters // len(train_loader) + 1

    model.train()
    model.module.base.freeze_bn()

    train_loader_iter = iter(train_loader)

    for epoch in tqdm(range(config.train.max_iters), total=config.train.max_iters, ascii=' 123456789>', dynamic_ncols=True):
        #for _, data in tqdm(enumerate(train_loader), total=len(train_loader), ascii=' 123456789>', dynamic_ncols=True):
        running_loss = 0.0
        #print('Training epoch %d / %d ...'%(epoch, max_epoch))

        #for _, data in tqdm(enumerate(train_loader), total=len(train_loader), ascii=' 123456789>', dynamic_ncols=True):
        # zero the parameter gradients
        optimizer.zero_grad()
        #_, inputs, labels = data
        try:
            _, inputs, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(train_loader)
            _, inputs, labels = next(train_loader_iter)
        inputs =  inputs.to(device)
        outputs = model(inputs)
            
        loss = 0.0
        for out in outputs:
            # resize labels
            resized_labels = imresize(labels, size=out.shape[2:])
            resized_labels = resized_labels.to(device)
            loss += criterion(out, resized_labels) / len(outputs)
            #resized_labels = F.interpolate(input=labels.unsqueeze(1), size=[41, 41], mode='nearest')
            
        loss.backward()
        
        optimizer.step()
    
        #running_loss += loss.item()
        
        # scheduler step
        iteration += 1
        ## poly scheduler
            
        #if iteration % config.train.update_iters == 0:
        for group in optimizer.param_groups:
            #g.setdefault('initial_lr', g['lr'])
            group['lr'] = group['initial_lr']*(1 - float(iteration) / config.train.max_iters) ** config.train.opt.power
            
        if iteration % config.train.save_iters == 0:
                    
            # save to tensorboard
            temp_k = 4
            inputs_part = inputs[0:temp_k,:]
            resized_outputs = F.interpolate(outputs[-1], size=inputs.shape[2:], mode='bilinear', align_corners=True)
            outputs_part = resized_outputs[0:temp_k,:]
            labels_part = labels[0:temp_k,:]

            grid_inputs, grid_outputs, grid_labels = imutils.tensorboard_image(inputs=inputs_part, outputs=outputs_part, labels=labels_part, bgr=config.dataset.mean_bgr)

            writer.add_image("train/images", grid_inputs, global_step=iteration)
            writer.add_image("train/preds", grid_outputs, global_step=iteration)
            writer.add_image("train/labels", grid_labels, global_step=iteration)
            writer.add_scalars("loss", {'train':loss}, global_step=iteration)

            #train_loss = running_loss / len(train_loader)
            #val_loss, score = validate(model=model, criterion=criterion, data_loader=val_loader, writer=None)
            #print('train loss: %f, val loss: %f, val pixel accuracy: %f, val mIoU: %f\n'%(train_loss, val_loss, score['Pixel Accuracy'], score['Mean IoU']))

            #writer.add_scalars("loss", {'train':train_loss}, global_step=epoch)
            #writer.add_scalar("val/acc", scalar_value=score['Pixel Accuracy'], global_step=epoch)
            #writer.add_scalar("val/miou", scalar_value=score['Mean IoU'], global_step=epoch)

    val_loss, score = validate(model=model, criterion=criterion, data_loader=val_loader, writer=None)
    print('val loss: %f, val pixel accuracy: %f, val mIoU: %f\n'%(val_loss, score['Pixel Accuracy'], score['Mean IoU']))

    dst_path = os.path.join(config.exp.path, config.exp.checkpoint_dir, config.exp.final_weights)
    #val_loss, score = validate(model=model, criterion=criterion, data_loader=val_loader)
    #print('val loss: %f, val pixel accuracy: %f, val mIoU: %f\n'%(val_loss, score['Pixel Accuracy'], score['Mean IoU']))
    torch.save(model.state_dict(), dst_path)
    torch.cuda.empty_cache()

    return True

if __name__=="__main__":

    config = OmegaConf.load(args.config)
    train(config)
