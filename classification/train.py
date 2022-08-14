# -*- coding: utf-8 -*-
import os
import sys
import math
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from argparse import RawTextHelpFormatter

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as torchdata
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from multiprocessing.reduction import ForkingPickler
from torch.utils.tensorboard import SummaryWriter

from utils.scheduler import WarmupMultiStepLR, GradualWarmupScheduler
from utils.rela import calc_rela
from utils.dataset import collate_fn_train, collate_fn_valid, collate_fn_pretext, dataset
from utils.transform import GaussianBlur, Identity
from utils.nce_loss import Info_NCE_Loss
from utils.LogWriter import LogWriter 
from models.resnet import resnet_simclr_lio as resnet

# +
# ========================= cuda imformation =========================
# -

# Releases all unoccupied cached memory
torch.cuda.empty_cache()
# prevent showing warnings
warnings.filterwarnings("ignore")

# print torch and cuda information
print('=========== torch & cuda infos ================')
print('torch version : ' + torch.__version__)
print('available: ' + str(torch.cuda.is_available()))
NUM_GPU = torch.cuda.device_count()
print('count: ' + str(NUM_GPU))
torch.backends.cudnn.benchmark = True # causes cuDNN to benchmark multiple convolution algorithms and select the fastest.

# 关闭pytorch的shared memory功能 (Bus Error)
# Ref: https://github.com/huaweicloud/dls-example/issues/26
for t in torch._storage_classes:
    if sys.version_info[0] == 2:
        if t in ForkingPickler.dispatch:
            del ForkingPickler.dispatch[t]
    else:
        if t in ForkingPickler._extra_reducers:
            del ForkingPickler._extra_reducers[t]


# +
# ========================= Arguments & Parameters =========================

# +
#Arguments
class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
    pass
parser = argparse.ArgumentParser(formatter_class=CustomFormatter)

# GPU
parser.add_argument('--num_worker', default=4, type=int, help="the number of workers. Suggest setting the number of your CPU core")
parser.add_argument('--cuda', default=0, type=int, 
                    help="-1: using all GPU for training.\notherwise: using selected GPU\n ")

# dataset
parser.add_argument('--valid_dir', default='./valid', type=str, help="folder path of your testing dataset")
parser.add_argument('--train_dir', default='./train', type=str, help="folder path of your training dataset")
parser.add_argument('--train_csv', default='training.csv', type=str, help="training csv for dataset labeling.\ncsv must include \"ImagePath\" column and its corresponding \"index\"")
parser.add_argument('--valid_csv', default='validation.csv', type=str, help="testing csv for dataset labeling.\ncsv must include \"ImagePath\" column and its corresponding \"index\"")
parser.add_argument('--num_classes', default=200, type=int, help="numbers of classes in your dataset")

# record
parser.add_argument('--save_dir', default='', type=str, help="where the model's weights save.\nIf you don't set this argment, model's weights will be saved to ./record/weight/{current time}/\n")
parser.add_argument('--log', default='', type=str, help="where you save the log file.\nIf you don't set this argment, log file will be written into ./record/log/{current time}.txt\n")
parser.add_argument('--log_explain', default='', type=str, help="additional explaination is written in log file.")
parser.add_argument('--mask_dir', default="", help="where correlation masks save when LIO model is open.\nIf you don't set this argment, masks will save to ./record/mask/{current time}/\n")
parser.add_argument('--epochs_per_mask', default=50, type=int, help="how many epochs to save correlation masks.")

# model information
parser.add_argument('--model_name', default='resnet', type=str, help="model's name (No matter what the name is, ResNet50 is used)")
parser.add_argument('--optimizer', default='sgd', type=str, help="optermizer's name (No matter what the name is, SGD is used)")

# hyperparameter for pretext task
parser.add_argument('--is_pretext', default='True', choices=('True','False'), help="whether do you want train pretext task")
parser.add_argument('--pretext_epochs', default=512, type=int, help="epochs for training pretext task")
parser.add_argument('--pretext_batch_size', default=32, type=int, help="batch size in pretext task")
parser.add_argument('--num_positive', default=2, type=int, help="number of positive images, suggest 2")
parser.add_argument('--temperature', default=1.0, type=float, help="temperature in NCE loss")
parser.add_argument('--with_LIO', default='True', choices=('True','False'), help="whether do you want to train pretext task with LIO")
parser.add_argument('--M', default='True', choices=('True','False'), help="whether do you want to train pretext task with LIO using M, suggest True")
parser.add_argument('--mask_weight', default=0.1, type=float, help="oel loss parameter")
parser.add_argument('--coor_weight', default=0.1, type=float, help="scl loss parameter")
parser.add_argument('--is_lio_loss_warmup', default='True', choices=('True','False'), help="whether do you want to use LIO loss parameter loss warm-up scheme")
parser.add_argument('--lio_loss_warmup', default=100, type=int, help="how many epochs do you want to warm-up the lio loss parameter")
parser.add_argument('--pretext_lr', default=0.01, type=float, help="learning rate in pretext task")
parser.add_argument('--pretext_scheduler', default='cos', choices=('cos','step'), help="learning rate scheduler in pretext task\ncos: cosine decay scheduler\nstep: step decay scheduler\n")
parser.add_argument('--pretext_resolution', default=224, type=int, help="input image resolution in pretext task")
parser.add_argument('--simclr_out_dim', default=128, type=int, help="output dimention of SimCLR projection head")
parser.add_argument('--pretext_pretrain_imagenet', default='False', choices=('True','False'), help="whether are the model's weights in pretext task pretrained by ImageNet classification task\n ")
            
# hyperparameter for downstream task
parser.add_argument('--downstream_epochs', default=128, type=int, help="epochs for training downstream task")
parser.add_argument('--downstream_batch_size', default=32, type=int, help="batch size in downstream task")
parser.add_argument('--downstream_lr', default=0.01, type=float, help="learning rate in downstream task")    
parser.add_argument('--downstream_scheduler', default='cos', choices=('cos','step'), help="learning rate scheduler in downstream task\ncos: cosine decay scheduler\nstep: step decay scheduler\n")
parser.add_argument('--downstream_resolution', default=224, type=int, help="input image resolution in downstream task")
parser.add_argument('--downstream_pretrain_imagenet', default='False', choices=('True','False'), help="whether are the model's weights in pretext task pretrained by ImageNet classification task")
parser.add_argument('--load_pretext', default='', help="the file path of the pretext model you want to load before training downstream task. \nIf you want to load pretext task, is_pretext must be False")

args = parser.parse_args()

# +
# Parameters
STAGE = 3

# GPU
NUM_WORKER = args.num_worker
CUDA = args.cuda

# dataset
TRAIN_DIR = args.train_dir
VALID_DIR = args.valid_dir
TRAIN_CSV = args.train_csv
VALID_CSV = args.valid_csv
NUM_CLASSES = args.num_classes

# record
SAVE_DIR = args.save_dir
LOG = args.log
LOG_EXPLAIN = args.log_explain
MASK_DIR = args.mask_dir
EPOCHS_PER_MASK = args.epochs_per_mask

# model information
MODEL_NAME = args.model_name
OPTIMIZER = args.optimizer

# hyperparameter for pretext task
IS_PRETEXT = args.is_pretext == 'True'
PRETEXT_EPOCHS = args.pretext_epochs
PRETEXT_BATCH_SIZE = args.pretext_batch_size
NUM_POSITIVE = args.num_positive
TEMPERATURE = args.temperature
WITH_LIO = args.with_LIO == 'True'
M = args.M == 'True'
MASK_WEIGHT = args.mask_weight
COOR_WEIGHT = args.coor_weight
IS_LIO_LOSS_WARMUP = args.is_lio_loss_warmup == 'True'
LIO_LOSS_WARMUP = args.lio_loss_warmup
PRETEXT_LR = args.pretext_lr
PRETEXT_SCHEDULER = args.pretext_scheduler
PRETEXT_RESOLUTION = args.pretext_resolution
MASK_SIZE = math.ceil(PRETEXT_RESOLUTION/32)
SIMCLR_OUT_DIM = args.simclr_out_dim
PRETEXT_PRETRAIN_IMAGENET = args.pretext_pretrain_imagenet == 'True'

# hyperparameter for downstream task
DOWNSTREAM_EPOCHS = args.downstream_epochs
DOWNSTREAM_BATCH_SIZE = args.downstream_batch_size
DOWNSTREAM_LR = args.downstream_lr
DOWNSTREAM_SCHEDULER = args.downstream_scheduler
DOWNSTREAM_RESOLUTION = args.downstream_resolution
DOWNSTREAM_PRETRAIN_IMAGENET = args.downstream_pretrain_imagenet == 'True'
LOAD_PRETEXT = args.load_pretext

assert not (IS_PRETEXT and LOAD_PRETEXT) # you can not load pretext and train pretext simultaneously

# +
# generate parameter of recording-use folder
now = datetime.datetime.now()
time = '{:02d}{:02d}{:02d}{:02d}'.format(now.month,now.day,now.hour,now.minute)
if not SAVE_DIR:
    SAVE_DIR = './record/weight/{}/'.format(time)
if WITH_LIO and not MASK_DIR:
    MASK_DIR = './record/mask/{}/'.format(time)
if not LOG:
    LOG = './record/log/{}.txt'.format(time)
    
PRETEXT_MODEL_DIR = '{}{}_pretext.pth'.format(SAVE_DIR,MODEL_NAME)
DOWNSTREAM_MODEL_DIR = '{}{}_downstream.pth'.format(SAVE_DIR,MODEL_NAME)
# -

# generate non-exist folder
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)
if IS_PRETEXT and WITH_LIO and not os.path.exists(MASK_DIR):
    os.makedirs(MASK_DIR)
if not os.path.exists(os.path.dirname(LOG)):
    os.makedirs(os.path.dirname(LOG))

# log writer
lprint = LogWriter(LOG)
if LOG_EXPLAIN: lprint("Note:", LOG_EXPLAIN)

# +
# ========================= data =========================
# -

# Read CSV
df_train = pd.read_csv(TRAIN_CSV, dtype={'ImagePath': str, 'index': int})
df_valid = pd.read_csv(VALID_CSV, dtype={'ImagePath': str, 'index': int})

lprint('==================== Dataset Info ====================')
lprint('train images:', df_train.shape)
lprint('valid images:', df_valid.shape)
lprint('num classes:', NUM_CLASSES)

# Data transforms (Preprocessing)
data_transforms = {
    'preprocess_pretext': transforms.Compose([
        transforms.Resize((int(PRETEXT_RESOLUTION/7*8),int(PRETEXT_RESOLUTION/7*8))),
    ]),
    'preprocess_train': transforms.Compose([
        transforms.Resize((int(DOWNSTREAM_RESOLUTION/7*8),int(DOWNSTREAM_RESOLUTION/7*8))),
    ]),
    'augment_pretext': transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop((PRETEXT_RESOLUTION,PRETEXT_RESOLUTION)),
        transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        GaussianBlur(kernel_size=int(0.1 * PRETEXT_RESOLUTION)),
        transforms.RandomHorizontalFlip(),
    ]),
    'augment_train': transforms.Compose([
        transforms.RandomRotation(degrees=15),
        transforms.RandomCrop((DOWNSTREAM_RESOLUTION,DOWNSTREAM_RESOLUTION)),
        transforms.RandomHorizontalFlip(),
    ]),
    'augment_valid': transforms.Compose([
        transforms.CenterCrop((DOWNSTREAM_RESOLUTION,DOWNSTREAM_RESOLUTION)),
    ]),
    'totensor_train': transforms.Compose([
        transforms.RandomCrop((DOWNSTREAM_RESOLUTION,DOWNSTREAM_RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'totensor_pretext': transforms.Compose([
        transforms.RandomCrop((PRETEXT_RESOLUTION,PRETEXT_RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'denormalize':transforms.Compose([ 
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
    ])
}

# Dataset
data_set = {
    'pretext': dataset(img_dir=TRAIN_DIR, anno_pd=df_train, preprocess=data_transforms["preprocess_pretext"],
                      augment=data_transforms["augment_pretext"], totensor=data_transforms["totensor_pretext"],
                      pretext=True, valid=False, n_view=NUM_POSITIVE),
    'train': dataset(img_dir=TRAIN_DIR, anno_pd=df_train, preprocess=data_transforms["preprocess_train"],
                      augment=data_transforms["augment_train"], totensor=data_transforms["totensor_train"],
                      pretext=False, valid=False, n_view=NUM_POSITIVE),
    'valid': dataset(img_dir=VALID_DIR, anno_pd=df_valid, preprocess=data_transforms["preprocess_train"],
                      augment=data_transforms["augment_valid"], totensor=data_transforms["totensor_train"],
                      pretext=False, valid=True, n_view=NUM_POSITIVE)  
}

# Dataloader
dataloader = {
    'pretext': torch.utils.data.DataLoader(data_set['pretext'], batch_size=PRETEXT_BATCH_SIZE, shuffle=True, 
                                           num_workers=NUM_WORKER, collate_fn=collate_fn_pretext),
    'train': torch.utils.data.DataLoader(data_set['train'], batch_size=DOWNSTREAM_BATCH_SIZE, shuffle=True, 
                                         num_workers=NUM_WORKER, collate_fn=collate_fn_train),
    'valid': torch.utils.data.DataLoader(data_set['valid'], batch_size=DOWNSTREAM_BATCH_SIZE, shuffle=False, 
                                         num_workers=NUM_WORKER, collate_fn=collate_fn_valid)
}


# +
# ========================= some function =========================

# +
def mask_to_binary_with_curr(x, coef):
    """change input masks to binary masks
    
    Args:
        x->tensor[c,h,w]: input masks
        coef->float: the coefficient of the thresholds. The higher coef is, the higher threshold is.
    
    Returns:
        tensor[c,h,w]: binary masks.
    """
    N, H, W = x.shape
    x = x.view(N, H*W)
    thresholds = coef * torch.mean(x, dim=1, keepdim=True)
    binary_x = (x > thresholds).float()
    return binary_x.view(N, H, W)

def visualize_images(images, rows = 1, titles = [], save_dir = ""):
    """visualize the input images, save images to save_dir
    
    cols = len(images)/rows
    images will be arrange to rows*cols and put into a picture in save_dir
    
    Args:
        images->tensor[] or np.array[] of arr[]: array of input images
        rowss->int: how many rows in a picture
        titless-> str[]: array of titles for each images. len(titles) must equal len(images)
        save_dirs-> str: picture path where is picture saved.
    """
    n = len(images)
    for i in range(n):
        plt.subplot(rows, math.floor(n/rows), i+1)
        plt.title(titles[i])
        plt.imshow(torch.clip(images[i], min=0, max=1))
    plt.axis('off')
    if save_dir != "":
        plt.savefig(save_dir, dpi=300)
    plt.show()


# +
# ========================= training =========================
# -

# record training information
lprint('==================== info ====================')
lprint("model:",MODEL_NAME)
lprint()
if IS_PRETEXT:
    lprint("*** Pretext imformation ***")
    lprint("batch size:", PRETEXT_BATCH_SIZE)
    lprint("epoch:", PRETEXT_EPOCHS)
    lprint("temperature:", TEMPERATURE)
    lprint("with LIO:", WITH_LIO)
    lprint("M:", M)
    if WITH_LIO:
        lprint("mask/coor loss weight:", MASK_WEIGHT, COOR_WEIGHT)
        if IS_LIO_LOSS_WARMUP:
            lprint("loss warmup epochs:", LIO_LOSS_WARMUP)
        lprint("LIO mask size:", MASK_SIZE)
    lprint("learning rate:", PRETEXT_LR)
    lprint("scheduler:", PRETEXT_SCHEDULER)
    lprint("input resolution:", PRETEXT_RESOLUTION)
    lprint("simclr out dim:", SIMCLR_OUT_DIM)
    lprint("pretext pretrain imagenet:", PRETEXT_PRETRAIN_IMAGENET)
    lprint()
lprint("*** downstream imformation ***")
lprint("batch size:", DOWNSTREAM_BATCH_SIZE)
lprint("epoch:", DOWNSTREAM_EPOCHS)
lprint("learning rate:", DOWNSTREAM_LR)
lprint("scheduler:", DOWNSTREAM_SCHEDULER)
lprint("input resolution:", DOWNSTREAM_RESOLUTION)
lprint("pretrain with imagenet:", DOWNSTREAM_PRETRAIN_IMAGENET)
if not IS_PRETEXT:
    lprint("Load pretext model from", LOAD_PRETEXT)
lprint()
lprint("*** record ***")
if IS_PRETEXT: lprint("pretext model save at ", PRETEXT_MODEL_DIR)
lprint("downstream model save at ", DOWNSTREAM_MODEL_DIR)
lprint("log write at ", LOG)
if IS_PRETEXT and WITH_LIO: lprint("mask save at ", MASK_DIR)

# call out gpu is possible
dev_str = "cuda" if CUDA < 0 else "cuda:{}".format(CUDA)
if CUDA >= 0: torch.cuda.set_device(CUDA)
device = torch.device(dev_str if torch.cuda.is_available() else "cpu")
lprint('{} will be used in the training process !!!'.format(device))

# +
# ------------------------- Pretext state -------------------------
# -

# model
model = resnet(stage=STAGE, in_dim=2048, out_dim=SIMCLR_OUT_DIM, num_classes=NUM_CLASSES, size=MASK_SIZE, 
               with_LIO=WITH_LIO, num_positive = NUM_POSITIVE, M = M, 
               pretext = True, pretrain = PRETEXT_PRETRAIN_IMAGENET)
model.to(device) # model.cuda()
if str(device) == "cuda":
    model = torch.nn.DataParallel(model)

# optimizer, loss function, & learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=PRETEXT_LR, momentum=0.9, weight_decay=5e-4)
criterion = CrossEntropyLoss().to(device)
if PRETEXT_SCHEDULER == 'cos':
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PRETEXT_EPOCHS, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, warmup_epoch=10, after_scheduler=cosine_scheduler)
elif PRETEXT_SCHEDULER == 'step':
    scheduler = WarmupMultiStepLR(optimizer, warmup_epoch = 2, milestones = [30,60,90,120,150])
info_nce_loss = Info_NCE_Loss(NUM_POSITIVE, TEMPERATURE, device)


def train_simclr(epoch):
    """train SimCLR pretext task
    
    Args:
        epoch->int: current epoch.
    """
    global loss_weight, best_pretext_loss
    model.train(True)
    bar_len_total, batch_total = 10, len(dataloader['pretext'])
    train_loss = 0
    lprint('current lr: %f' % optimizer.param_groups[0]['lr'])
    print('Epoch {}/{}'.format(epoch + 1, PRETEXT_EPOCHS))
    for batch_idx, (imgs, aug_imgs) in enumerate(dataloader['pretext']):
        aug_imgs = aug_imgs.to(device)
        optimizer.zero_grad()
        
        # ------------- data arrangement -------------
        N, C, H, W = imgs.shape
        aug_imgs = aug_imgs.view(N, NUM_POSITIVE, C, H, W).transpose(0, 1).contiguous().view(NUM_POSITIVE*N, C, H, W) # (NUM_POSITIVE*BATCH_SIZE, 3, H, W)
        
        # ------------- imput augmented images -------------
        aug_features = model(aug_imgs)      # (NUM_POSITIVE*BATCH_SIZE, 2048)
        
        # ------------- simclr --------------
        loss = info_nce_loss(aug_features)
        train_loss += loss.item()
        
        # ------------- backpropagation --------------
        loss.backward()
        optimizer.step()
        
        # showing current progress
        bar_len = math.floor(bar_len_total * (batch_idx + 1) / batch_total)  - 1
        print('{}/{} '.format(batch_idx + 1, batch_total) + 
              '[' + '=' * bar_len + '>' + '.' * (bar_len_total - (bar_len + 1))  + '] ' + 
              '- nce loss : {:.2f} '.format(loss.data.item()) , end='\r')  
    
    # record the final result in this epoch
    current_pretext_loss = train_loss / (batch_idx + 1)    
    lprint(str(datetime.datetime.now().replace(microsecond = 0)) + 
          '- epoch: {}/{} '.format(epoch+1, PRETEXT_EPOCHS) + 
          '- train loss : {:.3f} '.format(current_pretext_loss))
    
    # save the best model's weight
    if current_pretext_loss < best_pretext_loss:
        lprint('Best pretext loss achieved, saving model to {}'.format(PRETEXT_MODEL_DIR))
        state = {
            'model': model.state_dict(),  # save model by the data sturcture of dictionary
            'epoch': epoch+1,
            'pretext_loss': current_pretext_loss,
        }
        torch.save(state, PRETEXT_MODEL_DIR)
        best_pretext_loss = current_pretext_loss
    model.train(False)


def train_simclr_LIO(epoch):
    """train SimCLR pretext task with LIO
    
    Args:
        epoch->int: current epoch.
    """
    global loss_weight, best_pretext_loss
    model.train(True)
    bar_len_total, batch_total = 10, len(dataloader['pretext'])
    batchs_per_mask = int(batch_total/2)
    train_loss, nce_loss, coor_loss, mask_loss = 0, 0, 0, 0
    lprint('current lr: %f' % optimizer.param_groups[0]['lr'])
    print('Epoch {}/{}'.format(epoch + 1, PRETEXT_EPOCHS))
    
    for batch_idx, (imgs, aug_imgs) in enumerate(dataloader['pretext']):
        imgs, aug_imgs = imgs.to(device), aug_imgs.to(device)
        optimizer.zero_grad()
        
        # ------------- data arrangement -------------
        N, C, H, W = imgs.shape
        aug_imgs = aug_imgs.view(N, NUM_POSITIVE, C, H, W).transpose(0, 1).contiguous().view(NUM_POSITIVE*N, C, H, W) # (NUM_POSITIVE*BATCH_SIZE, 3, H, W)
        all_imgs = torch.cat((imgs, aug_imgs), dim=0) # ((1+NUM_POSITIVE)*BATCH_SIZE, 3, H, W)
        
        # ------------- input augmented images to model with LIO -------------
        all_output = model(all_imgs)
        all_mask = all_output[0]          # ((1+NUM_POSITIVE)*BATCH_SIZE , 8, 8)  # predicted masks
        all_object_extent = all_output[1] # ((1+NUM_POSITIVE)*BATCH_SIZE , 8, 8)  # correlation masks
        coord_loss = all_output[2]        # ((1+NUM_POSITIVE)*BATCH_SIZE , 8, 8)  # scl loss
        mask_reg_loss = all_output[3]     # ((1+NUM_POSITIVE)*BATCH_SIZE , 8, 8)  # oel loss
        all_features = all_output[4]      # ((1+NUM_POSITIVE)*BATCH_SIZE, 2048)   # vector output from projection head
        
        # ------------- simclr --------------
        aug_features = all_features[N:]   # features of augmented images
        sim_nce_loss = info_nce_loss(aug_features)
        
        # ------------- corr loss -------------
        curr_coef = 1.0
        bin_mask = mask_to_binary_with_curr(all_object_extent if M else all_mask, curr_coef)
        coord_loss = torch.mean(coord_loss * bin_mask)
        
        # ------------- rela calculation -------------
        mask_reg_loss = mask_reg_loss.mean()
        
        # *********** total loss ****************
        loss = sim_nce_loss + coord_loss * loss_weight['coord'] + mask_reg_loss * loss_weight['mask']
        train_loss += loss.item()
        nce_loss += sim_nce_loss.item()
        coor_loss += coord_loss
        mask_loss += mask_reg_loss
        
        # ------------- backpropagation --------------
        loss.backward()
        optimizer.step()
        
        # ------------- save predictied mask and correlation mask -------------
        if batch_idx % batchs_per_mask == 0 and epoch % EPOCHS_PER_MASK == 0:
            bin_all_mask = mask_to_binary_with_curr(all_mask, 1.0).cpu()
            bin_all_object_extent = mask_to_binary_with_curr(all_object_extent, 1.0).cpu()
            toshow = []
            for i in range(3):
                showimg1, showimg2 = data_transforms["denormalize"](all_imgs[i]), data_transforms["denormalize"](all_imgs[i+N])
                toshow.extend([showimg1.permute(1, 2, 0).cpu(), bin_all_mask[i], bin_all_object_extent[i]])
                toshow.extend([showimg2.permute(1, 2, 0).cpu(), bin_all_mask[i+N], bin_all_object_extent[i+N]])
            toshowtitles =  ["image", "mask", "OEL"] * 6
            visualize_images(toshow, titles =toshowtitles,rows = 3,
                            save_dir = MASK_DIR+"epoch_"+str(epoch)+"_batch_"+str(batch_idx)+'.png')
        
        # showing current progress
        bar_len = math.floor(bar_len_total * (batch_idx + 1) / batch_total)  - 1
        print('{}/{} '.format(batch_idx + 1, batch_total) + 
              '[' + '=' * bar_len + '>' + '.' * (bar_len_total - (bar_len + 1))  + '] ' + 
              '- train loss : {:.2f} '.format(loss.data.item()) +
              '- nce : {:.2f} '.format(sim_nce_loss.data.item()) + 
              '- coor(l/w) : {:.2f}/{:.2f} '.format(coord_loss,loss_weight['coord']) + 
              '- mask(l/w) : {:.2f}/{:.2f} '.format(mask_reg_loss,loss_weight['mask']), end='\r')  
    
    # record the final result in this epoch
    current_pretext_loss = train_loss / (batch_idx + 1)
    lprint(str(datetime.datetime.now().replace(microsecond = 0)) + 
          '- epoch: {}/{} '.format(epoch+1, PRETEXT_EPOCHS) + 
          '- train loss : {:.3f} '.format(current_pretext_loss) + 
          '- nce : {:.2f} '.format(nce_loss / (batch_idx + 1)) + 
          '- coor(l/w) : {:.2f}/{:.2f} '.format(coor_loss / (batch_idx + 1),loss_weight['coord']) + 
          '- mask(l/w) : {:.2f}/{:.2f} '.format(mask_loss / (batch_idx + 1),loss_weight['mask']))
    
    # save the best model's weight
    if current_pretext_loss < best_pretext_loss:
        lprint('Best pretext loss achieved, saving model to {}'.format(PRETEXT_MODEL_DIR))
        state = {
            'model': model.state_dict(),    # save model by the data sturcture of dictionary
            'epoch': epoch+1,
            'pretext_loss': current_pretext_loss,
        }
        torch.save(state, PRETEXT_MODEL_DIR)
        best_pretext_loss = current_pretext_loss
    model.train(False)


class Loss_Weight_Warmup:
    """LIO loss parameter warmup function

    Attributes:
        weight->float[2]: original lio loss parameter. include oel loss parameter and scl loss parameter
        is_warmup->bool: whether use lio loss parameter warmup scheme. If False, this function always returns the original lio loss parameter.
        warmup->int: how many epoch you want to warmup
        epochs: total epochs you train the model.
    """
    def __init__(self, weight, is_warmup, warmup, epochs):
        self.weight = weight
        self.is_warmup = is_warmup
        self.warmup = warmup
        self.epochs = epochs
        
    def __call__(self, epoch):
        """calculate current lio loss parameter
    
        Args:
            epoch->int: current epoch.
            
        Returns:
            float[2]: current lio loss parameter.
                      if not is_warmup, this function always returns the original lio loss parameter.
        """
        if not self.is_warmup:
            return self.weight
        output_dict = {}
        if epoch < self.warmup:  # lio loss parameter set 0 before warmup epochs 
            for key in self.weight:
                output_dict[key] = 0.
            return output_dict
        else: # lio loss parameter linearly raise to the original lio loss parameter after warmup epochs 
            factor = (epoch-self.warmup)/(self.epochs-self.warmup)
            for key in self.weight:
                output_dict[key] = self.weight[key] * factor # learning raising
            return output_dict


# loss weight for LIO
loss_weight = { 'mask': MASK_WEIGHT, 'coord': COOR_WEIGHT}

# train pretext stage
lprint("==================== pretext ====================")
best_pretext_loss, start_epoch =1000000, 0
if IS_PRETEXT:
    # load pretext weight if checkpoint exist
    if os.path.exists(PRETEXT_MODEL_DIR): # find checkpoint directory
        checkpoint = torch.load(PRETEXT_MODEL_DIR)
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch']
        best_pretext_loss = checkpoint['pretext_loss']
        lprint('weights loaded! epoch:{}, pretext loss:{:.3f}'.format(start_epoch,best_pretext_loss))
    else:
        lprint("no pretext model in {}".format(PRETEXT_MODEL_DIR))
        
    #start training pretext stage
    loss_weight_warmup = Loss_Weight_Warmup(loss_weight, IS_LIO_LOSS_WARMUP, LIO_LOSS_WARMUP, PRETEXT_EPOCHS)
    for epoch in range(start_epoch, PRETEXT_EPOCHS):
        scheduler.step(epoch)
        if WITH_LIO:
            loss_weight = loss_weight_warmup(epoch) # set lio loss pararmter
            train_simclr_LIO(epoch)
        else:
            train_simclr(epoch)

# +
# ------------------------- Downstream state -------------------------

# +
# model
model = resnet(stage=STAGE, in_dim=2048, out_dim=SIMCLR_OUT_DIM, num_classes=NUM_CLASSES, size=MASK_SIZE, 
               with_LIO=False, num_positive = NUM_POSITIVE, M = M, 
               pretext=False, pretrain=DOWNSTREAM_PRETRAIN_IMAGENET) 

# load backbone_model from pretext weight
if IS_PRETEXT:
    lprint("Load best pretext model.")
    msg = model.load_backbone_model(PRETEXT_MODEL_DIR) # load the model you just trained
    lprint(msg)
elif LOAD_PRETEXT:
    lprint("Load pretext model.")
    msg = model.load_backbone_model(LOAD_PRETEXT)      # load the model from LOAD_PRETEXT
    lprint(msg)
    
model.to(device) # model.cuda()
if str(device) == "cuda":
    model = torch.nn.DataParallel(model)
# -

# =============================================================================
# optimizer, loss function, & learning rate scheduler
optimizer = optim.SGD(model.parameters(), lr=DOWNSTREAM_LR, momentum=0.9, weight_decay=5*1e-4)
criterion = CrossEntropyLoss() 
if DOWNSTREAM_SCHEDULER == 'cos':
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=DOWNSTREAM_EPOCHS, eta_min=0, last_epoch=-1)
    scheduler = GradualWarmupScheduler(optimizer, warmup_epoch=10, after_scheduler=cosine_scheduler)
elif DOWNSTREAM_SCHEDULER == 'step':
    scheduler = WarmupMultiStepLR(optimizer, warmup_epoch = 2, milestones = [30,60,90,120,150])

train_losses, valid_losses = [], []
train_accs, valid_accs = [], []

def train(epoch):
    """train downstream task
    
    Args:
        epoch->int: current epoch.
    """
    global train_losses, train_accs
    model.train(True)        
    train_loss, correct, total = 0, 0, 0
    bar_len_total, batch_total = 10, len(dataloader['train'])
    lprint('current lr: %f' % optimizer.param_groups[0]['lr'])
    print('Epoch {}/{}'.format(epoch + 1, DOWNSTREAM_EPOCHS))

    for batch_idx, (org_imgs, imgs, labels) in enumerate(dataloader['train']):
        labels = torch.LongTensor(np.array(labels))
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()

        # *********** input main images ***************
        main_probs = model(imgs)
        loss = criterion(main_probs, labels)
        train_loss += loss.item()

        # *********** training acc **************
        _, predicted = main_probs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        train_acc = 1. * correct / total
        
        # *********** backpropagation ***************
        loss.backward()
        optimizer.step()

        # showing current progress
        bar_len = math.floor(bar_len_total * (batch_idx + 1) / batch_total)  - 1
        print('{}/{} '.format(batch_idx + 1, batch_total) + 
              '[' + '=' * bar_len + '>' + '.' * (bar_len_total - (bar_len + 1))  + '] ' + 
              '- train loss : {:.3f} '.format(loss.data.item()) +
              '- train acc : {:.3f} '.format(train_acc)
             , end='\r') 
    
    # record the result
    train_losses.append(train_loss / (batch_idx + 1))
    train_accs.append(train_acc)

    model.train(False)

def valid(epoch):
    """valid downstream task, no backpropagation
    
    Args:
        epoch->int: current epoch.
    """
    global best_acc, train_losses, train_accs, valid_losses, valid_accs
    model.eval()
    valid_loss, correct, total = 0, 0, 0
    bar_len_total, batch_total = 10, len(dataloader['valid'])

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloader['valid']):
            labels = torch.LongTensor(np.array(labels))
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            valid_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)

            correct += predicted.eq(labels).sum().item()
        
    # validation loss and accuracy
    valid_acc = 1. *correct / total

    # record result of this epoch
    train_loss, train_acc = train_losses[-1], train_accs[-1]
    lprint(str(datetime.datetime.now().replace(microsecond = 0))+
           "- epoch: {}/{} ".format(epoch+1, DOWNSTREAM_EPOCHS) + 
           '- train loss: {:.3f} '.format(train_loss) + 
           '- train acc: {:.3f} '.format(train_acc) + 
           '- valid loss: {:.3f} '.format(valid_loss / (batch_idx + 1)) + 
           '- valid acc: {:.4f} '.format(valid_acc))

    # record the final training loss and acc in this epoch
    valid_losses.append(valid_loss / (batch_idx + 1))
    valid_accs.append(valid_acc)

    # save checkpoint if the result achieved is the best
    if valid_acc > best_acc:
        lprint('Best accuracy achieved, saving model to ' + DOWNSTREAM_MODEL_DIR)
        state = {
            'model': model.state_dict(),    # save model by the data sturcture of dictionary
            'acc': valid_acc,
            'epoch': epoch+1,
        }
        torch.save(state, DOWNSTREAM_MODEL_DIR)
        best_acc = valid_acc

# +
lprint("==================== downstream ====================")
# load downstream weight
best_acc, start_epoch =0., 0
if os.path.exists(DOWNSTREAM_MODEL_DIR): # find checkpoint directory
    checkpoint = torch.load(DOWNSTREAM_MODEL_DIR)
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']
    best_acc = checkpoint['acc']
    lprint('weights loaded! epoch:{}, accuracy:{:.3f}'.format(start_epoch,best_acc))
    if IS_PRETEXT or LOAD_PRETEXT:
        lprint('pretext overwrite to downstream')
else:
    lprint("no downstream model in {}".format(DOWNSTREAM_MODEL_DIR))

# start train downstream stage
for epoch in range(start_epoch, DOWNSTREAM_EPOCHS):
    scheduler.step(epoch)           # learning rate scheduler
    train(epoch)                    # training state
    valid(epoch)                    # validation state
