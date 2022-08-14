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

import torch
from torchvision import transforms
from multiprocessing.reduction import ForkingPickler
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from utils.dataset import collate_fn_train, dataset
from utils.transform import GaussianBlur, ResizeToResolution, Identity
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

parser.add_argument('--num_worker', default=4, type=int, help="the number of workers. Suggest setting the number of your CPU core")
parser.add_argument('--cuda', default=0, type=int, 
                    help="-1: using all GPU for training.\notherwise: using selected GPU\n ")

parser.add_argument('--image_dir', default='./valid', type=str, help="the folder path of images")
parser.add_argument('--image_csv', default='validation.csv', type=str, help="the csv file path of images")
parser.add_argument('--log', default='', type=str, help="where you save the log file")
parser.add_argument('--log_explain', default='', type=str, help="additional explaination is written in log file.")

parser.add_argument('--model_dirs', default="", type=str, nargs='+', help="model weight pathes. you can input multiple pathes")
parser.add_argument('--titles', default="", type=str, nargs='+', help="titles of the images. len(titles) must equal to len(model_dirs)")

parser.add_argument('--resolution', default=224, type=int, help="input image resolution")
parser.add_argument('--save_dir', default='./record/gradcam/', type=str, help="where you want to save your results")
parser.add_argument('--num_cam', default=20, type=int, help="number of GradCAM")
args = parser.parse_args()

# +
# Params
STAGE = 3
NUM_WORKER = args.num_worker
CUDA = args.cuda

IMAGE_DIR = args.image_dir
IMAGE_CSV = args.image_csv

LOG = args.log
LOG_EXPLAIN = args.log_explain
MODEL_DIRS = args.model_dirs
TITLES = args.titles
assert len(TITLES) == len(MODEL_DIRS)
MODEL_NUM = len(MODEL_DIRS)

RESOLUTION = args.resolution
MASK_SIZE = math.ceil(RESOLUTION/32)
SAVE_DIR = args.save_dir
NUM_CAM = args.num_cam
# -

# call log writer
now = datetime.datetime.now()
time = '{:02d}{:02d}{:02d}{:02d}'.format(now.month,now.day,now.hour,now.minute)
if not LOG:
    LOG = './record/log/gradcam/log_gradcam.txt'.format(time)
if not os.path.exists(os.path.dirname(LOG)):
    os.makedirs(os.path.dirname(LOG))
lprint = LogWriter(LOG)
if LOG_EXPLAIN: lprint("Note:", LOG_EXPLAIN)  

if not os.path.exists(os.path.dirname(SAVE_DIR)):
    os.makedirs(os.path.dirname(SAVE_DIR))

# +
# ========================= data =========================
# -

# Read CSV
df_image = pd.read_csv(IMAGE_CSV, dtype={'ImagePath': str, 'index': int})

lprint('==================== Dataset Info ====================')
lprint('valid images:', df_image.shape)

# Data transforms (Preprocessing)
data_transforms = {
    'preprocess': transforms.Compose([
        transforms.Resize((int(RESOLUTION/4*5),int(RESOLUTION/4*5))),
    ]),
    'augment': transforms.Compose([
        transforms.CenterCrop((RESOLUTION,RESOLUTION)),
    ]),
    'totensor': transforms.Compose([
        transforms.CenterCrop((RESOLUTION,RESOLUTION)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'denormalize':transforms.Compose([ 
        transforms.Normalize(mean = [ 0., 0., 0. ], std = [ 1/0.229, 1/0.224, 1/0.225 ]),
        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ], std = [ 1., 1., 1. ]),
    ])
}

# Dataset
data_set = dataset(img_dir=IMAGE_DIR, anno_pd=df_image, preprocess=data_transforms["preprocess"],
                      augment=data_transforms["augment"], totensor=data_transforms["totensor"],
                      pretext=False, valid=False)

# Dataloader
dataloader = torch.utils.data.DataLoader(data_set, batch_size=1, shuffle=True, 
                                         num_workers=NUM_WORKER, collate_fn=collate_fn_train)

# +
# ========================= GradCAM =========================
# Ref: https://github.com/jacobgil/pytorch-grad-cam
# -

# record training information
lprint('==================== info ====================')
lprint("input resolution:", RESOLUTION)
lprint("model load at ", MODEL_DIRS)
lprint("titles: ", TITLES)
lprint("log write at ", LOG)
lprint("gradcam save at ", SAVE_DIR)

# call out gpu is possible
dev_str = "cuda" if CUDA < 0 else "cuda:{}".format(CUDA)
if CUDA >= 0: torch.cuda.set_device(CUDA)
device = torch.device(dev_str if torch.cuda.is_available() else "cpu")
lprint('{} will be used in the training process !!!'.format(device))

lprint('==================== load ====================')

models = []
target_layers = []
for i,model_dir in enumerate(MODEL_DIRS):
    # load checkpoint model
    checkpoint = torch.load(model_dir)
    checkpoint_state_dict_keys = checkpoint['model'].keys()
    lprint('load checkpoint from {}'.format(model_dir))
    
    model_dict_clear = {}
    # delete 'module'
    for key in checkpoint_state_dict_keys:
        if key.startswith('module'):
            model_dict_clear[key[len('module.'):]] = checkpoint['model'][key]
        else:
            model_dict_clear[key] = checkpoint['model'][key] 
            
    model_dict = {}
    # del lio module
    for key in model_dict_clear.keys():
        if not key.startswith('scl_lrx'):
            model_dict[key] = model_dict_clear[key]
        else:
            lprint('del lio module: {}'.format(key))
    del model_dict_clear
    
    # generate model
    lprint('generate model:', end = "")
    if 'prediction_head.weight' in model_dict.keys():      # is downstream model because it has prediction head
        dim = model_dict['prediction_head.weight'].shape[0]
        model = resnet(pretext = False, num_classes = dim, pretrain = False, with_LIO=False)
        lprint('downstream model')
    elif 'projection_head.0.weight' in model_dict.keys(): # is pretext model because it has projection head
        dim = model_dict['projection_head.2.weight'].shape[0]
        model = resnet(pretext = True,out_dim = dim, pretrain = False, with_LIO=False)
        lprint('pretext model')
    else:
        lprint("error module structute")
        exit()
    
    # load weights to the model
    try:
        model.load_state_dict(model_dict)
        lprint('Successfully load model from {}'.format(model_dir))
    except:
        lprint("Model load failed. Model and checkpoint not match!")
        lprint("model_dict not in model", list(set(model_dict.keys())-set(model.state_dict().keys())))
        lprint("model not in model_dict", list(set(model.state_dict().keys())-set(model_dict.keys())))
        exit()
    
    # get target layer
    target_layer = list(model.children())[3]
    target_layers.append(target_layer)
    lprint('get target layer')
    
    # model to gpu
    model.to(device) 
    if str(device) == "cuda": 
        model = torch.nn.DataParallel(model)
    model.train(True)
    models.append(model)

lprint('==================== CAM ====================')

# Construct the CAM object once, and then re-use it on many images:
cams = []
for model, target_layer in zip(models,target_layers):
    cams.append(GradCAM(model=model, target_layers=target_layer, use_cuda=True))
lprint('Constructed the CAM object')
# If target_category is None, the highest scoring category will be used for every image in the batch.
target_category = None

for batch_idx, (org_imgs, imgs, labels) in enumerate(dataloader):
    if batch_idx >= NUM_CAM: break
    print("process images {}/{}".format(batch_idx,NUM_CAM), end='\r')
    org_imgs = data_transforms["denormalize"](org_imgs.squeeze()) 
    org_imgs = org_imgs.permute(1,2,0)  # change org_imgs to [H,W,C]
    
    plt.subplot(1, MODEL_NUM+1, 1)
    plt.imshow(org_imgs)
    plt.title("image", fontsize=5)
    plt.axis('off')

    for cam_id, cam in enumerate(cams):
        plt.subplot(1, MODEL_NUM+1, cam_id+2)
        grayscale_cam = cam(input_tensor=imgs, targets=target_category)  # get GradCAM
        grayscale_cam = grayscale_cam[0, :]
        visualization = show_cam_on_image(org_imgs.numpy(), grayscale_cam,use_rgb=True) # draw heatmap on images
        plt.title(TITLES[cam_id], fontsize=5)
        plt.imshow(visualization)
        plt.axis('off')
    
    plt.savefig(SAVE_DIR+'gradcam'+str(batch_idx)+'.png',bbox_inches='tight',pad_inches = 0, dpi = 600)
lprint('Cam save at {}'.format(SAVE_DIR))
