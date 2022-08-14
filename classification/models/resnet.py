import numpy as np
import torch
from torch import nn
from torchvision import models

from .scl_module import SCLModule

import os
import sys
sys.path.append(os.path.abspath('../MYSEG_SIMCLR_LIO/mmsegmentation'))
from mmcv.utils import Config
from mmseg.models import build_segmentor
from mmseg.utils import (collect_env, get_device, get_root_logger,
                         setup_multi_processes)


class resnet_simclr_lio(nn.Module):
    def __init__(self, stage=3, in_dim=2048, out_dim=128, num_classes=200, size=7, with_LIO=True, M=True,
                 num_positive=2, pretext=True, pretrain=False):
        """ResNet50 with SimCLR projection head and LIO module

        Attributes:
            stage->int: always 3 in our experiment
            in_dim->int: input dimention of prediction head or projection head
            out_dim->int: output dimention of projection head
            num_classes->int: number of classes. output dimention of classifier
            size->int: the size of the correlation mask
            pretext->bool: whether train pretext task
            with_LIO->bool: whether train pretext task with LIO module
            M->bool: whether to train with pretext task LIO using M
            num_positive->int: number of positive images
            pretrain->bool: whether are the model's weights pretrained by ImageNet classification task
        """
        super(resnet_simclr_lio,self).__init__()
        self.num_positive = num_positive
        self.M = M
        self.stage = stage
        self.size = size
        self.with_LIO = with_LIO
        self.pretext = pretext
        
        resnet50 = models.resnet50(pretrained=pretrain)
        # backbone (4 stage module and 1 adaptive average pooling)
        self.stage1_img = nn.Sequential(*list(resnet50.children())[:5])  # return (batch_size, 256, /4, /4)
        self.stage2_img = nn.Sequential(*list(resnet50.children())[5:6]) # return (batch_size, 512, /2, /2)
        self.stage3_img = nn.Sequential(*list(resnet50.children())[6:7]) # return (batch_size, 1024, /2, /2)
        self.stage4_img = nn.Sequential(*list(resnet50.children())[7])   # return (batch_size, 2048, /2, /2)
        self.adaptive_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
        )
        
        # projection head
        if self.pretext:
            self.projection_head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim),
            )
            
        # prediction head
        else:
            self.prediction_head = nn.Linear(in_dim, num_classes)
            
        # LIO model
        if stage == 3:
            self.feature_dim = 2048
            self.structure_dim = 1024
        elif stage == 2:
            self.feature_dim = 1024
            self.structure_dim = 512
        elif stage == 1:
            self.feature_dim = 512
            self.structure_dim = 256
        else:
            raise NotImplementedError("No such stage")
        if self.with_LIO: # call LIO module
            self.scl_lrx = SCLModule(self.size, self.feature_dim, self.structure_dim, avg=False, 
                                     num_positive = self.num_positive, M = self.M)

    def forward(self, x):
        # process backbone model
        x2 = self.stage1_img(x)
        x3 = self.stage2_img(x2)
        x4 = self.stage3_img(x3)
        x5 = self.stage4_img(x4)  
        x6 = self.adaptive_avg_pool(x5)
        
        # process projection head or prediction head
        if self.training and self.pretext:
            if self.with_LIO:
                if self.stage == 3:
                    mask, object_extents, coord_loss, mask_loss = self.scl_lrx(x5)
                elif self.stage == 2:
                    mask, object_extents, coord_loss, mask_loss = self.scl_lrx(x4)
                elif self.stage == 1:
                    mask, object_extents, coord_loss, mask_loss = self.scl_lrx(x3)
                return mask, object_extents, coord_loss, mask_loss, self.projection_head(x6) # pretext training with LIO
            return self.projection_head(x6) # pretext training without LIO
        elif self.pretext: return self.projection_head(x6)
        return self.prediction_head(x6) # downstream training or validation
    
    def load_backbone_model(self, save_dir):
        """load backbone model weight from save_dir

        Args:
            save_dir->str: the path of backbone model

        Returns:
            str: the message of success loading
        """
        checkpoint = torch.load(save_dir) # load model
        # backbone (4 stage module) model state dictionary
        stage1_img_dict = {}
        stage2_img_dict = {}
        stage3_img_dict = {}
        stage4_img_dict = {}
            
        # fill these 4 model state dictionary
        for k in list(checkpoint['model'].keys()):
            if k.startswith('stage1_img'):
                stage1_img_dict[k[len('stage1_img.'):]] = checkpoint['model'][k]
            elif k.startswith('stage2_img'):
                stage2_img_dict[k[len('stage2_img.'):]] = checkpoint['model'][k]
            elif k.startswith('stage3_img'):
                stage3_img_dict[k[len('stage3_img.'):]] = checkpoint['model'][k]
            elif k.startswith('stage4_img'):
                stage4_img_dict[k[len('stage4_img.'):]] = checkpoint['model'][k]
            del checkpoint['model'][k]
            
        # load model from model state dictionary
        msg = self.stage1_img.load_state_dict(stage1_img_dict, strict=False)
        msg = self.stage2_img.load_state_dict(stage2_img_dict, strict=False)
        msg = self.stage3_img.load_state_dict(stage3_img_dict, strict=False)
        msg = self.stage4_img.load_state_dict(stage4_img_dict, strict=False)
        return 'Successfully load backbone model from {}'.format(save_dir)        


class deeplab_resnet(nn.Module):
    def __init__(self, stage=3, in_dim=2048, out_dim=128, num_classes=200, size=7, with_LIO=True, M=True,
                 num_positive=2, pretext=True, pretrain=False, config = None):
        """Deeplab V3 (encoder ResNet-50) with SimCLR projection head and LIO module

        Attributes:
            stage->int: always 3 in our experiment
            in_dim->int: input dimention of prediction head or projection head
            out_dim->int: output dimention of projection head
            num_classes->int: number of classes. output dimention of classifier
            size->int: the size of the correlation mask
            pretext->bool: whether train pretext task
            with_LIO->bool: whether train pretext task with LIO module
            M->bool: whether to train with pretext task LIO using M
            num_positive->int: number of positive images
            pretrain->bool: whether are the model's weights pretrained by ImageNet classification task
            config: mmsegmentation config file. Ref: https://github.com/open-mmlab/mmsegmentation
        """
        super(deeplab_resnet,self).__init__()
        self.num_positive = num_positive
        self.M = M
        self.stage = stage
        self.size = size
        self.with_LIO = with_LIO
        self.pretext = pretext
        
        # load deeplab encoder model from mmsegmentation toolkit
        cfg = Config.fromfile(config)
        cfg.gpu_ids = [0]
        setup_multi_processes(cfg)
        env_info_dict = collect_env()
        cfg.device = get_device()
        deeplab = build_segmentor(
                    cfg.model,
                    train_cfg=cfg.get('train_cfg'),
                    test_cfg=cfg.get('test_cfg'))
        if not pretrain:
            print('weight initialize...')
            deeplab.apply(self.weights_init)
        #deeplab.init_weights()
            
        # backbone
        self.backbone = deeplab.backbone
        self.flatten = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
        )
        self.avgpool = nn.AvgPool2d(8, 8)
        # projection head
        if self.pretext:
            self.projection_head = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
                nn.Linear(in_dim, out_dim),
            )
        # prediction head
        else:
            self.prediction_head = nn.Linear(in_dim, num_classes)
            
        # LIO model
        if stage == 3:
            self.feature_dim = 2048
            self.structure_dim = 1024
        elif stage == 2:
            self.feature_dim = 1024
            self.structure_dim = 512
        elif stage == 1:
            self.feature_dim = 512
            self.structure_dim = 256
        else:
            raise NotImplementedError("No such stage")
        if self.with_LIO: # call LIO module
            self.scl_lrx = SCLModule(self.size, self.feature_dim, self.structure_dim, avg=False, 
                                     num_positive = self.num_positive, M = self.M)

    def forward(self, x):
        # process backbone model
        x1,x2,x3,x5 = self.backbone(x)
        x_avg = self.avgpool(x5) # use average pooling to shrink feature size
        x6 = self.flatten(x5)    # flatten the feature to be feed to projection head or prediction head
        
        # process projection head or prediction head
        if self.training and self.pretext:
            if self.with_LIO:
                if self.stage == 3:
                    mask, object_extents, coord_loss, mask_loss = self.scl_lrx(x_avg)
                elif self.stage == 2:
                    mask, object_extents, coord_loss, mask_loss = self.scl_lrx(x4)
                elif self.stage == 1:
                    mask, object_extents, coord_loss, mask_loss = self.scl_lrx(x3)
                return mask, object_extents, coord_loss, mask_loss, self.projection_head(x6) # pretext training with LIO
            return self.projection_head(x6) # pretext training without LIO
        elif self.pretext: return self.projection_head(x6)
        return self.prediction_head(x6) # downstream training or validation  
    
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_uniform(m.weight)
            #torch.nn.init.uniform_(m.weight, a=0, b=0.001)
