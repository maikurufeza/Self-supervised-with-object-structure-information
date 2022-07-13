import numpy as np
import torch
from torch import nn
from torchvision import models

from .scl_module import SCLModule

class resnet_simclr_lio(nn.Module):
    def __init__(self, stage=3, in_dim=2048, out_dim=128, num_classes=200, size=7, with_LIO=True, attention="", M=True,
                 num_positive=2, pretext=True, pretrain=False, deeper = False):
        super(resnet_simclr_lio,self).__init__()
        self.num_positive = num_positive
        self.M = M
        self.attention = attention
        self.stage = stage
        self.size = size
        self.with_LIO = with_LIO
        self.pretext = pretext
        self.deeper = deeper
        
        resnet50 = models.resnet50(pretrained=pretrain)
        # backbone
        self.stage1_img = nn.Sequential(*list(resnet50.children())[:5])  # return (batch_size, 256, /4, /4)
        self.stage2_img = nn.Sequential(*list(resnet50.children())[5:6]) # return (batch_size, 512, /2, /2)
        self.stage3_img = nn.Sequential(*list(resnet50.children())[6:7]) # return (batch_size, 1024, /2, /2)
        self.stage4_img = nn.Sequential(*list(resnet50.children())[7])   # return (batch_size, 2048, /2, /2)
        self.adaptive_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=1),
            nn.Flatten(),
        )
        if self.deeper:
            self.mlp = nn.Sequential(
                nn.Linear(in_dim, in_dim),
                nn.ReLU(),
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
        if self.with_LIO:
            self.scl_lrx = SCLModule(self.size, self.feature_dim, self.structure_dim, avg=False, 
                                     attention=self.attention, num_positive = self.num_positive, M = self.M)

    def forward(self, x):
        x2 = self.stage1_img(x)
        x3 = self.stage2_img(x2)
        x4 = self.stage3_img(x3)
        x5 = self.stage4_img(x4)  
        x6 = self.adaptive_avg_pool(x5)
        if self.deeper:
            x6 = self.mlp(x6)
        
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
        checkpoint = torch.load(save_dir)
        stage1_img_dict = {}
        stage2_img_dict = {}
        stage3_img_dict = {}
        stage4_img_dict = {}
        if self.deeper:
            mlp_dict = {}
            
        for k in list(checkpoint['model'].keys()):
            if k.startswith('stage1_img'):
                stage1_img_dict[k[len('stage1_img.'):]] = checkpoint['model'][k]
            elif k.startswith('stage2_img'):
                stage2_img_dict[k[len('stage2_img.'):]] = checkpoint['model'][k]
            elif k.startswith('stage3_img'):
                stage3_img_dict[k[len('stage3_img.'):]] = checkpoint['model'][k]
            elif k.startswith('stage4_img'):
                stage4_img_dict[k[len('stage4_img.'):]] = checkpoint['model'][k]
            elif self.deeper and k.startswith('mlp.'):
                mlp_dict[k[len('mlp.'):]] = checkpoint['model'][k]
            del checkpoint['model'][k]
        msg = self.stage1_img.load_state_dict(stage1_img_dict, strict=False)
        msg = self.stage2_img.load_state_dict(stage2_img_dict, strict=False)
        msg = self.stage3_img.load_state_dict(stage3_img_dict, strict=False)
        msg = self.stage4_img.load_state_dict(stage4_img_dict, strict=False)
        if self.deeper:
            msg = self.mlp.load_state_dict(mlp_dict, strict=False)
        return 'Successfully load backbone model from {}'.format(save_dir)        
    
    def set_curr_coef(self, coef):
        self.scl_lrx.set_curr_coef(coef)
        
    def get_curr_coef(self):
        return self.scl_lrx.get_curr_coef()
