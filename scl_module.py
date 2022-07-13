import torch
import torch.nn as nn
import torch.nn.functional as F

from .coord_predictor import RelativeCoordPredictor as CoordPredictor
from .coordinate_attention import CA_Block
from .CBAM import CBAM

from utils.rela import calc_rela


class SCLModule(nn.Module):
    def __init__(self, size, feature_dim, structure_dim, *, avg, attention = "", num_positive, M):
        super().__init__()
        
        self.avg = avg
        self.size = size//2 if self.avg else size
        self.feature_dim = feature_dim
        self.structure_dim = structure_dim
        self.attention = attention
        self.M = M

        self.strutureg = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.feature_dim, self.structure_dim, 1, 1),
            nn.ReLU(),
        )
        self.coord_predictor = CoordPredictor(in_dim=self.structure_dim,
                                                size=self.size)
        self.ca = CA_Block(channel=self.feature_dim, h=self.size, w=self.size)
        self.cbam = CBAM(channel=self.feature_dim)
        self.maskg = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.feature_dim, 1, 1, 1),
        )

        if self.avg:
            self.avgpool = nn.AvgPool2d(2, 2)
        
        self.num_positive = num_positive
    
    def forward(self, feature):
        if self.avg:
            feature = self.avgpool(feature)
        if self.attention == "CA":
            feature = self.ca(feature)
        elif self.attention == "CBAM":
            feature = self.cbam(feature)
            
        mask = self.maskg(feature)
        N, _, H, W = mask.shape
        mask = mask.view(N, H, W)
        
        mask_loss, object_extents = calc_rela(feature, mask, self.num_positive) # (BATCH_SIZE + NUM_POSITIVE*BATCH_SIZE, 8,8)

        structure_map = self.strutureg(feature)
        if self.M:
            coord_loss = self.coord_predictor(structure_map, object_extents)
        else:
            coord_loss = self.coord_predictor(structure_map, mask)

        return mask, object_extents, coord_loss, mask_loss
    
    def set_curr_coef(self, coef):
        self.coord_predictor.set_curr_coef(coef)
        
    def get_curr_coef(self):
        return self.coord_predictor.get_curr_coef()
