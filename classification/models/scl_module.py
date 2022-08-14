import torch
import torch.nn as nn
import torch.nn.functional as F

from .coord_predictor import RelativeCoordPredictor as CoordPredictor

from utils.rela import calc_rela


class SCLModule(nn.Module):
    def __init__(self, size, feature_dim, structure_dim, *, avg, num_positive, M):
        """LIO modules
        Ref: https://github.com/JDAI-CV/LIO/tree/master/classification#readme

        Attributes:
            size->int: the size of the correlation mask, also the size of the feature
            feature_dim->int: the channel dimention of input features
            structure_dim->int: the channel dimention of structure representation features
            M->bool: whether to train LIO using M
            num_positive->int: number of positive images
            avg->bool: the use to adjust feature size. suggest false.
        """
        super().__init__()
        
        self.avg = avg
        self.size = size//2 if self.avg else size
        self.feature_dim = feature_dim
        self.structure_dim = structure_dim
        self.M = M

        # little network to transform feature to structure representation
        self.strutureg = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.feature_dim, self.structure_dim, 1, 1),
            nn.ReLU(),
        )
        # coordinate predictor to predict polar coordinate from structure representation
        self.coord_predictor = CoordPredictor(in_dim=self.structure_dim,
                                                size=self.size)
        
        # little network to predict object location
        self.maskg = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(self.feature_dim, 1, 1, 1),
        )

        if self.avg:
            self.avgpool = nn.AvgPool2d(2, 2)
        
        self.num_positive = num_positive
    
    def forward(self, feature):
        """
        Args:
            feature->tensor

        Returns:
            mask: predicted object location (m)
            object_extents: correlation masks (M)
            coord_loss: scl loss
            mask_loss: oel loss
        """
        if self.avg:
            feature = self.avgpool(feature)
        
        # OEL module
        mask = self.maskg(feature) 
        N, _, H, W = mask.shape
        mask = mask.view(N, H, W) # m
        # get oel loss and correlation mask (M).
        mask_loss, object_extents = calc_rela(feature, mask, self.num_positive) # (BATCH_SIZE + NUM_POSITIVE*BATCH_SIZE, 8,8)

        # SCL module
        structure_map = self.strutureg(feature)
        if self.M: # use M
            coord_loss = self.coord_predictor(structure_map, object_extents)
        else:      # use m
            coord_loss = self.coord_predictor(structure_map, mask)

        return mask, object_extents, coord_loss, mask_loss
