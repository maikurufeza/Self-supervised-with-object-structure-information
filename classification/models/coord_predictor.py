import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelativeCoordPredictor(nn.Module):
    def __init__(self, in_dim, size=7):
        """module is used to predicted polar coordinate

        Attributes:
            size->int: the size of the correlation mask, also the size of the feature
            in_dim->int: the channel dimention of structure representation features
        """
        super().__init__()
        self.in_dim = in_dim
        self.size = size

        # predictor is used to predict polar coordinate from concatenated features
        # thus, input dimention will be 2*in_dim
        self.predictor = nn.Sequential(
            nn.Linear(2*in_dim, 2),
            nn.ReLU(),
        )
        
        self.basic_label = torch.from_numpy(self.build_basic_label(size)).float()# get each grid position to calculate polar coordinate (psuedo label)
        self.dist_loss_f = nn.MSELoss(reduction='none')
        self.curr_coef = 1.

    def forward(self, x, mask): # x:(N,C1,H,W), mask:(N,H,W)
        N, C, H, W = x.shape
        reduced_x = x.view(N, C, H*W).transpose(1, 2).contiguous()  # (N, S, C) # S=H*W

        mask = mask.view(N, H*W)
        #thresholds = torch.mean(mask, dim=1, keepdim=True)
        thresholds = self.curr_coef * torch.mean(mask, dim=1, keepdim=True)
        binary_mask = (mask > thresholds).float()
        binary_mask = binary_mask.view(N, H, W)

        masked_x = x * binary_mask.view(N, 1, H, W)
        masked_x = masked_x.view(N, C, H*W).transpose(1, 2).contiguous()  # (N, S, C)
        _, reduced_x_max_index = torch.max(torch.mean(masked_x, dim=-1), dim=-1)

        basic_index = torch.from_numpy(np.array([i for i in range(N)])).cuda()
        max_features = reduced_x[basic_index, reduced_x_max_index, :]  # (N, C)
        max_features_to_concat = max_features.unsqueeze(1).expand((N, H*W, C))

        # predict polar coordinate
        discriminative_feature = torch.cat((max_features_to_concat, reduced_x), dim=-1)  # (N, S, 2*C)
        discriminative_feature = discriminative_feature.view(N*H*W, 2*C)
        preds_coord = self.predictor(discriminative_feature)  # (N*H*W, 2)
        preds_coord = preds_coord.view(N, H*W, 2)
        preds_coord[basic_index, reduced_x_max_index, :] = 0.
        preds_coord = preds_coord.view(N, H, W, 2)

        # Build Label
        label = self.basic_label.cuda()
        label = label.unsqueeze(0).expand((N, H, W, 2)).view(N, H*W, 2)  # (N, S, 2)
        basic_anchor = label[basic_index, reduced_x_max_index, :].unsqueeze(1)  # (N, 2)
        relative_coord = label - basic_anchor # (N, S, 2)
        relative_coord = relative_coord / self.size # (N, S, 2)
        relative_dist = torch.sqrt(torch.sum(relative_coord**2, dim=-1))  # (N, S)
        relative_angle = torch.atan2(relative_coord[:, :, 1], relative_coord[:, :, 0])  # (N, S) in (-pi, pi)
        relative_angle = (relative_angle / np.pi + 1) / 2  # (N, S) in (0, 1)

        # Calc Loss
        preds_dist, preds_angle = preds_coord[:, :, :, 0], preds_coord[:, :, :, 1]

        preds_dist = preds_dist.view(N, H, W)
        relative_dist = relative_dist.view(N, H, W)
        dist_loss = self.dist_loss_f(preds_dist, relative_dist)

        preds_angle = preds_angle.view(N, H*W)
        gap_angle = preds_angle - relative_angle  # (N, S) in (0, 1) - (0, 1) = (-1, 1)
        gap_angle[gap_angle < 0] += 1
        gap_angle = gap_angle - torch.mean(gap_angle, dim=-1, keepdim=True)  # (N, H*W)
        gap_angle = gap_angle.view(N, H, W)
        angle_loss = torch.pow(gap_angle, 2)
        return dist_loss + angle_loss
    
    def build_basic_label(self, size):
        """ 
        Get all indices of size*size grid
        """
        basic_label = np.array([[(i, j) for j in range(size)] for i in range(size)])
        return basic_label
    
    def set_curr_coef(self, coef):
        self.curr_coef = coef
        
    def get_curr_coef(self):
        return self.curr_coef
