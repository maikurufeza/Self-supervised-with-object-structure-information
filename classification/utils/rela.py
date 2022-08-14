import torch
import torch.nn as nn
import torch.nn.functional as F


def calc_loss(all_objects, all_masks):
    N, H, W = all_objects.shape
    all_objects_input = all_objects.view(N, H * W, 1)
    all_objects_target = all_masks.view(N, H * W, 1)
    all_loss = cosine_loss(all_objects_input, all_objects_target)
    return all_loss

def calc_mse_loss(all_objects, all_masks):
    """ calculate mse loss
    
    Args:
        all_objects->tensor: predicted object location
        all_masks->tensor: correlation mask

    Returns:
        tensor: mse loss
    """
    return (all_objects - all_masks)**2

def calc_mask(object_1, object_2, num_positive): #(BATCH_SIZE, H*W, c1), (NUM_POSITIVE*BATCH_SIZE, H*W, c1)
    """ phi function
    
    Args:
        object_1->tensor: first image feature
        object_2->tensor: first image feature
        num_positive->int: number of positive images 

    Returns:
        object_1_target: output of phi(object_1, object_2)
    """
    N_1, S_1, C_1 = object_1.shape # BATCH_SIZE, H*W, c1
    N_2, S_2, C_2 = object_2.shape # NUM_POSITIVE*BATCH_SIZE, H*W, c1
    object_1 = object_1.unsqueeze(1).expand((N_1, num_positive, S_1, C_1)).contiguous().view(N_1 * num_positive, S_1, C_1) #(NUM_POSITIVE*BATCH_SIZE, H*W, c1),
    relation = torch.matmul(object_1, object_2.transpose(1, 2)) / C_1  # (NUM_POSITIVE*BATCH_SIZE, H*W, H*W)
    localization_masks = torch.max(relation, dim=2)[0].unsqueeze(-1).view(N_1, num_positive, S_1) #(BATCH_SIZE, NUM_POSITIVE, H*W)
    object_1_target = torch.mean(localization_masks, dim=1) #(BATCH_SIZE, H*W)
    return object_1_target, localization_masks

def get_mask(all_objects, num_positive): # (BATCH_SIZE + NUM_POSITIVE*BATCH_SIZE, c1, h, w)
    """ calculate correlation mask
    
    Args:
        all_objects->tensor: all feature
        num_positive->int: number of positive images

    Returns:
        all_masks: correlation masks (M)
    """
    N, C, H, W = all_objects.shape
    N_1 = N // (num_positive + 1) # BATCH_SIZE
    N_2 = N_1 * num_positive      # NUM_POSITIVE*BATCH_SIZE
    S = H * W
    all_objects = all_objects.view(N, C, S).transpose(1, 2)
    all_objects = all_objects.view(N_1, num_positive + 1, S, C) #(BATCH_SIZE, 1+NUM_POSITIVE, H*W, c1)
    all_masks = torch.zeros(N_1, num_positive + 1, H, W).cuda() #(BATCH_SIZE, 1+NUM_POSITIVE, H, W)
    for i in range(num_positive + 1):
        main_index = torch.Tensor([i]).cuda().long()
        main_object = torch.index_select(all_objects, 1, main_index).view(N_1, S, C) #(BATCH_SIZE, H*W, c1)
        sub_indexs = torch.Tensor([k for k in range(num_positive + 1) if k != i]).cuda().long()
        sub_objects = torch.index_select(all_objects, 1, sub_indexs).view(N_1 * num_positive, S, C) #(NUM_POSITIVE*BATCH_SIZE, H*W, c1)
        main_mask, corr_masks = calc_mask(main_object, sub_objects, num_positive) #(BATCH_SIZE, H*W), (BATCH_SIZE, NUM_POSITIVE, H*W)
        all_masks[:, i, :, :] = main_mask.view(N_1, H, W)
    all_masks = all_masks.view(N_1 + N_2, H, W)
    return all_masks

def calc_rela(all_objects, pred_masks, num_positive): # (BATCH_SIZE + NUM_POSITIVE*BATCH_SIZE, c1, h, w), (BATCH_SIZE + NUM_POSITIVE*BATCH_SIZE, h, w)
    all_masks = get_mask(all_objects, num_positive) # phi function
    return calc_mse_loss(pred_masks, all_masks), all_masks
