import torch
from torch import nn

mse = nn.MSELoss()
soft = nn.Sequential(
            nn.Flatten(),
            nn.Softmax(dim=1)
        )


def object_extract(all_img_f,all_mask, NUM_POSITIVE, object_extractor, epoch):
    all_N,feature_dim, mask_H, mask_W = all_img_f.shape # all_N = (1+NUM_POSITIVE)*BATCH_SIZE
    N = all_N//(NUM_POSITIVE+1)
    all_imgs_trans = all_img_f.view(1+NUM_POSITIVE, N, feature_dim, mask_H, mask_W).transpose(0, 1).contiguous() #(BATCH_SIZE, 1+NUM_POSITIVE, feature_dim, H, W)
    all_imgs_trans = all_imgs_trans.view(N*(1+NUM_POSITIVE), feature_dim, mask_H, mask_W)  #(BATCH_SIZE(1+NUM_POSITIVE), feature_dim, H, W)

    all_mask_soft = soft(all_mask)                    # ((1+NUM_POSITIVE)*BATCH_SIZE , 8*8)
    all_mask_soft = all_mask_soft.view(all_N, mask_H, mask_W) # ((1+NUM_POSITIVE)*BATCH_SIZE , 8,8)

    all_mask_trans = all_mask_soft.view(1+NUM_POSITIVE, N, mask_H, mask_W).transpose(0, 1).contiguous() #(BATCH_SIZE, 1+NUM_POSITIVE, H, W)
    all_mask_trans = all_mask_trans.view(N*(1+NUM_POSITIVE),1,mask_H,mask_W) #(BATCH_SIZE*(1+NUM_POSITIVE),1, H, W)
    all_mask_trans = all_mask_trans.expand(N*(1+NUM_POSITIVE), feature_dim, mask_H, mask_W) #(BATCH_SIZE(1+NUM_POSITIVE), feature_dim, H, W)

    segmented_objects = all_mask_trans * all_imgs_trans                     # (BATCH_SIZE(1+NUM_POSITIVE), feature_dim, H, W)
    segmented_backgrounds = (1 - all_mask_trans) * all_imgs_trans           # (BATCH_SIZE(1+NUM_POSITIVE), feature_dim, H, W)
    object_features = object_extractor(segmented_objects)             # (BATCH_SIZE(1+NUM_POSITIVE),1000)
    object_features = object_features.view(N, (1+NUM_POSITIVE), -1)         # (BATCH_SIZE, 1+NUM_POSITIVE,1000)
    background_features = object_extractor(segmented_backgrounds)     # (BATCH_SIZE(1+NUM_POSITIVE),1000)
    background_features = background_features.view(N, (1+NUM_POSITIVE), -1) # (BATCH_SIZE, 1+NUM_POSITIVE,1000)

    object_loss = 0
    d_plus = 0
    if epoch < 1:
        for n in range(all_N):
            object_loss += calc_entropy(all_mask_soft[n,:,:])
        object_loss = object_loss/all_N/mask_H/mask_W/1000.
    else:
        
        for n in range(N):
            object_loss_batch = 0
            for q in range(1+NUM_POSITIVE):
                for j in range(1+NUM_POSITIVE):
                    if q == j:
                        continue
                    dij_plus = mse(object_features[n,q,:], object_features[n,j,:]) ** 2
                    dij_minus = mse(object_features[n,q,:], background_features[n,q,:]) ** 2
                    dij_minus = dij_minus + mse(object_features[n,j,:], background_features[n,j,:]) ** 2
                    dij_minus = dij_minus/2
                    object_loss_batch = object_loss_batch + (dij_plus/dij_minus)
            object_loss += object_loss_batch
        object_loss = object_loss / N
    return object_loss


def calc_entropy(input_tensor):
    input_tensor = input_tensor.double()
    entro = torch.where(input_tensor == 0., 0., (-1)*input_tensor*torch.log(input_tensor)) 
    return torch.sum(entro)
