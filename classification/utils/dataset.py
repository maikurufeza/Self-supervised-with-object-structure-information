from __future__ import division
import os
import PIL.Image as Image
import pandas as pd

import torch
import torch.utils.data as data
from torchvision import transforms

class dataset(data.Dataset):
    def __init__(self, img_dir, anno_pd, preprocess, augment, totensor, pretext=False, valid=False, n_view = 2):
        self.img_dir = img_dir
        self.anno_pd = anno_pd
        self.paths = anno_pd['ImagePath'].tolist()
        self.labels = anno_pd['index'].tolist()
        self.preprocess = preprocess
        self.augment = augment
        self.totensor = totensor
        self.pretext = pretext
        self.valid = valid
        self.n_view = n_view
        
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, item):
        label = self.labels[item]-1
        img_path = os.path.join(self.img_dir, self.paths[item])
        org_img = self.pil_loader(img_path)
        pre_img = self.preprocess(org_img)
        
        if self.pretext:
            aug_imgs = [self.augment(pre_img) for i in range(self.n_view)]
            out_imgs = [self.totensor(aug_img) for aug_img in aug_imgs]
            return self.totensor(pre_img), out_imgs
        
        aug_img = self.augment(pre_img)
        out_img = self.totensor(aug_img)
        if self.valid:
            return out_img, label
        return self.totensor(pre_img), out_img, label
            
    def pil_loader(self,imgpath):
        with open(imgpath, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB') #(modified) RBG -> L

def collate_fn_valid(batch):
    imgs = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        labels.append(sample[1])
    return torch.stack(imgs, 0), labels


def collate_fn_pretext(batch):
    imgs = []
    aug_imgs = []
    for sample in batch:
        imgs.append(sample[0])
        aug_imgs.extend(sample[1])
    return torch.stack(imgs, 0), torch.stack(aug_imgs, 0)


def collate_fn_train(batch):
    imgs = []
    aug_imgs = []
    labels = []
    for sample in batch:
        imgs.append(sample[0])
        aug_imgs.append(sample[1])
        labels.append(sample[2])
    return torch.stack(imgs, 0), torch.stack(aug_imgs, 0), labels
