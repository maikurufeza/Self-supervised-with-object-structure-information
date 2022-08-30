# Self-supervised Segmentationwith LIO


basic tutorials about the usage of this code.


## Setup


### Environment


This code is tested with:
1. Python 3.8.5
2. Pytorch 1.8.0+cu111
3. mmsegmentation (https://github.com/open-mmlab/mmsegmentation)

Install mmsegmentation: https://github.com/open-mmlab/mmsegmentation/blob/master/docs/zh_cn/get_started.md#installation


## Train a model


First, Install `mmcv` and `mmsegmentation` from https://mmsegmentation.readthedocs.io/en/latest/get_started.html 


### Pretext task


For convenience, we use classification code to train pretext task. Because we just train pretext task, downstream_epochs is setted 0.


```shell
python pretrain_seg.py \
--train_dir ../dataset/VOC2012/VOCdevkit/VOC2012/JPEGImages/ \
--train_csv ../dataset/VOC2012/train.txt \
--valid_dir ../dataset/VOC2012/VOCdevkit/VOC2012/JPEGImages/ \
--valid_csv ../dataset/VOC2012/val.txt \
--is_pretext True --downstream_epochs 0 \
--log ./record/log/seg_simclr_with_warmup_lio.txt \
--save_dir ./record/weight/seg_simclr_with_warmup_lio/ \
--mask_dir ./record/mask/seg_simclr_with_warmup_lio/ \
--config ../MYSEG_SIMCLR_LIO/mmsegmentation/configs/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12.py \
--model_name deeplab --pretext_batch_size 4 --pretext_resolution 512 \
--cuda -1
```


where 

- `--config`: mmsegmentation config file.
- `--model_name`: set deeplab to train the encoder of segmentation


### downstream task


```shell
sh tools/dist_pretrain.sh \
configs/deeplabv3/deeplabv3_r50-d8_512x512_20k_voc12.py 1 \
--load_pretext ../classification/record/weight/seg_simclr_with_warmup_lio/deeplab_pretext.pth \
--work-dir work_dirs/simclr_pretrain_with_warmup_lio
```


where 

- `--load_pretext`: path of pretext task model.
- `--work-dir`: where to save model weight and log files.


For more other detailed parameters setting, see mmsegmentation tutorials (https://github.com/open-mmlab/mmsegmentation)
