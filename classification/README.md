# Self-supervised Classification with LIO


basic tutorials about the usage of this code


## Setup


### Environment


This code is tested with:
1. Python 3.8.5
2. Pytorch 1.8.0+cu111


### Dataset


1. Download correspond dataset to a folder. For example, the image folder is "../dataset/CUB_200_2011/images/"
2. Create a csv file which contain "ImagePath" column and its corresponding "index". For example, the csv file is "../dataset/CUB_200_2011/training.csv", the following is an example of its content.


```shell
ImagePath,index
001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg,1
001.Black_footed_Albatross/Black_Footed_Albatross_0074_59.jpg,1
002.Laysan_Albatross/Laysan_Albatross_0003_1033.jpg,2
003.Sooty_Albatross/Sooty_Albatross_0038_1065.jpg,3
007.Parakeet_Auklet/Parakeet_Auklet_0078_2004.jpg,7
...
```


Then the code will fetch first data from "../dataset/CUB_200_2011/images/001.Black_footed_Albatross/Black_Footed_Albatross_0009_34.jpg" (image folder + ImagePath in csv file) and its label is 1.


Note: The label is started with 1.


## Train a model


Call this function in terminal to start training:


```shell
python train.py \
--train_dir ../dataset/CUB_200_2011/images/ \
--train_csv ../dataset/CUB_200_2011/training.csv \
--valid_dir ../dataset/CUB_200_2011/images/ \
--valid_csv ../dataset/CUB_200_2011/testing.csv \
--num_classes 200
```


where

- `--train_dir`: directory of training dataset
- `--train_csv`: path of training csv file 
- `--valid_dir`: directory of testing dataset 
- `--valid_csv`: path of testing csv file
- `--num_classes`: number of classed in your dataset

<!-- #region -->
Optional arguments are:

- `--cuda`: -1: using all GPU for training. otherwise: using one selected GPU. default: 0


- `--save_dir`: where the best model's weights save. If you don't set this argment, model's weights will be saved to ./record/weight/{current time}/
- `--log`: where you save the log file. If you don't set this argment, log file will be written into ./record/log/{current time}.txt\n
- `--mask_dir`: where correlation masks save when LIO model is open. If you don't set this argment, masks will save to ./record/mask/{current time}/


- `--is_pretext`: whether to train pretext task
- `--with_LIO`: whether to train pretext task with LIO
- `--is_lio_loss_warmup`: whether to use LIO loss parameter loss warm-up scheme


- `--load_pretext`: the file path of the pretext model you want to load before training downstream task. If you want to load pretext task, is_pretext must be False
<!-- #endregion -->

For more other detailed hyper-parameters setting, see help messages: 


```shell
python train.py --help
```


### GradCAM


Suppose you want to see GradCaM from 4 model you trained. You can call the following code to the terminal:


```shell
python gradcam.py \
--image_dir ../dataset/CUB_200_2011/images/ \
--image_csv ../dataset/CUB_200_2011/testing.csv \
--model_dirs \
./record/weight/05201958/resnet_pretext.pth \
./record/weight/05251729/resnet_pretext.pth \
./record/weight/05240233/resnet_downstream.pth \
./record/weight/05261448/resnet_downstream.pth \
--titles pretext_SimCLR pretext_LIO downstream_SimCLR downstream_LIO \
--save_dir ./record/gradcam/images/ \
--num_cam 2
```


where

- `--image_dir`: directory of images
- `--image_csv`: path of image csv file 
- `--model_dirs`: the pathes of models' weight. You can input multiple pathes.
- `--titles`: titles of each images. len(titles) must equal to len(model_dirs)
- `--save_dir`: where to save your GradCAM results
- `--num_cam`: number of output GradCAM


For more other detailed parameters setting, see help messages: 


```shell
python gradcam.py --help
```
