# Self-supervised-with-object-structure-information
Implementation of "Self-Supervised Object Structure Learning for Image Classification and Segmentation" paper


## Description


This project is implementation of self-supervised learning with object structure information. 


The modules in LIO (https://github.com/JDAI-CV/LIO) still need supervised labels training to provide basic image representations and have dependency on big human-annotated data pretraining model. Thus, we modify the model by replacing the supervised classifier in LIO with the self-supervised projection head in SimCLR. This modification makes the model focus on the holistic object by learning object structure information without any supervised labels. We also propose a warm-up scheme to solve the dependency on pretraining model. 


Our experimental results show that our proposed model has the attention on the holistic objects in both pretext task and classification downstream task. The results also show that the object structure information help the model to get better classification accuracy. We also apply our method to semantic segmentation.


## Get Start


Implementations of image classification are in [classification/](./classification/).

Implementations of semantic segmentation are in [segmentation/](./segmentation/).


## Result


![](./classification/record/gradcam/gradcam0.png)
![](./classification/record/gradcam/gradcam1.png)
![](./classification/record/gradcam/gradcam2.png)


## Citation


cite

```python

```
