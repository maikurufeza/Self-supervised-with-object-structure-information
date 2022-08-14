# Self-supervised-with-object-structure-information


## Description


This project is implementation of self-supervised learning with object structure information. We modify the LIO model (https://github.com/JDAI-CV/LIO) by replacing the supervised classifier in LIO with the self-supervised projection head in
SimCLR. This modification makes the model focus on the holistic object by learning object structure information without any supervised labels. We also propose a warm-up scheme to solve the dependency on pretraining model. Our experimental results show that our proposed model has the attention on the holistic objects in both pretext task and classification downstream task. The results also show that the object structure information help the model to get better classification accuracy. We also show our method is generalizable to other computer vision task
such as semantic segmentation.


## Result


![](../classification/record/gradcam/gradcam0.png)
![](../classification/record/gradcam/gradcam1.png)
![](../classification/record/gradcam/gradcam2.png)


## Citation


cite

```python

```
