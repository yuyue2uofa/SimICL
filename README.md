# SimICL

This is the repository of paper A Simple Framework for Visual In-context Learning  to Improve Ultrasound Segmentation.

Thanks to Zhenda Xie: code under "models" folder are mainly from Xie's paper: SimMIM: A Simple Framework for Masked Image Modeling, https://github.com/microsoft/SimMIM

## Introduction

SimICL is a simple framework combining visual in-context learning and masked image modeling. We considered image segmentation task as image inpainting, in which the model is shown an example of image input and output, and asked to paint the output of a new image. To achieve this, we concatenated a support pair consisting of image/mask with another query pair consisting of image/(mask). Random masking was added to the entire concatenated image before feeding the image to the model. 

Based on our experiments, we found that (1) random masking ratio 0.3-0.75 is crucial for query segmentation. (2) The support-query pair improved query segmentation. (3) Loss over masked areas or over the entire image led to the best performances. 

The experiment results demonstrated that: with concatenated support examples and random masking, the model (1) learned the relationship between the support examples and the query images (2) uncover image features, allowing the precise segmentation on query images.

![Image text](https://github.com/yuyue2uofa/SimICL/blob/main/figures/figure1.jpg)
## To run this code：

Please prepare concatenated images/ground truth before running training/inference code.

## Results

Please find the trained weights [here](https://drive.google.com/file/d/1CJJiyNcC53chDeWVoqRgU493RyL3fWnX/view?usp=sharing). (The model with the best performance on wrist ultrasound dataset)
