# SimICL

This is the repository of paper A Simple Framework for Visual In-context Learning  to Improve Ultrasound Segmentation.

Thanks to Zhenda Xie: code under "models" folder are mainly from Xie's paper: SimMIM: A Simple Framework for Masked Image Modeling, https://github.com/microsoft/SimMIM

## Introduction

SimICL is a simple framework combining visual in-context learning and masked image modeling. We considered image segmentation task as image inpainting, in which the model is shown an example of image input and output, and asked to paint the output of a new image. To achieve this, we concatenated a support pair consisting of image/mask with another query pair consisting of image/(mask). Random masking was added to the entire concatenated image before feeding the image to the model. 

Based on our experiments, we found that (1) random masking ratio 0.3-0.75 is crucial for query segmentation. (2) The support-query pair improved query segmentation (3) loss over masked areas or over the entire image led to the best performances. 

The experiment results demonstrated that (1) image reconstruction helped the model better understand image features, making the model achieving an excellent performance (2) model understood the support-query relationship.


## To run this code：

You need to prepare concatenated images/ground truth before running training code.

## Results

Please find the trained weights here. (Trained on wrist ultrasound dataset)
