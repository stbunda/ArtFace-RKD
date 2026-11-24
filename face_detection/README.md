# RetinaFace Face Detector

## Introduction
For the face detection we use RetinaFace from InsightFace: https://github.com/deepinsight/insightface

## Install
I advise you to run this part of the script in a Linux environment. 

1. Install MXNet with GPU support.
2. Type ``make`` to build cxx tools.

## Testing

Please check ``test_all.py`` for testing.

## RetinaFace Pretrained Models

Pretrained Model: RetinaFace-R50 ([baidu cloud](https://pan.baidu.com/s/1C6nKq122gJxRhb37vK0_LQ) or [googledrive](https://drive.google.com/file/d/1_DKgGxQWqlTqe78pw0KavId9BIMNUWfu/view?usp=sharing)) is a medium size model with ResNet50 backbone.
It can output face bounding boxes and five facial landmarks in a single forward pass.

WiderFace validation mAP: Easy 96.5, Medium 95.6, Hard 90.4. 

To avoid the confliction with the WiderFace Challenge (ICCV 2019), we postpone the release time of our best model.

## Third-party

[yangfly](https://github.com/yangfly): RetinaFace-MobileNet0.25 ([baidu cloud](https://pan.baidu.com/s/1P1ypO7VYUbNAezdvLm2m9w):nzof).
WiderFace validation mAP: Hard 82.5. (model size: 1.68Mb) 

[clancylian](https://github.com/clancylian/retinaface): C++ version

RetinaFace in [modelscope](https://modelscope.cn/models/damo/cv_resnet50_face-detection_retinaface/summary)

## References

```  
@inproceedings{Deng2020CVPR,
title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
booktitle = {CVPR},
year = {2020}
}
```


