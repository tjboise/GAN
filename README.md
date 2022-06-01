
---
### Time Table:


- [x] Crop all the raw images into 256 * 256 (by 05/04/2022)
- [ ] classify the cropped images into Alligator, Block, Longitudinal, Transverse, Patching, Sealing, Manhole and no-defect image
- [ ] Use GAN to train each of kind to produce fake images
- [ ] combine fake images and real images to train classification model: Resnet, ...
- [ ] write paper
---
Data:


| Categories  | Raw |  Tranditional annotation  |  DCGAN  |       
| ------------| ------------- |----|----|
| Transverse  |            |       |           |
|   Longitudinal|             |   |        |
|   Block|             |              |       |
|   Alligator|             |         |        |

--

| Categories  | Resnet50 |  VGG16  |  AlexNet  |       
| ------------| ------------- |----|----|
| Raw  |            |       |           |
|   Raw + tranditional annotation|             |   |        |
|   Raw + DCGAN|             |              |       |


----



#### Lab computer set up:
---

显卡驱动+cuda+cudnn安装： https://zhuanlan.zhihu.com/p/77874628

tensorflow-gpu安装：https://zhuanlan.zhihu.com/p/157473379

注意cuda，cudnn以及python版本相匹配：https://tensorflow.google.cn/install/source_windows

---> environment: gan 


## GAN


## DCGAN

train on you own dataset: https://www.topbots.com/step-by-step-implementation-of-gans-part-2/

