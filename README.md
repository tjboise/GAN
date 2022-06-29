
---
### Time Table:


- [x] Crop all the raw images into 256 * 256 (by 05/04/2022)
- [ ] classify the cropped images into **Alligator, Block, Longitudinal, Transverse**, Patching, Sealing, Manhole and no-defect image
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
|   Patching|             |         |        |
|   Sealing|             |         |        |
|   Manhole|             |         |        |
|   Normal|             |         |        |

note: DCGAN->https://github.com/t0nberryking/DCGAN256

--

- classification


| Categories  | Resnet50 |  VGG16  |  AlexNet  |  Resnet18|     
| ------------| ------------- |----|----|---|
| Raw  |            |       |           |  |
|   Raw + tranditional annotation|             |   |        |  |
|   Raw + GAN|             |              |       |  |
|   Raw + DCGAN|             |              |       |  |

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

https://github.com/Zeleni9/pytorch-wgan



### transfer from annotate to label

```

file_root = 'C:/Users/tjzha/Downloads/GAN_images-20220629T210557Z-001/GAN_images/annotate/'
file_list = os.listdir(file_root)
#filename='i3000.jpg'

for file in file_list:
    #print(file[:-4])
    img = cv2.imread(file_root+file,cv2.IMREAD_GRAYSCALE)
    ret2,th2 = cv2.threshold(img,230,255,cv2.THRESH_BINARY)
    cv2.imwrite(file_root+file[:-4]+'.png',th2)
    
```
