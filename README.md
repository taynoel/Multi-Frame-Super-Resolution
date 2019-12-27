# Super-Resolution-PROBA-V-
## Overview
Non-sequential multi-frame super-resolution image generation is a method to register and fuse multiple images (normally of low quality) to recover its high resolution counterpart. This work is a PyTorch implementation with close reference to [DeepSum](https://github.com/diegovalsesia/deepsum) on [Proba-V satelite images](https://kelvins.esa.int/proba-v-super-resolution/home/) provided by ESA’s [Advanced Concepts Team](http://www.esa.int/gsp/ACT/index.html).

The network consists of three modules (Primary, Stn and Fusion net)(Corresponds to SISRNet, RegNet and FusionNet of DeepSum), which performs single image recovery, image/feature registration and multi-frame fusion respectively.
There are some changes, which include replacing Global Dynamic Convolution with spatial transformer network (to support registration given affine transformation, though current net only concentrate on translations), and the inclusion of structural similarity index measure for loss generation.

## Usage
### Train
Proba-V dataset can be obtained [here](https://kelvins.esa.int/proba-v-super-resolution/data/)
Set the training data path configurations through arguments before training:
```
"--trainDataPath" : Training set folder path (For example, D:/dataset/probav_data/train/NIR) 
```
Primary Net and Stn Net requires pretraining, before performing end-to-end training that includes Fusion Net. The training sequence is: Primary net pre-training, Stn Net pre-training and end-to-end training. Types of training can be specied by the following argument:
```
"--mode" : Mode of implementation, where
          "primaryT" for Primary Net pre-training
          "stnT" for Stn Net pre-training
          "allT" for end-to-end training
          "allU" for usage mode
```



