# Super-Resolution-PROBA-V-
## Overview
Non-sequential multi-frame super-resolution image generation is a method to register and fuse multiple images (normally of low quality) to recover its high resolution counterpart. This work is a PyTorch implementation with close reference to [DeepSum](https://github.com/diegovalsesia/deepsum) on [Proba-V satelite images](https://kelvins.esa.int/proba-v-super-resolution/home/) provided by ESA’s [Advanced Concepts Team](http://www.esa.int/gsp/ACT/index.html).

There are two models: without and with refinement net. The networks consist of three modules (Primary, Stn and Fusion net)(Corresponds to SISRNet, RegNet and FusionNet of DeepSum), which performs single image recovery, image/feature registration and multi-frame fusion respectively. The network with refinement net has an additional layer very similar to Fusion net, which takes the residual output from fusion net as input to further refine the output.
There are some changes, which include replacing Global Dynamic Convolution with spatial transformer network (to support registration given affine transformation, though current net only concentrate on translations), and the inclusion of structural similarity index measure for loss generation.

## Implementation
All program is implemented via main.py
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
```

### Usage
One can use the trained network to perform multi-frame super-resolution bu first placing images in ./inputImg folder (minimum 9), and execute main.py with argument 
```
"--mode" = "allU"
```
Trained parameter file "wholeParam.dict" and example images in ./inputImg folder are included.

#### Output

##### Example 1
<p float="left">
  <img src="ref/ZexIm.jpg" width="200" />
  <img src="ref/mean.jpg" width="200" /> 
  <img src="ref/predicted.jpg" width="200" />
  <img src="ref/ZimHR.jpg" width="200" />
</p>
From left to right: 1. One of the low resolution images 2. Bicubic upsampling + mean 3. Reconstructed image 4. Gound truth high resolution image

##### Example 2
<p float="left">
  <img src="ref/ZexIm2.jpg" width="200" />
  <img src="ref/mean2.jpg" width="200" /> 
  <img src="ref/predicted2.jpg" width="200" />
  <img src="ref/ZimHR2.jpg" width="200" />
</p>
From left to right: 1. One of the low resolution images 2. Bicubic upsampling + mean 3. Reconstructed image 4. Gound truth high resolution image

 
### Test
Comparison test is performed between refined and non-refined output. From Proba-V training dataset folder, 156 scenes are selected from NIR, and 149 samples are selected from RED for testing, the remaining are used for training. The scores based on [here](https://kelvins.esa.int/proba-v-super-resolution/scoring/)are as follows:

| Measure  | Before Refinement |After Refinement |
| ------------- | ------------- |------------- |
| NIR  | 0.9861 |0.9819  |
| RED  | 0.9934  |0.9882  |

