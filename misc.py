"""
0 mean for input
sigmoid for stn output
cascaded refinement
inherent img shifts define the limit of clarity(need to fix my program)


to keep cuda out of memory: try stn with stride



Done:
stn net
structural similarity loss
use mask to discard bad data
"""

import os
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from PIL import Image

from dataset import ListDataset
from torch.utils.data import DataLoader
import argparse
import model
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trainDataPath', type=str, default='D:/dataset/probav_data/train/NIRtrain',
                        help='Training dataset path')
    parser.add_argument('--outImgSize', type=int, default=384,
                        help='Output image size. Output an Hr image')
    parser.add_argument('--inImgPatchSize', type=int, default=128,
                        help='Input image patch size')
    parser.add_argument('--batchSize', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--findBestLoss', type=bool, default=True,
                        help='Find HR and LR image shift difference that gives the best loss')
    
    parser.add_argument('--mode', type=str, default="primaryTesting",
                        help='mode: primaryT,stnT,allT,allU')
    args = parser.parse_args()
    return args


def stepOnce(net,inputs,optimizer,netName="unknown",lossList=[],epoch=0):
     dat,loss=net(*inputs)
     loss.backward()
     optimizer.step()
     print("Epoch %d %s net training loss: %.5f"%(epoch,netName,loss.item()))
     lossList.append(loss.item())
     return dat,loss

if __name__=='__main__':
     options=parse_args()
     device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
     net=model.SuperResolNet(options)
     net.to(device)
     trainDataset=ListDataset(options.trainDataPath,options)
     
     #meanList=[]
     #stdList=[]
     allImgList=[]
     for ind,f1 in enumerate(trainDataset):
         imgList=trainDataset[ind]
         for ind2,f2 in enumerate(imgList):
             img,mask=imgList[ind2]
             #a=img[mask>0.5]
             #mu=torch.mean(a)
             #std=torch.std(a)
             #meanList.append(mu)
             #stdList.append(std)
             allImgList.append(img)