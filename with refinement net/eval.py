import os
import torch
from torch.autograd import Variable
from model import findBestLoss,findBestLossForEval

from dataset import ListDataset
from torch.utils.data import DataLoader
import argparse
import model
import numpy as np
import cv2

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--findBestLoss', type=bool, default=True,
                        help='Find HR and LR image shift difference that gives the best loss')
    
   
    args = parser.parse_args()
    return args

if __name__=='__main__':
     options=parse_args()
     device = torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
     net=model.SuperResolNet(options)
     net.loadPretrainedParams("wholeParam.dict")
     net.to(device)
     
     testDat=torch.load("testDat")
     testDatNorm=torch.load("testNormDat")
     siz=384
     
     zScoreSum=torch.zeros(1)
     zScoreList=[]
     for fInd,fIm in enumerate(testDat):
         print(fInd)
         #fIm (1+9,2,384,384)
         HrImg,Hrmask=fIm[0] #(384,384)
         lrImg=fIm[1:]#(9,2,384,384)
         
         #cancel border pixels
         HrImg=HrImg[3:-3,3:-3]
         Hrmask=Hrmask[3:-3,3:-3]
         lrImg=lrImg[...,3:-3,3:-3]
         
         inp=Variable(lrImg[:,0,...].unsqueeze(0).unsqueeze(0).to(device))#[1, 1, 9, 384, 384]
         inp=inp[:,:,:,:siz,:siz]
         targ=Variable(HrImg.unsqueeze(0).unsqueeze(0).to(device))[...,:siz,:siz]
         mask=Variable(Hrmask.unsqueeze(0).unsqueeze(0).to(device))[...,:siz,:siz]
         
         
         net.eval()
         with torch.no_grad():
            #o,loss=net(inp,targ)
            o=net(inp)
         imgL=o[3]
         print("Mean Reg bicubic loss: %.6f"%findBestLossForEval(torch.mean(o[1],dim=2),targ,mask)[2])
         print("Mean bicubic loss: %.6f"%findBestLossForEval(torch.mean(inp,dim=2),targ,mask)[2])
         SSirLoss=findBestLossForEval(imgL,targ,mask)[2]
         print("SSIR loss: %.6f"%SSirLoss)
         
         SSirLoss=findBestLossForEval(imgL*0.0332+0.1194,targ*0.0332+0.1194,mask)[2]#re-normalize
         zScore=testDatNorm[fInd]/(-10*torch.log10(SSirLoss))
         print("ZScore: %.6f"%zScore.item())
         zScoreSum+=zScore
         zScoreList.append(zScore)
         
         print("Score: %.3f"%(zScoreSum/len(testDat)))
         
         
     
     