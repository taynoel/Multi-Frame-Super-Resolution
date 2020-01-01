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
    parser.add_argument('--trainDataPath', type=str, default= 'D:/dataset/probav_data/train/NIRtrain',
                        help='Training dataset path')
    parser.add_argument('--outImgSize', type=int, default=384,
                        help='Output image size. Output an Hr image')
    parser.add_argument('--inImgPatchSize', type=int, default=96,
                        help='Input image patch size')
    parser.add_argument('--batchSize', type=int, default=12,
                        help='Batch size')
    parser.add_argument('--findBestLoss', type=bool, default=True,
                        help='Find HR and LR image shift difference that gives the best loss')
    
    parser.add_argument('--epochNum', type=int, default=5000,
                        help='Num of epochs')
    
    parser.add_argument('--mode', type=str, default="allU",
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
     lossList=[]
     _trainCount=0
     
     if options.mode=="primaryT":
         dataloader= DataLoader(
                 trainDataset,
                 batch_size=options.batchSize,
                 shuffle=True,
                 num_workers=0,
                 pin_memory=True,
                 collate_fn=trainDataset.collate_fnSingleImg)
         net.primaryNet.loadPretrainedParams("primaryParam.dict")
         optimizer=optim.Adam(net.primaryNet.parameters(),lr=0.001,eps=1e-4)
         
         for fEpoch in range(options.epochNum):
             dataGen=iter(dataloader)
             while True:
                 try:
                     targ,_,inp,_=next(dataGen)
                     targ=Variable(targ).to(device)
                     inp=Variable(inp).to(device)
                     optimizer.zero_grad()
                     stepOnce(net.primaryNet,[inp,targ],optimizer,"primary",lossList,fEpoch)
                     _trainCount+=1
                     if _trainCount%100==0:
                         torch.save(net.primaryNet.state_dict(),"primaryParam_%d.dict"%_trainCount)
                     
                 except StopIteration:
                     break
     elif options.mode=="stnT":
        dataloader= DataLoader(
             trainDataset,
             batch_size=options.batchSize,
             shuffle=True,
             num_workers=0,
             pin_memory=True,
             collate_fn=trainDataset.collate_fnSTN)
        net.primaryNet.loadPretrainedParams("primaryParam.dict")
        net.stnNet.loadPretrainedParams("stnParam.dict")
        optimizer=optim.Adam(net.stnNet.parameters(),lr=0.001,eps=1e-4)
        net.primaryNet.eval() #primary net is in evaluation mode
        
        for fEpoch in range(options.epochNum):
            dataGen=iter(dataloader)
            while True:
                try:
                    targ,_,inp,_,shftList=next(dataGen)
                    targ=Variable(targ).to(device)
                    inp=Variable(inp).to(device)
                    shftList=Variable(shftList).to(device)
                    optimizer.zero_grad()
                    
                    #obtain features
                    batchNum=targ.shape[0]
                    _inp=torch.cat([targ,inp],dim=0)
                    with torch.no_grad():
                        _out=net.primaryNet(_inp)
                    
                    #input reshapping
                    targF,inpF=_out.squeeze(2).detach().split(batchNum,dim=0)
                    _inp=torch.cat([targF,inpF],dim=1)
                    
                    
                    stepOnce(net.stnNet,[_inp,shftList],optimizer,"stn",lossList,fEpoch)
                    _trainCount+=1
                    if _trainCount%100==0:
                         torch.save(net.stnNet.state_dict(),"stnParam_%d.dict"%_trainCount)
                except StopIteration:
                    break
     elif options.mode=="allT":
        dataloader= DataLoader(
             trainDataset,
             batch_size=options.batchSize,
             shuffle=True,
             num_workers=0,
             pin_memory=True,
             collate_fn=trainDataset.collate_fnEnd2End)
        net.primaryNet.loadPretrainedParams("primaryParam.dict")#in case first time
        net.stnNet.loadPretrainedParams("stnParam.dict")#in case first time
        net.loadPretrainedParams("wholeParam.dict")#overwrite all net param if available
        
        optimizer=optim.Adam([
                {"params":net.primaryNet.parameters(),"lr":0.0001,"eps":1e-4},
                {"params":net.stnNet.parameters(),"lr":0.0001,"eps":1e-4},
                {"params":net.fusionNet.parameters(),"lr":0.001,"eps":1e-4},
                {"params":net.fusionRefineNet.parameters(),"lr":0.001,"eps":1e-4}
                ])
        
        for fEpoch in range(options.epochNum):
            dataGen=iter(dataloader)
            while True:
                try:
                    targ,_,inp,inpMask=next(dataGen)
                    targ=Variable(targ).to(device)
                    inp=Variable(inp).to(device)
                    optimizer.zero_grad()
                    
                    stepOnce(net,[inp,targ,inpMask],optimizer,"whole",lossList,fEpoch)
                    _trainCount+=1
                    if _trainCount%400==0:
                         torch.save(net.state_dict(),"wholeParam_%d.dict"%_trainCount)
                except StopIteration:
                    break
                
     elif options.mode=="allU":
         import cv2
         import random
         net.loadPretrainedParams("wholeParam.dict")
         net.eval()
         
         inDir="./inputImg"
         imgNameL=[f for f in os.listdir(inDir) if os.path.isfile(os.path.join(inDir,f))]
         imgNameL=random.sample(imgNameL,k=9)
         imgL=[]
         for fImN in imgNameL:
             img=Image.open(os.path.join(inDir,fImN)).resize((options.outImgSize,)*2,Image.BICUBIC)
             img=transforms.ToTensor()(img).type(torch.float32)/65536
             img=(img-0.1194)/0.0332
             imgL.append(img)
         imgL=Variable(torch.stack(imgL,dim=1).unsqueeze(0)).to(device)
         
         with torch.no_grad():
             out=net(imgL)
         
         regLrIm=out[1][0,0].cpu().numpy()    
         cv2.imshow("predicted",(out[2][0,0]*0.0332+0.1194).cpu().numpy()*3)
         cv2.imshow("mean",(np.mean(regLrIm,axis=0)*0.0332+0.1194)*3)
         
         
         
     elif options.mode=="primaryTesting":
         #!Note: Only for testing!!
        import cv2
        from model import findBestLoss
        dataloader= DataLoader(
                 trainDataset,
                 batch_size=options.batchSize,
                 shuffle=True,
                 num_workers=0,
                 pin_memory=True,
                 collate_fn=trainDataset.collate_fnSingleImg)
        net.primaryNet.loadPretrainedParams("primaryParam.dict")
        optimizer=optim.Adam(net.primaryNet.parameters(),lr=0.001,eps=1e-4)
         
        dataGen=iter(dataloader)
        targ,_,inp,_=next(dataGen)
        targ=Variable(targ).to(device)
        inp=Variable(inp).to(device)
        net.primaryNet.eval()
        with torch.no_grad():
            o=net.primaryNet(inp,targ)
        imgL=o[0]
        n=0
        print("Residual: ",torch.mean(o[0][0,0][50:90,50:90]))
        print("Mean bicubic loss: %.6f"%findBestLoss(inp[...,10:80,10:80],targ[...,10:80,10:80],net.lossCriterion)[2])
        print("SSIR loss: %.6f"%findBestLoss(imgL[...,10:80,10:80],targ[...,10:80,10:80],net.lossCriterion)[2])
        cv2.imshow("lr",(inp[n,0]*0.0332+0.1194).cpu().numpy()*3) 
        cv2.imshow("ssir",(imgL[n,0]*0.0332+0.1194).cpu().numpy()*3) 
        cv2.imshow("HR",(targ[n,0]*0.0332+0.1194).cpu().numpy()*3)
     
     else:#For testing
        import cv2
        from model import findBestLoss
        dataloader= DataLoader(
             trainDataset,
             batch_size=options.batchSize,
             shuffle=False,
             num_workers=0,
             pin_memory=True,
             collate_fn=trainDataset.collate_fnEnd2End)
        net.loadPretrainedParams("wholeParam.dict")#overwrite all net param if available
        
        dataGen=iter(dataloader)
        targ,_,inp,inpMask=next(dataGen)
        targ=Variable(targ).to(device)
        inp=Variable(inp).to(device)
        net.eval()
        with torch.no_grad():
            #o,loss=net(inp,targ)
            o=net(inp)
        imgL=o[2]
        n=0
        cv2.imshow("inputMean",(torch.mean(inp,dim=2)[n,0]*0.0332+0.1194).cpu().numpy()*3) 
        cv2.imshow("ssir",(imgL[n,0]*0.0332+0.1194).cpu().numpy()*3) 
        cv2.imshow("HR",(targ[n,0]*0.0332+0.1194).cpu().numpy()*3)
        print("Residual: ",torch.mean(o[0][0,0][50:90,50:90]))
        print("Mean Reg bicubic loss: %.6f"%findBestLoss(torch.mean(o[1],dim=2)[...,50:200,50:200],targ[...,50:200,50:200],net.lossCriterion)[2])
        print("Mean bicubic loss: %.6f"%findBestLoss(torch.mean(inp,dim=2)[...,50:200,50:200],targ[...,50:200,50:200],net.lossCriterion)[2])
        print("SSIR loss: %.6f"%findBestLoss(imgL[...,50:200,50:200],targ[...,50:200,50:200],net.lossCriterion)[2])
       
             
                 
     
         
         
     
     
     
     
     
     