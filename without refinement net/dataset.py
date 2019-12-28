import os
from os.path import join
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import numpy as np
import random

"""
For reproducibility
"""
from random import seed as rndSeed
#rndSeed(5)
#torch.manual_seed(0)

class ListDataset(Dataset):
    class ImgList(object):
        def __init__(self,sceneDir,outImSiz,setRandom=False):
            self.sceneDir=sceneDir
            self.outImSiz=outImSiz
            self.setRandom=setRandom
            self.imPreprocessMode="constant"
            
            #Obtain list of LR image file names from scene directory
            self.lrFileList=[f for f in os.listdir(sceneDir) if os.path.isfile(join(sceneDir,f)) and "LR" in f]
        
        def _imPreprocess(self,img):
            if self.imPreprocessMode=="none":
                return img
            elif self.imPreprocessMode=="constant":
                return (img-0.1194)/0.0332
            elif self.imPreprocessMode=="normalize":
                mu=torch.mean(img)
                std=torch.std(img)
                return (img-mu)/std
        
        def __call__(self):
            """
            Returns HR image and mask of the scene
            """
            img=transforms.ToTensor()( Image.open(join(self.sceneDir,"HR.png"))).type(torch.float32)/65536
            img=self._imPreprocess(img)
            mask=transforms.ToTensor()( Image.open(join(self.sceneDir,"SM.png")))
            return [img,mask]
        
        def __getitem__(self,index):
            """
            Returns (index)th resized LR image and mask of the scene 
            """
            lrImgName=self.lrFileList[index]
            lrMskName=lrImgName.replace("LR","QM")
            
            img=Image.open(join(self.sceneDir, lrImgName)).resize((self.outImSiz,)*2,Image.BICUBIC)
            img=transforms.ToTensor()(img).type(torch.float32)/65536
            img=self._imPreprocess(img)
            
            mask=Image.open(join(self.sceneDir, lrMskName)).resize((self.outImSiz,)*2,Image.BICUBIC)
            mask=transforms.ToTensor()(mask)
            return [img,mask]
            
        def __len__(self):
            return len(self.lrFileList)
            
            
    def __init__(self,dataDirectory,opt,setRandom=False):
        self.opt=opt
        self.dataDirectory=dataDirectory
        self.outImSiz=opt.outImgSize
        self.inPtchSiz=opt.inImgPatchSize
        self.batchSiz=opt.batchSize
        self.setRandom=setRandom
        self.shiftMax=7
        
        #directory list of scenes
        self.sceneDirList=[join(self.dataDirectory,f) for f in os.listdir(self.dataDirectory) if os.path.isdir(join(self.dataDirectory,f))]
    
    def __len__(self):
        return len(self.sceneDirList)
    
    def __getitem__(self,index):
        return self.ImgList(self.sceneDirList[index],self.opt.outImgSize,self.setRandom)
    
    
    def collate_fnSTN(self,batch):
        """
        Produce samples with random patches for LR images and its shifted counterpart. Used for STN layer pretraining
        Output: [targetImgList, targetMaskList, inpImgList, inpMaskList,shftTarget]
        BacthNum:=number of samples per batch
        pcSize:=patch size
        channel:=channel depth
        targetImgList is a tensor with of HR image patch with shape (BatchNum,channel,pcSize,pcSize)
        targetMaskList is a tensor of HR image mask patch with shape (BatchNum,pcSize,pcSize)
        inpImgList is a tensor of LR images patch with shape (BatchNum,channel,pcSize,pcSize)
        inpMaskList is a tensor of LR image mask patch with shape (BatchNum,pcSize,pcSize)
        shftTarget is a tensor of target shifts with shape (BatchNum,2)
        """
        targetImgList=[]
        targetMaskList=[]
        inpImgList=[]
        inpMaskList=[]
        shftTarget=[]
        for fScene in batch:
            maxSiz=self.outImSiz-self.inPtchSiz
            ptx,pty=random.choices(range(maxSiz),k=2)
            shftH,shftW=random.choices(range(-self.shiftMax,self.shiftMax,1),k=2)
            sampleId=random.choice(range(len(fScene)))
            
            imgTarget=torch.stack(fScene[sampleId],dim=0) #(2,1,h,w)
            imgShift=imgTarget.clone()
            imgShift=torch.from_numpy(np.roll(np.roll(imgShift.numpy(),shftH,axis=-2),shftW,axis=-1))
            
            imT,maskT=torch.split(imgTarget[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz],1,dim=0)
            imS,maskS=torch.split(imgShift[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz],1,dim=0)
            
            targetImgList.append(imT)
            targetMaskList.append(maskT.squeeze(0))
            inpImgList.append(imS)
            inpMaskList.append(maskS.squeeze(0))
            
            #target[x,y]=inp[x+shftW,y+shftH]
            shftW=2*shftW/self.inPtchSiz #normalize for affine_grid
            shftH=2*shftH/self.inPtchSiz #normalize for affine grid
            shftTarget.append([shftW,shftH])#arranged in this way to have similar order as affine_grid input
        
        targetImgList=torch.cat(targetImgList,dim=0)
        targetMaskList=torch.cat(targetMaskList,dim=0)
        inpImgList=torch.cat(inpImgList,dim=0)
        inpMaskList=torch.cat(inpMaskList,dim=0)
        shftTarget=torch.Tensor(shftTarget)
        
        return targetImgList,targetMaskList,inpImgList,inpMaskList,shftTarget
  
    
    def collate_fnSingleImg(self,batch):
        """
        Produce samples with random patches for HR and LR images. Used for primary layer pretraining
        Output: [targetImgList, targetMaskList, inpImgList, inpMaskList]
        BacthNum:=number of samples per batch
        pcSize:=patch size
        channel:=channel depth
        targetImgList is a tensor with of HR image patch with shape (BatchNum,channel,pcSize,pcSize)
        targetMaskList is a tensor of HR image mask patch with shape (BatchNum,pcSize,pcSize)
        inpImgList is a tensor of LR images patch with shape (BatchNum,channel,pcSize,pcSize)
        inpMaskList is a tensor of LR image mask patch with shape (BatchNum,pcSize,pcSize)
        """
        targetImgList=[]
        targetMaskList=[]
        inpImgList=[]
        inpMaskList=[]
        for fScene in batch:
            maxSiz=self.outImSiz-self.inPtchSiz
            
            while True:#make sure samples are not masked too much
                ptx,pty=random.choices(range(maxSiz),k=2)
                sampleId=random.choice(range(len(fScene)))
                
                #!Note:for testing!
                #ptx,pty=18,105
                #print(ptx,pty,sampleId)
                
                img,mask=fScene()
                imgHr=img[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz]
                maskHr=mask[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz]
                
                
                img,mask=fScene[sampleId]
                imgLr=img[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz]
                maskLr=mask[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz]
                
                if torch.mean(maskHr)>0.8 and torch.mean(maskLr)>0.8:
                    break
            
            targetImgList.append(imgHr)
            targetMaskList.append(maskHr)
            inpImgList.append(imgLr)
            inpMaskList.append(maskLr)
            
        targetImgList=torch.stack(targetImgList,dim=0)
        targetMaskList=torch.stack(targetMaskList,dim=0).squeeze(1)
        inpImgList=torch.stack(inpImgList,dim=0)
        inpMaskList=torch.stack(inpMaskList,dim=0).squeeze(1)
        return targetImgList,targetMaskList,inpImgList,inpMaskList

        
    
    def collate_fnEnd2End(self,batch):
        """
        Produce samples with random patches for HR and LR images (lrNum images). Used for final end-to-end training
        
        Output: [targetImgList, targetMaskList, inpImgList, inpMaskList]
        BacthNum:=number of samples per batch
        pcSize:=patch size
        channel:=channel depth
        lrNum:=number of intended LR images
        targetImgList is a tensor with of HR image patch with shape (BatchNum,channel,pcSize,pcSize)
        targetMaskList is a tensor of HR image mask patch with shape (BatchNum,pcSize,pcSize)
        inpImgList is a tensor of LR images patch with shape (BatchNum,channel,lrNUm,pcSize,pcSize)
        inpMaskList is a tensor of LR image mask patch with shape (BatchNum,lrNum,pcSize,pcSize)
        """
        targetImgList=[]
        targetMaskList=[]
        inpImgList=[]
        inpMaskList=[]
        for fScene in batch:
            #get random top-left point of patch
            maxSiz=self.outImSiz-self.inPtchSiz
            
            while True:
                ptx,pty=random.choices(range(maxSiz),k=2)
                
                
                #obtain patch for img and mask of HR image
                img,mask=fScene()
                imgH=img[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz]
                maskH=mask[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz]
                if torch.mean(maskH)<=0.8: continue
                
                #obtain patch for img and mask of 9 LR images
                sampleIndList=random.sample(range(len(fScene)),k=9)
                _inpImgList=[]
                _inpMaskList=[]
                badMaskFlag=False
                for fSamp in sampleIndList:
                    img,mask=fScene[fSamp]
                    _inpImgList.append(img[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz])
                    _mask=mask[...,ptx:ptx+self.inPtchSiz,pty:pty+self.inPtchSiz]
                    if torch.mean(_mask)<=0.8:
                        badMaskFlag=True
                        break
                    _inpMaskList.append(_mask)
                if badMaskFlag: 
                    continue
                else: 
                    break
            
            targetImgList.append(imgH)
            targetMaskList.append(maskH)
            inpImgList.append(torch.stack(_inpImgList,dim=1))
            inpMaskList.append(torch.stack(_inpMaskList,dim=1))
            
        targetImgList=torch.stack(targetImgList,dim=0)
        targetMaskList=torch.stack(targetMaskList,dim=0).squeeze(1)
        inpImgList=torch.stack(inpImgList,dim=0)
        inpMaskList=torch.stack(inpMaskList,dim=0).squeeze()
        return targetImgList,targetMaskList,inpImgList,inpMaskList
            
            
        
        