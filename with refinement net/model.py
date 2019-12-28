import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

    
def findBestLoss(hrIm,target,lossCriterion):
    _b,_c,_h,_w=hrIm.shape
    x1,y1,x2,y2=6,6,_w-6,_h-6
    target=target[...,y1:y2,x1:x2]
    
    shftImL=[]
    pixLossL=[]
    for fBatchInd,fIm in enumerate(hrIm):
        pixLoss=1e10
        for fx in range(-6,7,1):
            for fy in range(-6,7,1):
                _hrIm=fIm[...,y1+fy:y2+fy,x1+fx:x2+fx]
                
                equalizer=torch.mean(target[fBatchInd]-_hrIm).detach()
                _loss=torch.mean(lossCriterion(_hrIm+equalizer,target[fBatchInd]))
                
                if _loss<pixLoss:
                    pixLoss=_loss
                    bestHrIm=_hrIm
                    
        shftImL.append(bestHrIm)
        pixLossL.append(pixLoss)
        
        
    hrIm=torch.stack(shftImL,dim=0)
    pixLossL=torch.stack(pixLossL,dim=0)
    pixLoss=torch.mean(pixLossL)
    return hrIm, target,pixLoss


def findBestLossForEval(hrIm,target,targetMask):
    lossCriterion=nn.MSELoss(reduction="none")
    _b,_c,_h,_w=hrIm.shape
    x1,y1,x2,y2=6,6,_w-6,_h-6
    target=target[...,y1:y2,x1:x2]
    targetMask=targetMask[...,y1:y2,x1:x2]
    
    shftImL=[]
    pixLossL=[]
    for fBatchInd,fIm in enumerate(hrIm):
        pixLoss=1e10
        for fx in range(-6,7,1):
            for fy in range(-6,7,1):
                _hrIm=fIm[...,y1+fy:y2+fy,x1+fx:x2+fx]
                
                equalizer=torch.mean(target[fBatchInd]-_hrIm).detach()
                msk=targetMask[fBatchInd]>0.5
                lossC=lossCriterion((_hrIm+equalizer)[msk],target[fBatchInd][msk])
                _loss=torch.mean(lossC)
                
                if _loss<pixLoss:
                    pixLoss=_loss
                    bestHrIm=_hrIm
                    
        shftImL.append(bestHrIm)
        pixLossL.append(pixLoss)
        
        
    hrIm=torch.stack(shftImL,dim=0)
    pixLossL=torch.stack(pixLossL,dim=0)
    pixLoss=torch.mean(pixLossL)
    return hrIm, target,pixLoss


class PrimaryNet(nn.Module):
    def __init__(self,opt):
        super(PrimaryNet,self).__init__()
        self._opt=opt
        self.lossCriterion=nn.MSELoss()
        repPad=nn.ReplicationPad3d((1,1,1,1,0,0))
        
        self.conv = nn.Sequential(
                repPad, nn.Conv3d(1,64,(1,3,3),stride=1,padding=0),nn.InstanceNorm3d(64),nn.LeakyReLU(),
                repPad, nn.Conv3d(64,64,(1,3,3),stride=1,padding=0),nn.InstanceNorm3d(64),nn.LeakyReLU(),
                repPad, nn.Conv3d(64,64,(1,3,3),stride=1,padding=0),nn.InstanceNorm3d(64),nn.LeakyReLU(),
                repPad, nn.Conv3d(64,64,(1,3,3),stride=1,padding=0),nn.InstanceNorm3d(64),nn.LeakyReLU(),
                repPad, nn.Conv3d(64,64,(1,3,3),stride=1,padding=0),nn.InstanceNorm3d(64),nn.LeakyReLU(),
                repPad, nn.Conv3d(64,64,(1,3,3),stride=1,padding=0),nn.InstanceNorm3d(64),nn.LeakyReLU(),
                repPad, nn.Conv3d(64,64,(1,3,3),stride=1,padding=0),nn.InstanceNorm3d(64),nn.LeakyReLU(),
                repPad, nn.Conv3d(64,64,(1,3,3),stride=1,padding=0),nn.InstanceNorm3d(64),nn.LeakyReLU())
        
        self.convProj=nn.Sequential(repPad, nn.Conv3d(64,1,(1,3,3),stride=1,padding=0))
        
    def loadPretrainedParams(self,paramFile):
        deviceBool=next(self.parameters()).is_cuda
        device=torch.device("cuda:0" if deviceBool else "cpu")
        try:
            pretrainedDict=torch.load(paramFile,map_location=device.type)
            modelDict=self.state_dict()
            pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
            modelDict.update(pretrainedDict)
            self.load_state_dict(modelDict)
        except:
            print("Can't load pre-trained PrimaryNet parameter files")
        
    def forward(self,xIn,target=None):
        """
        Input and output are tensors of shape (BatchNum,channel,depth,height,width). 
        Shape (BatchNum,channel,height,width) input tensor will be modified by adding depth=1
        """
        assert (len(xIn.shape) in [4,5]),"Invalid input tensor shape"
        
        x=xIn
        if len(x.shape)<5:
            x=x.unsqueeze(2)
        x=self.conv(x)    
        
        
        if target is None:
            return x
        else:
            x=self.convProj(x).squeeze(2)+xIn
            _,_,loss=findBestLoss(x,target,self.lossCriterion)
            #loss=self.lossCriterion(x,target)    
            return x,loss


class STNNet(nn.Module):
    def __init__(self,opt):
        super(STNNet,self).__init__()
        self._opt=opt
        self.lossCriterion=nn.MSELoss()
        
        self.conv0 = nn.Sequential(
                 nn.Conv2d(128,128,(3,3),stride=2,padding=(1,1)),nn.LeakyReLU(),
                 nn.Conv2d(128,64,(3,3),stride=1,padding=(1,1)),nn.LeakyReLU(),
                 nn.Conv2d(64,64,(3,3),stride=1,padding=(1,1)),nn.LeakyReLU(),
                 nn.Conv2d(64,64,(3,3),stride=1,padding=(1,1)),nn.LeakyReLU())
        self.conv1 = nn.Conv2d(64, 2, 3, 1, 1)
        
    def loadPretrainedParams(self,paramFile):
        deviceBool=next(self.parameters()).is_cuda
        device=torch.device("cuda:0" if deviceBool else "cpu")
        try:
            pretrainedDict=torch.load(paramFile,map_location=device.type)
            modelDict=self.state_dict()
            pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
            modelDict.update(pretrainedDict)
            self.load_state_dict(modelDict)
        except:
            print("Can't load pre-trained StnNet parameter files")
            
    def forward(self,xIn,target=None):
        """
        Input is a tensor of shape (BatchNum,channel,height,width). 
        Outputs tensor of shape (BatchNum,2) where the 2 elements are horizontal and vertical shifts
        """
        x=self.conv1(self.conv0(xIn))
        shftRes=nn.AvgPool2d((x.shape[2],x.shape[3]))(x).view(-1,2)
        if target is None:
            return shftRes
        else:
            loss=self.lossCriterion(shftRes,target)
            return shftRes,loss
        

class FusionNet(nn.Module):
    """
    Note: This network model is dependent on the number of LR samples per superresolution 
    """
    def __init__(self,opt):
        super(FusionNet,self).__init__()
        self._opt=opt
        repPad=nn.ReplicationPad3d((1,1,1,1,0,0))
        self.conv0=nn.Sequential(repPad, nn.Conv3d(64,64,(3,3,3),stride=1,padding=0),
                                 nn.InstanceNorm3d(64),nn.LeakyReLU())
        self.conv1=nn.Sequential(repPad, nn.Conv3d(64,64,(3,3,3),stride=1,padding=0),
                                 nn.InstanceNorm3d(64),nn.LeakyReLU())
        self.conv2=nn.Sequential(repPad, nn.Conv3d(64,64,(3,3,3),stride=1,padding=0),
                                 nn.InstanceNorm3d(64),nn.LeakyReLU())
        self.conv3=nn.Sequential(repPad, nn.Conv3d(64,64,(3,3,3),stride=1,padding=0),
                                 nn.InstanceNorm3d(64),nn.LeakyReLU())
        
        self.conv4=nn.Conv3d(64,1,(1,3,3),stride=1,padding=(0,1,1))
    
    def forward(self,xIn,target=None):
        """
        Input is a tensor of shape (BatchNum,channel,lrNum,height,width). 
        Outputs tensor of shape (BatchNum,channel,height,width)
        """
        x=self.conv0(xIn)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        return x.squeeze(2)

class FusionRefineNet(nn.Module):
    """
    Note: This network model is dependent on the number of LR samples per superresolution 
    """
    def __init__(self,opt):
        super(FusionRefineNet,self).__init__()
        self._opt=opt
        repPad=nn.ReplicationPad3d((1,1,1,1,0,0))
        self.conv0=nn.Sequential(repPad, nn.Conv3d(65,64,(3,3,3),stride=1,padding=0),
                                 nn.InstanceNorm3d(64),nn.LeakyReLU())
        self.conv1=nn.Sequential(repPad, nn.Conv3d(64,64,(3,3,3),stride=1,padding=0),
                                 nn.InstanceNorm3d(64),nn.LeakyReLU())
        self.conv2=nn.Sequential(repPad, nn.Conv3d(64,64,(3,3,3),stride=1,padding=0),
                                 nn.InstanceNorm3d(64),nn.LeakyReLU())
        self.conv3=nn.Sequential(repPad, nn.Conv3d(64,64,(3,3,3),stride=1,padding=0),
                                 nn.InstanceNorm3d(64),nn.LeakyReLU())
        
        self.conv4=nn.Conv3d(64,1,(1,3,3),stride=1,padding=(0,1,1))
    
    def forward(self,xIn,target=None):
        """
        Input is a tensor of shape (BatchNum,channel,lrNum,height,width). 
        Outputs tensor of shape (BatchNum,channel,height,width)
        """
        x=self.conv0(xIn)
        x=self.conv1(x)
        x=self.conv2(x)
        x=self.conv3(x)
        x=self.conv4(x)
        return x.squeeze(2)
       

class SuperResolNet(nn.Module):
    def __init__(self,opt):
        super(SuperResolNet,self).__init__()
        self._opt=opt
        self.device=torch.device("cuda:0" if torch.cuda.is_available()  else "cpu")
        self.lossCriterion=nn.MSELoss(reduction="none")
        
        self.primaryNet=PrimaryNet(opt).to(self.device)
        self.stnNet=STNNet(opt).to(self.device)
        self.fusionNet=FusionNet(opt).to(self.device)
        self.fusionRefineNet=FusionRefineNet(opt).to(self.device)
        self.findBestLossFlag=opt.findBestLoss
    
    def _genTransMat_onlyShift(self, inp):
        """
        input tensor is of shape (batchNum,2), where the 2 elemens are vertical and horizontal shifts
        outSize is the output tensor size
        """
        inp=inp.unsqueeze(-1) #(batchNum,2,1)
        batchNum=inp.shape[0]
        iden=torch.eye(2).unsqueeze(0).repeat(batchNum,1,1).to(self.device) #(bacthNum,2,2)
        return torch.cat([iden,inp],dim=-1) #(batchNum,2,3)
    
    def loadPretrainedParams(self,paramFile):
        deviceBool=next(self.parameters()).is_cuda
        device=torch.device("cuda:0" if deviceBool else "cpu")
        try:
            pretrainedDict=torch.load(paramFile,map_location=device.type)
            modelDict=self.state_dict()
            pretrainedDict = {k: v for k, v in pretrainedDict.items() if k in modelDict}
            modelDict.update(pretrainedDict)
            self.load_state_dict(modelDict)
        except:
            print("Can't load pre-trained wholeNet parameter files")    
            
    
        
    def forward(self,xIm,target=None,inpMask=None,hrMask=None):
        #Primary net
        x=self.primaryNet(xIm) #outp shape (batchnum,chan,lrNum,h,w)
        
        #Reshape
        refxOri=x[...,0:1,:,:]
        refy=x[...,1:  ,:,:]
        refx=refxOri.repeat(*(1,1,refy.shape[2],1,1))
        xReSh=torch.cat([refx,refy],dim=1)  #outp shape (batchnum,2*chan,lrNum,h,w)
        #Note: diffferent from paper, which uses 3D conv before 2D conv 
        
        #Reshape again for 2D conv
        _batchNum,_chan,_lrNum,_h,_w=xReSh.shape
        xReSh=xReSh.permute(0,2,1,3,4).contiguous().view(-1,_chan,_h,_w) #outp shape (batchnum*lrNum,2*chan,h,w)
        
        #stn net
        _w=xReSh.shape[-1]
        xReSh=self.stnNet(xReSh)
        
        #min,max clip
        xReSh=torch.clamp(xReSh,min=-4/_w,max=4/_w)
        
        
        #generate affine matrix
        affMat=self._genTransMat_onlyShift(xReSh)#outp shape (batchNum*lrNum,2,3)
        
        #register index>0 LR image feature to image feature of index 0
        _batchNum,_chan,_lrNum,_h,_w=refy.shape
        refy=refy.permute(0,2,1,3,4).contiguous().view(-1,_chan,_h,_w)#outp shape (batchnum*lrNum,chan,h,w)
        grid=F.affine_grid(affMat, refy.size())
        refy=F.grid_sample(refy, grid)
        
        #reshape feature tensor back to normal
        refy=refy.view(_batchNum,_lrNum,_chan,_h,_w).permute(0,2,1,3,4)
        regFeat=torch.cat([refxOri,refy],dim=2)
        
        #Fusion net
        hfResidual=self.fusionNet(regFeat) #outp shape (batchNum, chan=1,h,w)
        
        #Fusion Refine Net
        hfResidualExp=hfResidual.unsqueeze(2).repeat([1,1,regFeat.shape[2],1,1])
        hfResidual2=self.fusionRefineNet(torch.cat([hfResidualExp,regFeat],dim=1))
        
        
        #register index>0 LR image to image of index 0
        _batchNum,_chan,_lrNum,_h,_w=xIm[:,:,1:,...].shape
        regIm=xIm[:,:,1:,...].permute(0,2,1,3,4).contiguous().view(-1,_chan,_h,_w)#outp shape (batchnum*lrNum,chan,h,w)
        grid=F.affine_grid(affMat, regIm.size())
        regIm=F.grid_sample(regIm, grid)
        regIm=regIm.view(_batchNum,_lrNum,_chan,_h,_w).permute(0,2,1,3,4)
        regIm=torch.cat([xIm[:,:,0:1,...],regIm],dim=2) #outp shape (batchNum,chan=1,lrNum,h,w)
        
        #generate Hr image
        meanIm=torch.mean(regIm,dim=2)
        hrIm=meanIm+hfResidual
        hrIm2=meanIm+hfResidual2
        
        if target is None:
            return hfResidual,regIm,hrIm,hrIm2
        else:
            #check and discard samples where LR images have high mask ratio
            if inpMask is not None:
                goodsamp=torch.LongTensor(utils.classifyGoodBadSamples(inpMask,0.8)).to(hrIm.device)
                hrIm=torch.index_select(hrIm,0,goodsamp)
                target=torch.index_select(target,0,goodsamp)
            
            #loss from pixel similarity 
            if self.findBestLossFlag:
                hrIm  ,target1,pixLoss  =findBestLoss(hrIm,target,self.lossCriterion)
                hrIm2,target2,pixLoss2=findBestLoss(hrIm2,target,self.lossCriterion)
            else:
                #Note: Obsolete
                pixLoss=torch.mean(self.lossCriterion(hrIm,target))    
            
            #loss from structural similarity index
            ssi=utils.structuralSimilarityGrayScale(hrIm,target1)
            ssimLoss1=torch.mean(1.-ssi)
            ssi=utils.structuralSimilarityGrayScale(hrIm2,target2)
            ssimLoss2=torch.mean(1.-ssi)
            
            
            
            
            print("loss: %.5f  %.5f"%(pixLoss,pixLoss2))
            loss=pixLoss+2*pixLoss2+(ssimLoss1+ssimLoss2)*2e-3
            
            return [hfResidual,regIm,hrIm,hrIm2],loss
            

