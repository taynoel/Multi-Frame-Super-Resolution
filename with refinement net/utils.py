import torch

def structuralSimilarityGrayScale(imgSet1,imgSet2,windowSize=8,stride=8):
        """
        Perform structural similarity index measurement between gray-scale image set imgSet1 and imgSet2
        Each input tensor (imgSet1,imgSet2) has a shape of (batchNum,chan=1,h,w)
        outputs ssim as (batchNum) tensor 
        """
        imgSet=torch.cat([imgSet1,imgSet2],dim=1)#(batch,chan=2,h,w)
        _h,_w=imgSet.shape[2],imgSet.shape[3]
        whList=[f*stride for f in range(_h) if f*stride+windowSize<_h]
        wwList=[f*stride for f in range(_w) if f*stride+windowSize<_w]
        newBatch=[]
        for fSample in imgSet:
            for fh in whList:
                for fw in wwList:
                    newBatch.append(fSample[:,fh:fh+windowSize,fw:fw+windowSize])
        newBatch=torch.stack(newBatch,0)
        newBatch=newBatch.view(newBatch.shape[0],newBatch.shape[1],-1) #ex: 13520,2,64
        newBatchMean=torch.mean(newBatch,dim=2)
        newBatchVar=torch.var(newBatch,dim=2)
        
        #Calc Covariance
        newBatchCent=newBatch-newBatchMean.unsqueeze(2)#ex: 13520,2,64
        newBatchCov=torch.mean(newBatchCent[:,0,:]*newBatchCent[:,1,:],dim=1).unsqueeze(1)
        
        #Calc SSIM
        #(2*u1.u2+c1)(2cov+c2)/[ (u1^2+u2^2 +c1)(a1^2+a2^2 +c2)]
        c1,c2=0.0001,0.001
        u1u2=2*newBatchMean[:,:1]*newBatchMean[:,1:]+c1
        covv=2*newBatchCov+c2
        u1squ2sq=newBatchMean[:,:1]*newBatchMean[:,:1]+newBatchMean[:,1:]*newBatchMean[:,1:]+c1
        a1sqa2sq=newBatchVar[:,:1]+newBatchVar[:,1:]+c2
        ssim=u1u2*covv/(u1squ2sq*a1sqa2sq)
        
        return ssim

def classifyGoodBadSamples(inpMask,thres=0):
    """
    input tensor has shape (bacthNum,numImg,h,w)
    """
    inpMask=inpMask.view(inpMask.shape[0],-1)
    ratio=torch.mean(inpMask,dim=1)
    decis=ratio>thres
    indexDescend=torch.argsort(-ratio)
    if torch.sum(decis)>0.5:#if there are good samples
        return [ind for ind,val in enumerate(decis) if val]
    else:#else,return best one
        return [indexDescend[0].item()]
        