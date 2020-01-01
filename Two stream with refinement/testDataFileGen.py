import torch
from dataset import ListDataset
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testDataPath', type=str, default='D:/dataset/probav_data/train/NIRtest',
                        help='Training dataset path')
    parser.add_argument('--normCsvFile', type=str, default='D:/dataset/probav_data/norm.csv',
                        help='Location of norm.csv file')
    
    parser.add_argument('--outImgSize', type=int, default=384,
                        help='Output image size. Output an Hr image')
    parser.add_argument('--inImgPatchSize', type=int, default=64,
                        help='Input image patch size')
    parser.add_argument('--batchSize', type=int, default=4,
                        help='Batch size')
    
    args = parser.parse_args()
    return args

if __name__=='__main__':
     options=parse_args()
     testDataset=ListDataset(options.testDataPath,options)
     
     with open(options.normCsvFile) as fil:
         strr=fil.read()
     strL=strr.split("\n")
     dic={f.split(" ")[0]:float(f.split(" ")[1]) for f in strL }
     
     
     allSceneList=[]
     allNormList=[]
     for fInd in range(len(testDataset)):
         imO=testDataset[fInd]
         print(imO.sceneDir.split("\\")[-1])
         norm=dic[imO.sceneDir.split("\\")[-1]]
         _a,_b=imO()
         Hr=torch.cat([_a,_b],dim=0).unsqueeze(0)
         if torch.mean(_b)<=0.8: continue
         c=torch.Tensor([torch.mean(f[1]) for f in imO ])
         cArg=(-c).sort()[1]
         lrList=[]
         for f in range(9):
             lrList.append(torch.cat(imO[cArg[f]],dim=0))
         lrList=torch.stack(lrList,dim=0) #lrNum,2=im and mask,h,w
         sceneList=List=torch.cat([Hr,lrList],dim=0)
         allSceneList.append(sceneList)
         allNormList.append(norm)
     allSceneList=torch.stack(allSceneList,dim=0)#sceneNum,lrNum,2=im and mask,h,w
     allNormList=torch.Tensor(allNormList)
     
     torch.save(allSceneList,"testDat")
     torch.save(allNormList,"testNormDat")
         
             
         
         