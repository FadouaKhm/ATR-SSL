import os
import numpy as np
import pandas as pd
from pdb import set_trace
from utils.arf import arf_open
from collections import OrderedDict
from config import *
import utils.atlas_scorer
from utils.atlas_scorer.models import AtlasDecl, BBOX, Declaration, FrameDeclaration
from utils.atlas_scorer.schemas import AtlasDeclSchema
from utils.atlas_scorer.score import Scorer



def GenInputFileList(FileTS):
    InputFileList=[]
    for ifel in FileTS:
        if ifel[-4:]=='.txt':
            with open(ifel) as f:
                extra_lines=f.read().splitlines()
                InputFileList.extend(extra_lines)
        else:
            InputFileList.append(ifel)

    return InputFileList

def GenInputLines_NoGtAnno(InputFileSet,AllFrames=False, SampleRate=1):
    InputLines=[]
    
    for FileName in InputFileSet:
        arf_file=os.path.join(DataRoot,FileName)+'.arf'
        o = arf_open(arf_file)
        InputLines.extend([ [arf_file,k,[]] for k in range(0,o.num_frames,SampleRate)])
    return InputLines

def GenInputLines_(InputFileSet,AllFrames=False,use_json=False):
    InputFileList=[]
    FLSampRate=[]
    pcount=-1
    if use_json:
        scorer=Scorer()

    #EntityList=['COMMERCIAL','MILITARY']
    EntityList={'PICKUP':'COMMERCIAL','SUV':'COMMERCIAL', \
            'BTR70':'MILITARY','BRDM2':'MILITARY','BMP2':'MILITARY','T72':'MILITARY','ZSU23-4':'MILITARY', \
            '2S3':'MILITARY','MTLB':'MILITARY','D20':'MILITARY'};
    DataRoot = config["data_config"]['root']+r'/'  
    #"D:/Nada/ATR project/ATR Database/"
    TargetRangeLimit = 4000
    Prescreener=False
    
    EntityNameDict=dict(zip(EntityList,range(len(EntityList))))
    for ifel in InputFileSet:
        if ',' in ifel:
            NSR=int(ifel[ifel.find(',')+1:])
            ifel=ifel[:ifel.find(',')]
        else:
            NSR=1
        InputFileList.append(ifel)
        FLSampRate.append(NSR)
    InputLines=[]
    for FileName,NSR in zip(InputFileList,FLSampRate):
        Ann=OrderedDict()
        if use_json:
            truth_file=DataRoot+FileName.replace('/arf/','/json_truth/')+'.truth.json'
            truth_obj = scorer.load_truth_file(truth_file,return_dict=False)
            df_Labels=scorer.annotation_objects_to_table_expanded(truth_obj)
            df_Labels=df_Labels.rename(columns={'class':'TgtType'})
        else:
            FilePath=os.path.join(DataRoot,FileName.replace('/arf/','/labels/'))+'.csv'
            print(FilePath)        
            df_Labels=pd.read_csv(FilePath,names=CSVHeader)
        for i in range(0,df_Labels.shape[0]):
            Frame=df_Labels['frame'][i]
            if str(Frame) not in Ann:
                Ann[str(Frame)]=[]
            TgtType=df_Labels['TgtType'][i]
            TgtRange=df_Labels['range'][i]
            if (TgtType not in EntityNameDict.keys()) or (TgtRange>TargetRangeLimit):
                continue                
            if use_json:
                curbb=df_Labels['bbox'][i]
                x_min=int(curbb.x)
                y_min=int(curbb.y)
                x_max=int(curbb.x+curbb.w)
                y_max=int(curbb.y+curbb.h)                
            else:
                x_min=df_Labels['upper_left_x'][i]
                y_min=df_Labels['upper_left_y'][i]

                x_center=df_Labels['CenterX'][i]
                y_center=df_Labels['CenterY'][i]

                x_dim=x_center-x_min
                y_dim=y_center-y_min

                x_min=int(x_center-1*x_dim)
                y_min=int(y_center-1*y_dim)

                x_max=int(x_center+1*x_dim)
                y_max=int(y_center+1*y_dim)
            if (x_max-x_min<1) or (y_max-y_min<1) or (x_max>639) or (x_min<2) or (y_max>511) or (y_max<2):
                continue
            Ann[str(Frame)].append([x_min,y_min,x_max,y_max,
                EntityNameDict[TgtType] if Prescreener==False else 0])                   
        if AllFrames:
            minframe=10**7
            maxframe=-10**7
            fo = arf_open(os.path.join(DataRoot,FileName)+'.arf')
            for key, value in Ann.items():
                if value:
                    minframe=min(minframe,int(key))
                    maxframe=max(maxframe,int(key))
            for cfi in range(minframe,maxframe):
                cur_frame=str(cfi+1)
                if cur_frame in Ann:
                    cur_anno=Ann[cur_frame]
                else:
                    cur_anno=[]
                AnnList=[os.path.join(DataRoot,FileName)+'.arf',int(cur_frame)-1,cur_anno]
                InputLines.append(AnnList)
        else:
            for Frame in Ann:
                
                if (len(Ann[Frame])==0) or ((int(Frame)-1)%NSR > 0):
                    continue
                AnnList=[os.path.join(DataRoot,FileName)+'.arf',int(Frame)-1,Ann[str(Frame)]]
                InputLines.append(AnnList)

    return InputLines
