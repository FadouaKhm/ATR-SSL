from utils.arf import arf_open
import os
from collections import OrderedDict
from utils.atlas_scorer.score import Scorer
from config import config

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

def GenInputLines(InputFileSet, DataRoot,  ylim_max, xlim_max):
    InputFileList = []
    FLSampRate = []
    scorer = Scorer()
    TargetRangeLimit = config['data_config']["range_limit"]
    EntityNameDict = config["ClassMapping"]
    for ifel in InputFileSet:
        if ',' in ifel:
            NSR=int(ifel[ifel.find(',')+1:])
            ifel=ifel[:ifel.find(',')]
        else:
            NSR=1
        InputFileList.append(ifel)
        FLSampRate.append(NSR)
    InputLines = []
    for FileName,NSR in zip(InputFileList,FLSampRate):
        Ann = OrderedDict()
        truth_file = os.path.join(DataRoot, FileName.replace('/arf/','/json_truth/')+'.truth.json')
        truth_obj = scorer.load_truth_file(truth_file,return_dict=False)
        df_Labels = scorer.annotation_objects_to_table_expanded(truth_obj)
        df_Labels = df_Labels.rename(columns={'class':'TgtType'})

        for i in range(0,df_Labels.shape[0]):
            Frame = df_Labels['frame'][i]
            if str(Frame) not in Ann:
                Ann[str(Frame)] = []
            TgtType = df_Labels['TgtType'][i]
            TgtRange = df_Labels['range'][i]
            if  (TgtRange>TargetRangeLimit):
                continue
            elif (TgtType not in EntityNameDict.keys()):
                TgtType="NT"
            curbb = df_Labels['bbox'][i]
            x_min = int(curbb.x)
            y_min = int(curbb.y)
            x_max = int(curbb.x+curbb.w)
            y_max = int(curbb.y+curbb.h)

            if (x_max-x_min<1) or (y_max-y_min<1) or (x_max>xlim_max) or (x_min<2) or (y_max>ylim_max) or (y_max<2):
                continue
            Ann[str(Frame)].append([x_min,y_min,x_max,y_max, TgtType])
        for Frame in Ann:
            if (len(Ann[Frame]) == 0) or ((int(Frame)-1)%NSR > 0):
                continue
            AnnList = [os.path.join(DataRoot, FileName)+'.arf', int(Frame)-1, Ann[str(Frame)]]
            InputLines.append(AnnList)
    return InputLines
